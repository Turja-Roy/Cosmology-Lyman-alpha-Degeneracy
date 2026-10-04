"""
Observed QSO spectra and the forward model that makes simulated spectra look like them.

An observed spectrum is continuum-normalised flux on a wavelength grid, with noise, a
non-zero instrumental width and stretches that carry no usable Lyman-alpha signal. A
simulated sightline is noiseless optical depth on a velocity grid. Nothing can be compared
until both sides sit in the same space, so the simulation is degraded rather than the
observation repaired:

    tau -> UVB rescale -> exp(-tau) -> LSF convolution -> obs pixel grid -> noise -> mask

The statistics here are the fitting-free ones: per-chunk effective optical depth, the pixel
flux PDF, rest-frame equivalent widths and the lengths of the transmitted gaps between
absorbers. A chunk is one box length of velocity, so one chunk of an observed spectrum and
one simulated sightline cover the same path and their distributions can be compared.

Run:
    python scripts/observations.py --self-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np

C_KMS = 299792.458
LYA = 1215.67

# Galactic ISM lines seen in the COS band. Observed at z=0, so these are fixed wavelengths
# in every spectrum and can be masked without knowing anything about the sightline.
MW_LINES = [1190.42, 1193.29, 1199.55, 1200.22, 1200.71, 1206.50, 1238.82, 1242.80,
            1250.58, 1253.81, 1259.52, 1260.42, 1302.17, 1304.37, 1317.22, 1334.53,
            1335.71, 1393.76, 1402.77, 1526.71, 1548.20, 1550.78, 1608.45, 1670.79]

# Milky Way Lyman-alpha is damped: its wings swallow everything near 1216 A.
MW_LYA_TROUGH = (1208.0, 1226.0)


def load_spectrum(path):
    """Read a three-column spectrum: wavelength [A], normalised flux, 1 sigma error."""
    wave, flux, err = np.loadtxt(path, unpack=True)
    order = np.argsort(wave)
    return wave[order], flux[order], err[order]


def load_cos_list(path):
    """Read the collaborator's sightline table.

    Each row is: name, z_em, the overall usable range, then up to seven sub-ranges, then a
    trailing pair. The sub-ranges are her masking: for ton580 they exclude 1258.7-1261.0
    (Si II 1260), 1283.6-1307.5 (O I 1302 + Si II 1304), 1333.8-1336.0 (C II 1334) and
    1525.9-1527.2 (Si II 1526), and start redward of the damped Galactic Lyman-alpha. That is
    per sightline and supersedes the generic MW_LINES list. Zero pairs are padding.

    The trailing pair sits inside a usable sub-range and carries ordinary data and errors, so
    it is not a detector gap; its purpose is unconfirmed (possibly where S/N was measured) and
    it is returned as `extra` without being applied.
    """
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            values = [float(p) for p in parts[1:]]
        except ValueError:
            continue
        z_em, bounds = values[0], values[1:]
        pairs = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2)]
        windows = [(lo, hi) for lo, hi in pairs[1:-1] if hi > lo > 0]
        out[name] = {'z_em': z_em, 'range': pairs[0] if pairs else None,
                     'windows': windows, 'extra': pairs[-1] if len(pairs) > 1 else None}
    return out


def mask_from_windows(wave, err, windows, z_em, proximity=3000.0, z_min=0.0):
    """Usable Lyman-alpha pixels given explicit wavelength windows.

    The windows already exclude the Galactic lines, so only the zero-error pixels, the
    Lyman-alpha range itself and the quasar's proximity zone still have to be cut.
    """
    good = np.zeros_like(wave, dtype=bool)
    for lo, hi in windows:
        good |= (wave >= lo) & (wave <= hi)
    good &= err > 0
    good &= wave >= LYA * (1 + z_min)
    good &= wave <= LYA * (1 + z_em) * (1 - proximity / C_KMS)
    return good


def usable_mask(wave, err, z_em, mw_halfwidth=100.0, proximity=3000.0, z_min=0.0):
    """Pixels that carry Lyman-alpha forest signal.

    Drops zero-error pixels (airglow, filled in by the reduction), the Galactic lines, the
    damped Milky Way Lyman-alpha trough, everything blueward of the lowest usable redshift
    and everything within `proximity` km/s of the quasar, where its own radiation thins the
    gas.
    """
    good = np.isfinite(flux_safe(err)) & (err > 0)
    for line in MW_LINES:
        good &= np.abs(wave - line) / line * C_KMS > mw_halfwidth
    good &= (wave < MW_LYA_TROUGH[0]) | (wave > MW_LYA_TROUGH[1])
    good &= wave >= LYA * (1 + z_min)
    good &= wave <= LYA * (1 + z_em) * (1 - proximity / C_KMS)
    return good


def flux_safe(x):
    return np.where(np.isfinite(x), x, np.nan)


def velocity_width(wave):
    """Width of each pixel in km/s. The grid is uniform in wavelength, not in velocity."""
    edges = np.concatenate(([wave[0] - 0.5 * (wave[1] - wave[0])],
                            0.5 * (wave[:-1] + wave[1:]),
                            [wave[-1] + 0.5 * (wave[-1] - wave[-2])]))
    return np.diff(edges) / wave * C_KMS


def chunk_slices(wave, good, dv_chunk, min_good_frac=0.5):
    """Split the usable spectrum into runs of `dv_chunk` km/s.

    Chunks are laid end to end from the blue edge; one that has lost too much path to
    masking is dropped rather than compared against a full simulated sightline.
    """
    dv = velocity_width(wave)
    out = []
    start = 0
    while start < len(wave):
        cum = np.cumsum(dv[start:])
        end = start + int(np.searchsorted(cum, dv_chunk)) + 1
        if end > len(wave):
            break
        sl = slice(start, end)
        if good[sl].sum() >= min_good_frac * (end - start):
            out.append(sl)
        start = end
    return out


def tau_eff(flux, good):
    """Effective optical depth, -ln<F>, over the unmasked pixels.

    A mean flux at or below zero is not a measurement of anything, so it returns NaN rather
    than a floored stand-in: a clamped value looks like data and propagates into medians and
    percentiles as if it were one.
    """
    f = flux[good]
    if f.size == 0:
        return np.nan
    mean = np.mean(f)
    return -np.log(mean) if mean > 0 else np.nan


def flux_pdf(flux, good, bins=np.linspace(-0.1, 1.3, 29)):
    """Pixel flux histogram. Noise pushes flux outside [0, 1], so the range runs wider."""
    counts, edges = np.histogram(flux[good], bins=bins)
    return counts / max(counts.sum(), 1) / np.diff(edges), edges


def find_absorbers(wave, flux, err, good, n_sigma=2.0, min_pixels=2, ew_sigma=3.0):
    """Contiguous runs of absorbed pixels whose equivalent width is itself significant.

    A per-pixel cut alone turns noise into absorbers at the rate the cut implies, which at
    two sigma is one pixel in forty. The feature as a whole has to clear `ew_sigma`.
    """
    absorbed = good & (flux < 1.0 - n_sigma * err)
    runs, start = [], None
    for i, a in enumerate(np.append(absorbed, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_pixels:
                runs.append(slice(start, i))
            start = None
    dlam = np.gradient(wave)
    keep = []
    for sl in runs:
        ew = np.sum((1.0 - flux[sl]) * dlam[sl])
        ew_err = np.sqrt(np.sum((err[sl] * dlam[sl]) ** 2))
        if ew_err > 0 and ew / ew_err >= ew_sigma:
            keep.append(sl)
    return keep


def absorber_properties(wave, flux, err, good, n_sigma=2.0, min_pixels=2):
    """One record per detected absorber: EW, redshift, velocity width, depth, S/N.

    These are the fitting-free stand-ins for a Voigt catalogue. EW against width is the
    plane the b-N diagram would occupy, and the minimum flux shows saturation, which the
    LSF pushes upwards by filling in line cores.
    """
    dlam = np.gradient(wave)
    dv = velocity_width(wave)
    out = []
    for sl in find_absorbers(wave, flux, err, good, n_sigma, min_pixels):
        w, depth = wave[sl], 1.0 - flux[sl]
        centre = np.average(w, weights=np.clip(depth, 0, None) + 1e-12)
        z = centre / LYA - 1.0
        ew_obs = np.sum(depth * dlam[sl])
        ew_err = np.sqrt(np.sum((err[sl] * dlam[sl]) ** 2))
        out.append({
            'ew': 1e3 * ew_obs / (1 + z),          # rest-frame, mA
            'z': z,
            'width': float(dv[sl].sum()),          # km/s of absorbed pixels
            'depth': float(1.0 - np.min(flux[sl])),
            'ew_snr': float(ew_obs / ew_err) if ew_err > 0 else np.nan,
        })
    return out


def equivalent_widths(wave, flux, err, good, n_sigma=2.0, min_pixels=2):
    """Rest-frame equivalent widths [mA] and redshifts, as plain arrays."""
    recs = absorber_properties(wave, flux, err, good, n_sigma, min_pixels)
    return (np.array([r['ew'] for r in recs]), np.array([r['z'] for r in recs]))


def gap_lengths(wave, flux, err, good, n_sigma=2.0, min_pixels=2):
    """Velocity lengths [km/s] of the transmitted stretches between absorbers."""
    runs = find_absorbers(wave, flux, err, good, n_sigma, min_pixels)
    dv = velocity_width(wave)
    edges = [0] + [i for sl in runs for i in (sl.start, sl.stop)] + [len(wave)]
    gaps = [dv[a:b][good[a:b]].sum() for a, b in zip(edges[::2], edges[1::2]) if b > a]
    return np.array([g for g in gaps if g > 0])


def chunk_statistics(wave, flux, err, good, dv_chunk, min_good_frac=0.5):
    """Per-chunk scalars. One row per chunk, to be compared against one simulated sightline."""
    rows = []
    for sl in chunk_slices(wave, good, dv_chunk, min_good_frac):
        w, f, e, g = wave[sl], flux[sl], err[sl], good[sl]
        ew, _ = equivalent_widths(w, f, e, g)
        gaps = gap_lengths(w, f, e, g)
        rows.append({
            'lambda_mid': 0.5 * (w[0] + w[-1]),
            'z_mid': 0.5 * (w[0] + w[-1]) / LYA - 1.0,
            'mean_flux': np.mean(f[g]),
            'tau_eff': tau_eff(f, g),
            'n_absorbers': len(ew),
            'ew_median': np.median(ew) if len(ew) else np.nan,
            'ew_sum': ew.sum(),
            'gap_max': gaps.max() if len(gaps) else np.nan,
            'good_frac': g.mean(),
        })
    return rows


# ------------------------------------------------------------------ forward model


def convolve_kernel(flux, dv_pixel, kernel_weights, kernel_dv):
    """Convolve with a tabulated kernel given on its own velocity spacing.

    The COS LSF is tabulated per native detector pixel, not per simulation pixel, so the
    kernel is resampled onto the simulation's grid before use. Its wings are what a Gaussian
    of the same width misses, and they are the reason weak lines blend.
    """
    w = np.asarray(kernel_weights, dtype=float)
    centre = (len(w) - 1) / 2.0
    v_kernel = (np.arange(len(w)) - centre) * kernel_dv
    half = np.ceil(np.abs(v_kernel).max() / dv_pixel)
    v_out = np.arange(-half, half + 1) * dv_pixel
    k = np.interp(v_out, v_kernel, w, left=0.0, right=0.0)
    if k.sum() <= 0:
        return flux
    k /= k.sum()
    pad = len(k) // 2
    padded = np.concatenate([flux[-pad:], flux, flux[:pad]])  # sightlines are periodic
    return np.convolve(padded, k, mode='same')[pad:-pad]


def gaussian_lsf(flux, dv_pixel, fwhm_kms):
    """Convolve a spectrum sampled uniformly in velocity with a Gaussian of `fwhm_kms`.

    The real COS LSF has broad non-Gaussian wings from the mirror figure; this underfills
    them and so under-predicts blending. Adequate while the statistics stay fitting-free.
    """
    if fwhm_kms <= 0:
        return flux
    sigma_pix = fwhm_kms / 2.3548200 / dv_pixel
    half = max(int(np.ceil(4 * sigma_pix)), 1)
    x = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([flux[-half:], flux, flux[:half]])  # sightlines are periodic
    return np.convolve(padded, kernel, mode='same')[half:-half]


def rebin(wave_in, flux_in, wave_out):
    """Average flux into the output pixels, conserving the integral.

    Interpolating the cumulative integral and differencing it is the flux-conserving way;
    sampling would throw away the sub-pixel structure the convolution just created.
    """
    edges_in = np.concatenate(([wave_in[0]], 0.5 * (wave_in[:-1] + wave_in[1:]), [wave_in[-1]]))
    edges_out = np.concatenate(([wave_out[0] - 0.5 * (wave_out[1] - wave_out[0])],
                                0.5 * (wave_out[:-1] + wave_out[1:]),
                                [wave_out[-1] + 0.5 * (wave_out[-1] - wave_out[-2])]))
    cum = np.concatenate(([0.0], np.cumsum(flux_in * np.diff(edges_in))))
    integral = np.interp(edges_out, edges_in, cum)
    return np.diff(integral) / np.diff(edges_out)


def pixels_per_resel(wave, fwhm_kms):
    return fwhm_kms / np.median(velocity_width(wave))


def noise_sigma(flux, wave, snr_resel, fwhm_kms, floor=0.05):
    """Per-pixel 1 sigma for a target signal-to-noise per resolution element.

    S/N per resolution element is the observers' unit; per pixel it is lower by the square
    root of the number of pixels a resolution element covers. Counting statistics make the
    error shrink with flux, so sigma scales as sqrt(F) with a floor for saturated cores.
    """
    if not np.isfinite(snr_resel) or snr_resel <= 0:
        return np.zeros_like(flux)
    sigma_cont = np.sqrt(pixels_per_resel(wave, fwhm_kms)) / snr_resel
    return sigma_cont * np.sqrt(np.clip(flux, floor, None))


def forward_model(tau, dv_pixel, z, wave_out, snr_resel=np.inf, fwhm_kms=17.0,
                  uvb_factor=1.0, continuum_error=0.0, sigma=None, kernel=None, rng=None):
    """Turn one simulated sightline into a mock observed spectrum on `wave_out`.

    `tau` is optical depth on a grid uniform in velocity with spacing `dv_pixel`, placed at
    redshift `z`. `uvb_factor` scales tau, since tau ~ 1/Gamma_HI and the low-redshift UV
    background is uncertain by about a factor of two. `sigma` takes the error array of a real
    spectrum and overrides `snr_resel`, which is how a mock inherits the observed noise
    instead of an idealised one. Returns (flux, sigma); sigma is the error array a reduction
    would report, so mocks and data carry the same columns.
    """
    rng = np.random.default_rng() if rng is None else rng
    flux = np.exp(-uvb_factor * np.asarray(tau, dtype=float))
    if kernel is not None:
        flux = convolve_kernel(flux, dv_pixel, kernel[0], kernel[1])
    else:
        flux = gaussian_lsf(flux, dv_pixel, fwhm_kms)

    # The box is a velocity interval; hang it at z, centred on the output window. A window
    # longer than the box is covered by repeating the sightline, which is periodic anyway --
    # otherwise the pixels past the box edge would fall outside the input grid.
    centre = 0.5 * (wave_out[0] + wave_out[-1])
    span_kms = (wave_out[-1] - wave_out[0]) / centre * C_KMS
    n_tile = int(np.ceil(span_kms / (len(flux) * dv_pixel))) + 1
    n_tile += 1 - n_tile % 2   # odd, so the middle copy stays centred on the window
    flux = np.tile(flux, n_tile)
    v = (np.arange(len(flux)) - 0.5 * len(flux)) * dv_pixel
    wave_in = centre * (1 + v / C_KMS)
    flux = rebin(wave_in, flux, wave_out)

    sigma = np.asarray(sigma, dtype=float) if sigma is not None else \
        noise_sigma(flux, wave_out, snr_resel, fwhm_kms)
    flux = flux + rng.normal(0.0, 1.0, flux.shape) * sigma
    if continuum_error > 0:
        phase = rng.uniform(0, 2 * np.pi)
        span = wave_out[-1] - wave_out[0]
        flux *= 1.0 + continuum_error * np.sin(2 * np.pi * (wave_out - wave_out[0]) / span + phase)
    return flux, np.where(sigma > 0, sigma, 1e-3)


def box_velocity_length(box_size_ckpc_h, z, h=0.6774, omega_m=0.3089):
    """Velocity length of the simulation box at redshift z, i.e. one chunk."""
    e_z = np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))
    return box_size_ckpc_h * 1e-3 / h * 100.0 * h * e_z / (1 + z)


# ------------------------------------------------------------------ self-test


def self_test():
    rng = np.random.default_rng(7)

    # Convolution conserves equivalent width and widens a narrow line to the stated FWHM.
    dv = 1.0
    v = (np.arange(4000) - 2000) * dv
    tau_line = 2.0 * np.exp(-0.5 * (v / 8.0) ** 2)
    sharp = np.exp(-tau_line)
    smeared = gaussian_lsf(sharp, dv, 30.0)
    assert abs(np.sum(1 - smeared) - np.sum(1 - sharp)) < 1e-6 * np.sum(1 - sharp)
    width = lambda f: np.sum(1 - f) / (1 - f.min())
    assert width(smeared) > 1.4 * width(sharp)
    assert smeared.min() > sharp.min()  # core filled in, the reason weak lines get lost

    # A tabulated kernel that happens to be Gaussian reproduces the Gaussian path.
    k_dv = 3.0
    k_v = (np.arange(41) - 20) * k_dv
    k_w = np.exp(-0.5 * (k_v / (30.0 / 2.35482)) ** 2)
    tabulated = convolve_kernel(sharp, dv, k_w, k_dv)
    assert np.max(np.abs(tabulated - smeared)) < 0.02, np.max(np.abs(tabulated - smeared))

    # Rebinning conserves the integral.
    wave_in = np.linspace(1300, 1310, 4001)
    flux_in = 1 - 0.4 * np.exp(-0.5 * ((wave_in - 1305) / 0.05) ** 2)
    wave_out = np.linspace(1300.05, 1309.95, 271)
    flux_out = rebin(wave_in, flux_in, wave_out)
    ew_in = np.sum((1 - flux_in) * np.gradient(wave_in))
    ew_out = np.sum((1 - flux_out) * np.gradient(wave_out))
    assert abs(ew_out - ew_in) < 0.01 * ew_in, (ew_in, ew_out)

    # tau_eff of a uniform half-transmitted spectrum.
    f = np.full(100, 0.5)
    assert abs(tau_eff(f, np.ones(100, bool)) - np.log(2)) < 1e-12

    # Requested S/N per resolution element comes back out of a flat mock.
    wave_obs = np.arange(1300.0, 1310.0, 0.0367)
    flat, sigma = forward_model(np.zeros(20000), 0.5, 0.075, wave_obs, snr_resel=50.0, rng=rng)
    measured = np.sqrt(pixels_per_resel(wave_obs, 17.0)) / np.std(flat)
    assert 0.8 < measured / 50.0 < 1.2, measured

    # Equivalent width survives the round trip through the forward model.
    tau_sim = 1.5 * np.exp(-0.5 * (((np.arange(20000) - 10000) * 0.5) / 15.0) ** 2)
    mock, sig = forward_model(tau_sim, 0.5, 0.075, wave_obs, snr_resel=np.inf, rng=rng)
    good = np.ones_like(mock, bool)
    ew, _ = equivalent_widths(wave_obs, mock, np.full_like(mock, 1e-3), good)
    lam_pix = LYA * (1 + 0.075) * 0.5 / C_KMS
    ew_true = 1e3 * np.sum(1 - np.exp(-tau_sim)) * lam_pix / 1.075
    assert abs(ew.sum() - ew_true) < 0.05 * ew_true, (ew.sum(), ew_true)

    # Masking removes the Galactic lines and the airglow-filled pixels.
    wave = np.arange(1135.0, 1500.0, 0.0367)
    err = np.full_like(wave, 0.1)
    err[np.abs(wave - 1215.67) < 0.5] = 0.0
    good = usable_mask(wave, err, z_em=0.167)
    assert not good[np.argmin(np.abs(wave - 1206.50))]      # Si III
    assert not good[np.argmin(np.abs(wave - 1215.67))]      # airglow
    assert not good[np.argmin(np.abs(wave - 1412.0))]       # inside the proximity zone
    assert good[np.argmin(np.abs(wave - 1300.0))]

    # Chunks are one box length long and none of them overlap.
    dv_chunk = box_velocity_length(25000.0, 0.075)
    assert 2000 < dv_chunk < 3000, dv_chunk
    flux = np.ones_like(wave)
    slices = chunk_slices(wave, good, dv_chunk)
    assert slices and all(a.stop <= b.start for a, b in zip(slices, slices[1:]))
    rows = chunk_statistics(wave, flux, err, good, dv_chunk)
    assert all(abs(r['tau_eff']) < 1e-9 for r in rows)

    # cos.list parsing: padding pairs dropped, windows kept, z_em read.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'cos.list'
        p.write_text('ton580 0.29010 1219.0 1542.0  1219.0 1258.7  1261.0 1283.6  '
                     '0 0  0 0  0 0  0 0  1423.3 1426.8\n'
                     'bad_row 0.1\n')
        table = load_cos_list(p)
        assert set(table) == {'ton580'}, table
        row = table['ton580']
        assert abs(row['z_em'] - 0.2901) < 1e-9
        assert row['windows'] == [(1219.0, 1258.7), (1261.0, 1283.6)], row['windows']
        assert row['extra'] == (1423.3, 1426.8)

        w = np.arange(1200.0, 1500.0, 0.05)
        e = np.full_like(w, 0.1)
        g = mask_from_windows(w, e, row['windows'], row['z_em'])
        assert g[np.argmin(np.abs(w - 1240.0))]
        assert not g[np.argmin(np.abs(w - 1260.0))]   # between her windows
        assert not g[np.argmin(np.abs(w - 1210.0))]   # blueward of the first window

    print(f"observations self-test passed ({len(slices)} chunks of {dv_chunk:.0f} km/s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spectrum', nargs='?', help='three-column .dat spectrum')
    parser.add_argument('--z-em', type=float, help='quasar emission redshift')
    parser.add_argument('--box-size', type=float, default=25000.0, help='ckpc/h')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.spectrum or args.z_em is None:
        parser.error('need a spectrum and --z-em (or --self-test)')

    wave, flux, err = load_spectrum(args.spectrum)
    good = usable_mask(wave, err, args.z_em)
    z_mid = np.median(wave[good]) / LYA - 1.0
    dv_chunk = box_velocity_length(args.box_size, z_mid)
    rows = chunk_statistics(wave, flux, err, good, dv_chunk)
    ew, _ = equivalent_widths(wave, flux, err, good)

    path_dz = np.sum(np.gradient(wave)[good]) / LYA
    print(f"{args.spectrum}: {good.sum()} / {len(wave)} usable pixels, dz ~ {path_dz:.3f}")
    print(f"chunk length {dv_chunk:.0f} km/s at z = {z_mid:.3f}, {len(rows)} chunks")
    print(f"tau_eff per chunk: {np.nanmean([r['tau_eff'] for r in rows]):.4f} "
          f"+- {np.nanstd([r['tau_eff'] for r in rows]):.4f}")
    print(f"{len(ew)} absorbers, median rest EW {np.median(ew):.0f} mA")
    return 0


if __name__ == '__main__':
    sys.exit(main())
