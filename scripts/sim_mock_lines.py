"""
Turn simulated sightlines into mock COS spectra and measure lines in them.

The simulated line catalogue we have was measured on noiseless optical depth at 0.1 km/s
resolution. No real spectrum looks like that, so comparing it to an observed catalogue
compares two different measurements rather than two universes. This script puts the
simulation through what a real sightline goes through -- the COS line spread function, a real
sightline's pixel grid, a real sightline's errors, a real sightline's masked windows -- and
then fits Voigt components to the result with scripts/voigt_fit.py.

Mocks cycle through the observed sightlines that have a VPFIT fit, so the mock sample
inherits the survey's actual mix of noise, wavelength coverage, masking and LSF rather than
one idealised setup.

The LSF is the tabulated STScI COS kernel (data/Obs/lsf/), chosen per pixel from the
grating, cenwave and lifetime position of her nearest fitted region and the nearest
tabulated wavelength. `--lsf gauss` swaps in a Gaussian of `--fwhm` for comparison.

Give it one spectra file per snapshot: every pixel of a mock is drawn from the snapshot
nearest its own redshift, and the path above `--z-max` is masked on both sides.

S/N rungs are per pixel at the continuum; 'data' uses the template's own error array.
Output is one CSV of components per rung, with the same columns as the observed VPFIT
catalogue, and one CSV of per-mock summaries including the pixel flux histogram.

    python scripts/sim_mock_lines.py \\
        spectra/IllustrisTNG/EX/EX_0/camel_lya_lya_h_spectra_snap_0{76,78,80,82,84,86,88,90}_n10000.hdf5 \\
        --n 255 --snr data --snr 10 --workers 48 --out output/obs/mock_lines/IllustrisTNG/EX_0

    python scripts/sim_mock_lines.py SPECTRA.hdf5 ... --uvb match --out .../uvb_match
    python scripts/sim_mock_lines.py --self-test
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import h5py
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observations as obs                                       # noqa: E402
import voigt_fit                                                 # noqa: E402
from vpfit_catalog import parse_all                              # noqa: E402

FIELDS = ['sightline', 'template', 'mock', 'snr', 'ion', 'ion_key',
          'z', 'z_err', 'b', 'b_err', 'logN', 'logN_err',
          'region_lo', 'region_hi', 'chi2', 'n_pix', 'snap']

SIGHTLINE_FIELDS = ['template', 'mock', 'snr', 'ew_sum', 'ew_median',
                    'n_absorbers', 'tau_eff', 'n_good', 'n_components', 'flux_counts']

# Pixel flux histogram written per mock, so flux PDFs and CDFs are rebuilt without spectra.
FLUX_EDGES = np.linspace(-0.2, 1.4, 161)

# Midpoint of snap-076 (z = 0.3966) and snap-074 (z = 0.4653). Above it the nearest
# generated snapshot is more than half a snapshot spacing away.
Z_MAX = 0.431

# The grid a mock is built and convolved on. COS native pixels are ~2.3 km/s.
DV_GRID = 1.0

# Same sqrt(F) floor as observations.noise_sigma, for saturated cores.
NOISE_FLOOR = 0.05

# Her kernel names, e.g. COS_g130m_L1300_LF1.dat: grating, cenwave, lifetime position. The
# L/U prefix is not explained in the delivery; both map to the same STScI table.
LSF_NAME = re.compile(r'COS_g(1[36]0)m_[LU]?(\d{4})_LF(\d)', re.I)

# Sightline pools, templates and LSF tables are held module-level so forked workers inherit
# them instead of receiving a copy of a multi-hundred-megabyte array with every job.
# _TAU entries: [tau, z, dv, snap label, uvb factor].
_TAU = []
_TEMPLATES = []
_LSF_TABLES = {}


def read_sightlines(path, index, tau_path='tau/H/1/1215'):
    """Read only the requested rows of the tau array.

    The arrays are 10000 x 24999 doubles, about 2 GB each, and reading one whole is what
    exhausted the login node. Fancy indexing on the dataset reads just these rows.
    """
    index = np.unique(np.asarray(index, dtype=int))
    with h5py.File(path, 'r') as f:
        tau = f[tau_path][index, :]
        header = f['Header'].attrs
        z = float(header.get('redshift', header.get('Redshift')))
        dv = float(header['dvbin']) if 'dvbin' in header else None
        if dv is None:
            box = float(header['box']) / 1000.0
            om = float(header.get('omegam', header.get('Omega0', 0.3)))
            hz = 100.0 * np.sqrt(om * (1 + z) ** 3 + (1 - om))
            dv = box * hz / (1 + z) / tau.shape[1]
    return tau, abs(z), dv


def load_templates(pattern, cos_list_path, z_max=Z_MAX):
    """Observed sightlines to borrow grids, errors, masks and LSFs from.

    Only sightlines with a VPFIT fit (`<name>.all`) are used: the fit names the COS kernel of
    every region, and its catalogue is what the mocks are compared against.
    """
    meta = obs.load_cos_list(cos_list_path) if cos_list_path else {}
    out = []
    for path in sorted(glob.glob(pattern)):
        name = Path(path).stem.replace('_all', '')
        fit = Path(path).with_name(f'{name}.all')
        if not fit.exists():
            print(f'skip template {name}: no VPFIT fit')
            continue
        wave, flux, err = obs.load_spectrum(path)
        info = meta.get(name)
        if info and info['windows']:
            good = obs.mask_from_windows(wave, err, info['windows'], info['z_em'])
        elif info:
            good = obs.usable_mask(wave, err, info['z_em'])
        else:
            good = np.isfinite(err) & (err > 0)
        good &= wave <= voigt_fit.LYA * (1 + z_max)
        regions = [r for r in parse_all(fit)[0]
                   if LSF_NAME.search(r['lsf']) and np.isfinite(r['wave_lo'])]
        if good.sum() < 200 or len(regions) < 2:
            print(f'skip template {name}: {good.sum()} usable pixels, {len(regions)} LSF regions')
            continue
        out.append({'name': name, 'fit': fit, 'wave': wave, 'flux': flux, 'err': err,
                    'good': good,
                    'z': float(np.median(wave[good])) / voigt_fit.LYA - 1.0,
                    'snr_pixel': float(np.median(flux[good] / err[good])),
                    'regions': regions})
    return out


def _nearest(sorted_x, values):
    """Index of the nearest element of an ascending array (at least two long)."""
    i = np.clip(np.searchsorted(sorted_x, values), 1, len(sorted_x) - 1)
    return np.where(values - sorted_x[i - 1] < sorted_x[i] - values, i - 1, i)


def lsf_path(lsf_dir, name):
    g, cenwave, lp = LSF_NAME.search(name).groups()
    return Path(lsf_dir) / f'aa_LSFTable_G{g}M_{cenwave}_LP{lp}_cn.dat'


def load_lsf_table(path):
    """(column wavelengths, kernels, native A/pixel). A missing table is an error, not a
    silent fallback to the Gaussian."""
    if path not in _LSF_TABLES:
        waves, kernels = obs.load_cos_lsf(path)
        _LSF_TABLES[path] = (waves, kernels, obs.COS_DISPERSION[path.name.split('_')[2]])
    return _LSF_TABLES[path]


def kernel_lookup(t, lam):
    """(table index, column index) of the COS kernel at each wavelength.

    The table is the one her nearest fitted region used, which follows the G130M/G160M switch
    and each visit's cenwave the way she did; the column is the nearest tabulated wavelength.
    """
    tables = t['lsf_region_table'][_nearest(t['lsf_centres'], lam)]
    cols = np.empty(lam.size, dtype=int)
    for j in np.unique(tables):
        sel = tables == j
        cols[sel] = _nearest(t['lsf_tables'][j][0], lam[sel])
    return tables, cols


def _column(t, j, col):
    """One tabulated kernel and its spacing in km/s at its own wavelength."""
    waves, kernels, disp = t['lsf_tables'][j]
    return kernels[:, col], disp / waves[col] * voigt_fit.C_KMS


def kernel_at(t, lam):
    """Kernel for the fitter at `lam`; (None, None) means the Gaussian of t['fwhm']."""
    if t is None or 'lsf_tables' not in t:
        return None, None
    tables, cols = kernel_lookup(t, np.array([lam]))
    return _column(t, tables[0], cols[0])


def prepare_model_grid(t, lsf_dir=None, fwhm=17.0, z_max=Z_MAX, dv=DV_GRID):
    """The grid a mock is built and convolved on, and which kernel applies at each pixel.

    Uniform in log wavelength at `dv` km/s over the template's forest, padded by more than a
    COS kernel half-width (160 native pixels, < 2 A). Without `lsf_dir` every pixel gets one
    Gaussian of `fwhm`.
    """
    lo = max(t['wave'][0], voigt_fit.LYA) - 3.0
    hi = min(t['wave'][-1], voigt_fit.LYA * (1 + z_max)) + 3.0
    n = int(np.log(hi / lo) * voigt_fit.C_KMS / dv)
    t.update(grid=lo * np.exp((np.arange(n) + 0.5) * dv / voigt_fit.C_KMS), dv=dv, fwhm=fwhm)
    if lsf_dir is None:
        t['kernel_id'] = np.zeros(n, dtype=int)
        t['kernels'] = [voigt_fit.lsf_kernel(dv, fwhm)]
        return t

    regions = sorted(t['regions'], key=lambda r: r['wave_lo'] + r['wave_hi'])
    paths = [lsf_path(lsf_dir, r['lsf']) for r in regions]
    unique = sorted(set(paths))
    t['lsf_tables'] = [load_lsf_table(p) for p in unique]
    t['lsf_centres'] = np.array([0.5 * (r['wave_lo'] + r['wave_hi']) for r in regions])
    t['lsf_region_table'] = np.array([unique.index(p) for p in paths])

    tables, cols = kernel_lookup(t, t['grid'])
    keys, t['kernel_id'] = np.unique(tables * 1000 + cols, return_inverse=True)
    t['kernels'] = [voigt_fit.lsf_kernel(dv, table=k, table_dv=k_dv)
                    for k, k_dv in (_column(t, key // 1000, key % 1000) for key in keys)]
    return t


def stitch_path(tau, dv_sim, span_kms, rng):
    """Lay independent simulated sightlines end to end to cover `span_kms`.

    An observed sightline spans 0.1-0.5 in redshift; one box is 2500 km/s, about 0.008.
    Repeating one periodic sightline would make the same few absorbers recur twenty times
    over -- inflating the line count and correlating a path that should be independent.
    Drawing a fresh sightline per box length instead gives the path the independent
    structure a real one has.
    """
    n_box = int(np.ceil(span_kms / (tau.shape[1] * dv_sim))) + 2
    picks = rng.integers(0, tau.shape[0], size=n_box)
    return np.concatenate([tau[i] for i in picks])


def snapshot_index(z):
    """Nearest loaded snapshot for each redshift: the slice edges are the midpoints."""
    z_snap = np.array([p[1] for p in _TAU])
    return np.argmin(np.abs(np.asarray(z)[:, None] - z_snap[None, :]), axis=1)


def mock_flux(t, rng):
    """Simulated transmitted flux on the template's model grid, before the LSF.

    Every pixel comes from the snapshot nearest its own redshift, so a path spanning
    z = 0-0.4 crosses several boxes in turn. The stretches are joined before the LSF is
    applied, so nothing breaks at their edges.
    """
    isnap = snapshot_index(t['grid'] / voigt_fit.LYA - 1.0)
    flux = np.empty(t['grid'].size)
    edges = np.flatnonzero(np.diff(isnap)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, isnap.size]):
        tau, _, dv_sim, _, uvb = _TAU[isnap[a]]
        fine = np.exp(-uvb * stitch_path(tau, dv_sim, (b - a) * t['dv'], rng))
        # Flux-conserving average of the fine pixels into each grid pixel.
        cum = np.concatenate(([0.0], np.cumsum(fine))) * dv_sim
        edges_v = np.arange(b - a + 1) * t['dv']
        flux[a:b] = np.diff(np.interp(edges_v, np.arange(cum.size) * dv_sim, cum)) / t['dv']
    return flux


def fitting_windows(wave, flux, sigma, good, n_sigma=2.0, pad_kms=25.0, merge_kms=40.0,
                    ew_sigma=3.0):
    """Where to put fitting regions on a mock, the way an observer places them by eye.

    Contiguous runs of pixels significantly below the continuum are grouped, runs closer
    than `merge_kms` are joined because a blend has to be fitted as one piece, and the group
    is padded so the fit sees continuum on both sides. Distances are in velocity, not pixels,
    because the observed sightlines run from 2.5 to 7.7 km/s per pixel.

    The equivalent width is then tested over the *padded* window. Testing it over only the
    pixels that were selected for being low is circular -- the selection guarantees a deficit,
    so pure noise clears a 3 sigma bar easily, which is how the first pass invented lines.
    """
    hit = good & np.isfinite(flux) & ((1.0 - flux) > n_sigma * sigma)
    if not hit.any():
        return []

    dv = obs.velocity_width(wave)
    flips = np.diff(np.concatenate(([0], hit.astype(int), [0])))
    groups = list(zip(np.flatnonzero(flips == 1), np.flatnonzero(flips == -1)))

    merged = []
    for s, e in groups:
        if merged and np.sum(dv[merged[-1][1]:s]) <= merge_kms:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    dlam = np.gradient(wave)
    out = []
    for s, e in merged:
        lo, hi = s, e
        while lo > 0 and np.sum(dv[lo:s]) < pad_kms:
            lo -= 1
        while hi < wave.size and np.sum(dv[e:hi]) < pad_kms:
            hi += 1
        if hi - lo < 10 or good[lo:hi].mean() <= 0.8:
            continue
        ew = float(np.sum((1.0 - flux[lo:hi]) * dlam[lo:hi]))
        ew_err = float(np.sqrt(np.sum((sigma[lo:hi] * dlam[lo:hi]) ** 2)))
        if ew_err > 0 and ew / ew_err >= ew_sigma:
            out.append((lo, hi))
    return out


def measure_mock(wave, flux, sigma, good, fwhm_kms=17.0, template=None):
    """Voigt components in one mock spectrum, region by region, with the template's LSF."""
    rows = []
    for lo, hi in fitting_windows(wave, flux, sigma, good):
        sel = slice(lo, hi)
        table, table_dv = kernel_at(template, 0.5 * (wave[lo] + wave[hi - 1]))
        params, errs, info = voigt_fit.fit_region(
            wave[sel], flux[sel], sigma[sel], fwhm_kms,
            kernel_table=table, kernel_dv=table_dv)
        for c in voigt_fit.components_from_fit(params, errs):
            rows.append({**c, 'region_lo': float(wave[lo]), 'region_hi': float(wave[hi - 1]),
                         'chi2': info['chi2'], 'n_pix': info['n']})
    return rows


def _one_mock(job):
    """One mock sightline observed the way one real template sightline was. For Pool."""
    itemplate, snr, continuum_error, seed = job
    t = _TEMPLATES[itemplate]
    rng = np.random.default_rng(seed)
    wave, err, good = t['wave'], t['err'], t['good']

    convolved = obs.convolve_varying(mock_flux(t, rng), t['kernel_id'], t['kernels'])
    inner = (wave > t['grid'][0]) & (wave < t['grid'][-1])
    flux = np.ones_like(wave)
    flux[inner] = obs.rebin(t['grid'], convolved, wave[inner])
    sigma = err if snr == 'data' else np.sqrt(np.clip(flux, NOISE_FLOOR, None)) / float(snr)
    flux, sigma = obs.add_noise(flux, wave, sigma, continuum_error, rng)

    rows = measure_mock(wave, flux, sigma, good, t['fwhm'], t)
    snaps = snapshot_index([r['z'] for r in rows]) if rows else []
    for r, k in zip(rows, snaps):
        r.update({'sightline': f"mock{seed}", 'template': t['name'], 'mock': seed,
                  'snr': snr, 'ion': 'H I', 'ion_key': 'HI', 'snap': _TAU[k][3]})

    # Measured off the spectrum rather than off the fit. Summed equivalent width does not
    # depend on how a blend was divided into components, where a count of components does,
    # so the two together separate a decomposition difference from a difference in the gas.
    ew, _ = obs.equivalent_widths(wave, flux, sigma, good)
    counts, _ = np.histogram(flux[good], FLUX_EDGES)
    summary = {'template': t['name'], 'mock': seed, 'snr': snr,
               'ew_sum': float(ew.sum()), 'ew_median': float(np.median(ew)) if len(ew) else np.nan,
               'n_absorbers': int(len(ew)), 'tau_eff': float(obs.tau_eff(flux, good)),
               'n_good': int(good.sum()), 'n_components': len(rows),
               'flux_counts': ' '.join(map(str, counts))}
    return rows, summary


def snap_label(path):
    m = re.search(r'snap_(\d+)', str(path))
    return m.group(1) if m else Path(path).stem


def observed_by_snapshot(templates):
    """Observed path dz and mean flux in each loaded snapshot's redshift slice."""
    n = len(_TAU)
    dz, fsum, npix = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in templates:
        g = t['good']
        k = snapshot_index(t['wave'][g] / voigt_fit.LYA - 1.0)
        dz += np.bincount(k, weights=np.gradient(t['wave'])[g] / voigt_fit.LYA, minlength=n)
        fsum += np.bincount(k, weights=t['flux'][g], minlength=n)
        npix += np.bincount(k, minlength=n)
    return dz, np.where(npix > 0, fsum / np.maximum(npix, 1), np.nan)


def match_uvb(mean_flux_obs, n_rows=500):
    """Per-snapshot factor on tau that reproduces the observed mean flux in its slice.

    tau ~ 1/Gamma_HI, so this is the UV background each box would need. The LSF and the
    rebinning conserve flux and the noise has zero mean, so matching the noiseless mean flux
    of the raw pool is matching the mock.
    """
    for p, f_obs in zip(_TAU, mean_flux_obs):
        if not 0 < f_obs < 1:
            print(f'  snap {p[3]}: no usable observed mean flux, uvb factor stays {p[4]}')
            continue
        tau = p[0][:n_rows]
        target = -np.log(f_obs)
        try:
            p[4] = brentq(lambda a: -np.log(np.mean(np.exp(-a * tau))) - target, 1e-2, 1e2)
        except ValueError:
            print(f'  snap {p[3]}: tau_eff {target:.4f} unreachable, uvb factor stays {p[4]}')
            continue
        print(f'  snap {p[3]}: observed tau_eff {target:.4f} -> uvb factor {p[4]:.3f}')


def run(spectra_paths, templates, n_mocks, snrs, uvb='1.0', continuum_error=0.0, workers=1,
        seed=0, out_dir=None, tau_path='tau/H/1/1215', pool=None):
    rng = np.random.default_rng(seed)
    if isinstance(spectra_paths, (str, Path)):
        spectra_paths = [spectra_paths]

    global _TAU, _TEMPLATES
    _TAU, _TEMPLATES = [], templates
    for path in spectra_paths:
        with h5py.File(path, 'r') as f:
            n_available = f[tau_path].shape[0]
        # Each mock stitches tens of box lengths, so the pool it draws from has to be much
        # larger than the number of mocks or the same sightlines recur across the sample.
        size = pool or min(n_available, max(500, 20 * n_mocks))
        picks = rng.choice(n_available, size=min(size, n_available), replace=False)
        tau, z_sim, dv_sim = read_sightlines(path, picks, tau_path)
        _TAU.append([tau, z_sim, dv_sim, snap_label(path), 1.0 if uvb == 'match' else float(uvb)])
        print(f"snap {snap_label(path)}: pool of {tau.shape[0]} sightlines, "
              f"z = {z_sim:.4f}, dv = {dv_sim:.4f} km/s")

    dz, mean_flux = observed_by_snapshot(templates)
    print(f"{len(templates)} templates, {n_mocks} mocks; observed path per snapshot slice:")
    for p, d, f in zip(_TAU, dz, mean_flux):
        print(f"  snap {p[3]} (z = {p[1]:.4f}): dz = {d:.3f} ({d / dz.sum():5.1%}), "
              f"mean flux {f:.4f}")
    if uvb == 'match':
        match_uvb(mean_flux)

    summary = {}
    for snr in snrs:
        jobs = [(i % len(templates), snr, continuum_error, int(seed) * 100000 + i)
                for i in range(n_mocks)]

        if workers > 1:
            import multiprocessing
            # fork, not the 3.14 default: the workers need to inherit the sightline
            # pool rather than have it pickled to them once per job.
            with multiprocessing.get_context('fork').Pool(workers) as workpool:
                batches = workpool.map(_one_mock, jobs, chunksize=1)
        else:
            batches = [_one_mock(j) for j in jobs]
        rows = [r for batch, _ in batches for r in batch]
        per_sightline = [s for _, s in batches]

        summary[snr] = _summarise(rows, snr, n_mocks)
        if out_dir:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f'mock_lines_snr-{snr}.csv'
            with open(path, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)
            print(f'  written: {path}')

            path = out / f'mock_sightlines_snr-{snr}.csv'
            with open(path, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=SIGHTLINE_FIELDS, extrasaction='ignore')
                w.writeheader()
                w.writerows(per_sightline)
            print(f'  written: {path}')
    return summary


def _summarise(rows, snr, n_mocks, max_b_err=10.0, max_logN_err=0.5):
    kept = [r for r in rows if r['b_err'] <= max_b_err and r['logN_err'] <= max_logN_err]
    if not kept:
        print(f'  S/N {snr:<5} no components survived the error cuts')
        return None
    b = np.array([r['b'] for r in kept])
    n = np.array([r['logN'] for r in kept])
    print(f'  S/N {str(snr):<5} {len(rows):5d} components, {len(kept):5d} well measured, '
          f'{len(kept) / n_mocks:5.1f} per sightline')
    print(f'            b    median {np.median(b):6.1f}  16-84% '
          f'{np.percentile(b, 16):5.1f}-{np.percentile(b, 84):5.1f}')
    print(f'            logN median {np.median(n):6.2f}  16-84% '
          f'{np.percentile(n, 16):5.2f}-{np.percentile(n, 84):5.2f}')
    return {'n': len(kept), 'b_median': float(np.median(b)),
            'logN_median': float(np.median(n))}


def self_test():
    global _TAU, _TEMPLATES
    rng = np.random.default_rng(3)
    wave = np.arange(1250.0, 1256.0, 0.00997)
    good = np.ones(wave.size, dtype=bool)

    # Two real lines plus pure noise elsewhere: the windows must find the lines and not
    # the noise, whatever the noise does.
    fine = voigt_fit.subpixel_grid(wave)
    dv = np.median(np.diff(fine)) / np.median(fine) * voigt_fit.C_KMS
    kernel = voigt_fit.lsf_kernel(dv, 17.0)
    tau = sum(voigt_fit.voigt_tau(fine, z, 25.0, 14.0) for z in (0.0288, 0.0325))
    clean = voigt_fit.convolve(np.exp(-tau), kernel).reshape(wave.size, -1).mean(axis=1)

    sigma = np.full_like(wave, 0.03)
    flux = clean + rng.normal(0.0, sigma)
    windows = fitting_windows(wave, flux, sigma, good)
    assert len(windows) == 2, windows
    for lo, hi in windows:
        assert hi - lo >= 10

    rows = measure_mock(wave, flux, sigma, good)
    kept = [r for r in rows if r['b_err'] <= 10 and r['logN_err'] <= 0.5]
    assert len(kept) == 2, [(r['z'], r['b'], r['logN']) for r in rows]
    for r, z in zip(sorted(kept, key=lambda r: r['z']), (0.0288, 0.0325)):
        assert abs(r['z'] - z) / (1 + z) * voigt_fit.C_KMS < 6.0, (r, z)
        assert abs(r['b'] - 25.0) < 6.0, r
        assert abs(r['logN'] - 14.0) < 0.2, r

    # Stitching covers the span with independent draws rather than one repeated box.
    tau_block = np.zeros((5, 2000))
    tau_block[2, 1000] = 3.0
    path = stitch_path(tau_block, 0.1, 1500.0, np.random.default_rng(1))
    assert path.size * 0.1 >= 1500.0 + 0.1, path.size
    assert path.size % 2000 == 0

    # Pure noise must yield no windows at all, so a mock of empty sky stays empty.
    assert fitting_windows(wave, 1.0 + rng.normal(0.0, sigma), sigma, good) == []

    # Masking a line removes its window, so a mock loses the same path the data loses.
    masked = good.copy()
    masked[:150] = False
    kept_windows = fitting_windows(wave, flux, sigma, masked)
    assert len(kept_windows) == 1, kept_windows
    assert kept_windows[0][0] > 150

    # Each pixel takes the snapshot nearest its own redshift, and two snapshots join without
    # a seam: an empty box next to a uniformly absorbing one gives exactly 1 and exp(-1).
    template = prepare_model_grid({'name': 'selftest', 'wave': wave, 'err': sigma,
                                   'good': good, 'flux': np.ones_like(wave)})
    _TAU = [[np.zeros((3, 2000)), 0.0290, 0.1, '088', 1.0],
            [np.ones((3, 2000)), 0.0320, 0.1, '086', 1.0]]
    z_grid = template['grid'] / voigt_fit.LYA - 1.0
    model = mock_flux(template, np.random.default_rng(2))
    near_a = z_grid < 0.0305 - 1e-4
    near_b = z_grid > 0.0305 + 1e-4
    assert np.allclose(model[near_a], 1.0) and np.allclose(model[near_b], np.exp(-1.0))
    assert np.allclose(snapshot_index([0.0, 0.0304, 0.0306, 0.5]), [0, 0, 1, 1])

    # Kernel lookup follows her nearest region's table, then the nearest tabulated column.
    fake = {'lsf_centres': np.array([1300.0, 1500.0]), 'lsf_region_table': np.array([0, 1]),
            'lsf_tables': [(np.array([1290.0, 1310.0]), np.ones((3, 2)) / 3, 0.00997),
                           (np.array([1490.0, 1510.0]), np.ones((3, 2)) / 3, 0.01223)]}
    t_idx, c_idx = kernel_lookup(fake, np.array([1295.0, 1305.0, 1395.0, 1405.0, 1515.0]))
    assert list(t_idx) == [0, 0, 0, 1, 1] and list(c_idx) == [0, 1, 1, 0, 1], (t_idx, c_idx)
    _, k_dv = kernel_at(fake, 1505.0)
    assert abs(k_dv - 0.01223 / 1510.0 * voigt_fit.C_KMS) < 1e-9

    # One mock end to end: the per-sightline summary must carry every column the CSV writes,
    # or a field added on one side and not the other is silently dropped at write time.
    _TAU = [[tau_block, 0.05, 0.1, '086', 1.0]]
    _TEMPLATES = [template]
    rows, per_sightline = _one_mock((0, 'data', 0.0, 7))
    assert set(SIGHTLINE_FIELDS) <= set(per_sightline), \
        set(SIGHTLINE_FIELDS) - set(per_sightline)
    assert per_sightline['n_components'] == len(rows)
    assert per_sightline['ew_sum'] >= 0.0 and per_sightline['n_good'] == int(good.sum())
    assert sum(map(int, per_sightline['flux_counts'].split())) <= good.sum()
    rows, per_sightline = _one_mock((0, '20', 0.0, 8))
    assert per_sightline['n_good'] == int(good.sum())
    _TAU, _TEMPLATES = [], []

    # Two lines closer than the merge distance are one window, because a blend has to be
    # fitted together rather than as two independent regions.
    close = voigt_fit.convolve(
        np.exp(-sum(voigt_fit.voigt_tau(fine, z, 25.0, 14.0) for z in (0.0300, 0.0303))),
        kernel).reshape(wave.size, -1).mean(axis=1)
    assert len(fitting_windows(wave, close + rng.normal(0.0, sigma), sigma, good)) == 1

    print('sim_mock_lines self-test passed')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spectra', nargs='*',
                   help='generated spectra HDF5, one per snapshot; each pixel is drawn from '
                        'the snapshot nearest its redshift')
    p.add_argument('--templates', default='data/Obs/*_all.dat',
                   help='observed sightlines to borrow grids, errors and masks from')
    p.add_argument('--cos-list', default='data/Obs/cos.list')
    p.add_argument('--n', type=int, default=255, help='mock sightlines per rung')
    p.add_argument('--snr', action='append', default=[],
                   help="'data' for the template's own errors, or S/N per pixel; repeatable")
    p.add_argument('--lsf', choices=['cos', 'gauss'], default='cos')
    p.add_argument('--lsf-dir', default='data/Obs/lsf', help='STScI aa_LSFTable_*_cn.dat files')
    p.add_argument('--fwhm', type=float, default=17.0, help='Gaussian FWHM for --lsf gauss')
    p.add_argument('--z-max', type=float, default=Z_MAX)
    p.add_argument('--uvb', default='1.0',
                   help="factor on tau (1/Gamma_HI), or 'match' for per-snapshot factors "
                        "that reproduce the observed mean flux in each slice")
    p.add_argument('--continuum-error', type=float, default=0.0)
    p.add_argument('--tau-path', default='tau/H/1/1215')
    p.add_argument('--pool', type=int, default=None,
                   help='simulated sightlines to draw stitched paths from')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.spectra:
        p.error('need a spectra file (or --self-test)')
    snrs = args.snr or ['data']
    for s in snrs:
        if s != 'data' and not (np.isfinite(float(s)) and float(s) > 0):
            p.error(f"--snr {s}: use 'data' or a finite S/N per pixel; the noiseless "
                    "reference is the C++ catalogue in output/analysis")

    templates = load_templates(args.templates, args.cos_list, args.z_max)
    if not templates:
        p.error(f'no usable templates matched {args.templates}')
    for t in templates:
        prepare_model_grid(t, args.lsf_dir if args.lsf == 'cos' else None, args.fwhm,
                           args.z_max)
    for path, (waves, kernels, disp) in sorted(_LSF_TABLES.items()):
        j = len(waves) // 2
        above = np.flatnonzero(kernels[:, j] >= 0.5 * kernels[:, j].max())
        fwhm = (above[-1] - above[0] + 1) * disp / waves[j] * voigt_fit.C_KMS
        print(f'COS LSF {path.name}: FWHM {fwhm:.1f} km/s at {waves[j]:.0f} A')

    run(args.spectra, templates, args.n, snrs, args.uvb, args.continuum_error,
        args.workers, args.seed, args.out, args.tau_path, args.pool)
    return 0


if __name__ == '__main__':
    sys.exit(main())
