"""
Turn simulated sightlines into mock COS spectra and measure lines in them.

The simulated line catalogue we have was measured on noiseless optical depth at 0.1 km/s
resolution. No real spectrum looks like that, so comparing it to an observed catalogue
compares two different measurements rather than two universes. This script puts the
simulation through what a real sightline goes through -- the COS line spread function, a real
sightline's pixel grid, a real sightline's errors, a real sightline's masked windows -- and
then fits Voigt components to the result with scripts/voigt_fit.py.

Mocks cycle through the observed sightlines, so the mock sample inherits the survey's actual
mix of noise, wavelength coverage and masking rather than one idealised setup.

Give it one spectra file per snapshot and each observed sightline is measured against the
snapshot nearest its own redshift. The observed absorbers run from z = 0 to 0.61, so a
single z = 0 snapshot is a 0.13 redshift error on the typical line.

Output is one CSV of components per cell, with the same columns as the observed VPFIT
catalogue, so anything that reads one reads the other.

    python scripts/sim_mock_lines.py \\
        spectra/IllustrisTNG/EX/EX_0/camel_lya_lya_h_spectra_snap_0{72,80,90}_n10000.hdf5 \\
        --n 200 --snr data --workers 24 --out output/obs/mock_lines/EX_0

    python scripts/sim_mock_lines.py SPECTRA.hdf5 --snr 80 --snr 50 --snr 10
    python scripts/sim_mock_lines.py --self-test
"""

import argparse
import csv
import glob
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observations as obs                                       # noqa: E402
import voigt_fit                                                 # noqa: E402

FIELDS = ['sightline', 'template', 'mock', 'snr', 'ion', 'ion_key',
          'z', 'z_err', 'b', 'b_err', 'logN', 'logN_err',
          'region_lo', 'region_hi', 'chi2', 'n_pix', 'snap']

# One sightline pool per snapshot, held module-level so forked workers inherit them
# instead of receiving a copy of a multi-hundred-megabyte array with every job.
_TAU = []


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


def load_templates(pattern, cos_list_path):
    """Observed sightlines to borrow grids, errors and masks from."""
    meta = obs.load_cos_list(cos_list_path) if cos_list_path else {}
    out = []
    for path in sorted(glob.glob(pattern)):
        name = Path(path).stem.replace('_all', '')
        wave, flux, err = obs.load_spectrum(path)
        info = meta.get(name)
        if info and info['windows']:
            good = obs.mask_from_windows(wave, err, info['windows'], info['z_em'])
        elif info:
            good = obs.usable_mask(wave, err, info['z_em'])
        else:
            good = np.isfinite(err) & (err > 0)
        if good.sum() < 200:
            continue
        z_mid = float(np.median(wave[good])) / voigt_fit.LYA - 1.0
        out.append({'name': name, 'wave': wave, 'err': err, 'good': good, 'z': z_mid})
    return out


def stitch_path(tau, dv_sim, wave, rng):
    """Lay independent simulated sightlines end to end to cover an observed window.

    An observed sightline spans 0.1-0.5 in redshift; one box is 2500 km/s, about 0.008. The
    forward model covers the difference by repeating the sightline, which is correct for a
    periodic box but means the same few absorbers recur twenty times over -- inflating the
    line count and correlating a path that should be independent. Drawing a fresh sightline
    per box length instead gives the window the independent structure a real one has.
    """
    span = (wave[-1] - wave[0]) / (0.5 * (wave[0] + wave[-1])) * voigt_fit.C_KMS
    n_box = int(np.ceil(span / (tau.shape[1] * dv_sim))) + 2
    picks = rng.integers(0, tau.shape[0], size=n_box)
    return np.concatenate([tau[i] for i in picks])


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


def measure_mock(wave, flux, sigma, good, fwhm_kms=17.0, kernel=None):
    """Voigt components in one mock spectrum, region by region."""
    rows = []
    for lo, hi in fitting_windows(wave, flux, sigma, good):
        sel = slice(lo, hi)
        params, errs, info = voigt_fit.fit_region(
            wave[sel], flux[sel], sigma[sel], fwhm_kms,
            kernel_table=None if kernel is None else kernel[0],
            kernel_dv=None if kernel is None else kernel[1])
        for c in voigt_fit.components_from_fit(params, errs):
            rows.append({**c, 'region_lo': float(wave[lo]), 'region_hi': float(wave[hi - 1]),
                         'chi2': info['chi2'], 'n_pix': info['n']})
    return rows


def _one_mock(job):
    """One mock sightline observed the way one real template sightline was. For Pool."""
    (isnap, template, snr, fwhm, kernel, uvb, continuum_error, seed) = job
    tau, z_sim, dv_sim, snap = _TAU[isnap]
    rng = np.random.default_rng(seed)
    wave, err, good = template['wave'], template['err'], template['good']
    path = stitch_path(tau, dv_sim, wave, rng)

    flux, sigma = obs.forward_model(
        path, dv_sim, z_sim, wave,
        snr_resel=np.inf if snr == 'data' else float(snr),
        sigma=err if snr == 'data' else None,
        fwhm_kms=fwhm, kernel=kernel, uvb_factor=uvb,
        continuum_error=continuum_error, rng=rng)

    rows = measure_mock(wave, flux, sigma, good, fwhm, kernel)
    for r in rows:
        r.update({'sightline': f"mock{seed}", 'template': template['name'],
                  'mock': seed, 'snr': snr, 'ion': 'H I', 'ion_key': 'HI', 'snap': snap})
    return rows


def snap_label(path):
    m = re.search(r'snap_(\d+)', str(path))
    return m.group(1) if m else Path(path).stem


def assign_snapshots(templates, z_snap):
    """Each observed sightline is compared against the snapshot nearest its own redshift.

    With one snapshot this is a no-op; with several it is what stops the whole survey,
    which reaches z = 0.61, being compared against a z = 0 box.
    """
    z_snap = np.asarray(z_snap, dtype=float)
    for t in templates:
        t['isnap'] = int(np.argmin(np.abs(z_snap - t['z'])))
    return templates


def run(spectra_paths, templates, n_mocks, snrs, fwhm=17.0, kernel=None, uvb=1.0,
        continuum_error=0.0, workers=1, seed=0, out_dir=None, tau_path='tau/H/1/1215',
        pool=None):
    rng = np.random.default_rng(seed)
    if isinstance(spectra_paths, (str, Path)):
        spectra_paths = [spectra_paths]

    global _TAU
    _TAU = []
    for path in spectra_paths:
        with h5py.File(path, 'r') as f:
            n_available = f[tau_path].shape[0]
        # Each mock stitches tens of box lengths, so the pool it draws from has to be much
        # larger than the number of mocks or the same sightlines recur across the sample.
        size = pool or min(n_available, max(500, 20 * n_mocks))
        picks = rng.choice(n_available, size=min(size, n_available), replace=False)
        tau, z_sim, dv_sim = read_sightlines(path, picks, tau_path)
        _TAU.append((tau, z_sim, dv_sim, snap_label(path)))
        print(f"snap {snap_label(path)}: pool of {tau.shape[0]} sightlines, "
              f"z = {z_sim:.4f}, dv = {dv_sim:.4f} km/s")

    assign_snapshots(templates, [t[1] for t in _TAU])
    counts = Counter(_TAU[t['isnap']][3] for t in templates)
    print(f"{len(templates)} templates, {n_mocks} mocks, assigned to snapshots: "
          + ', '.join(f'{k} x{v}' for k, v in sorted(counts.items())))

    summary = {}
    for snr in snrs:
        jobs = [(templates[i % len(templates)]['isnap'], templates[i % len(templates)], snr,
                 fwhm, kernel, uvb, continuum_error, int(seed) * 100000 + i)
                for i in range(n_mocks)]

        if workers > 1:
            import multiprocessing
            # fork, not the 3.14 default: the workers need to inherit the sightline
            # pool rather than have it pickled to them once per job.
            with multiprocessing.get_context('fork').Pool(workers) as workpool:
                batches = workpool.map(_one_mock, jobs, chunksize=1)
        else:
            batches = [_one_mock(j) for j in jobs]
        rows = [r for batch in batches for r in batch]

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

    # Stitching covers the window with independent draws rather than one repeated box.
    tau_block = np.zeros((5, 2000))
    tau_block[2, 1000] = 3.0
    path = stitch_path(tau_block, 0.1, wave, np.random.default_rng(1))
    span = (wave[-1] - wave[0]) / np.mean(wave[[0, -1]]) * voigt_fit.C_KMS
    assert path.size * 0.1 >= span, (path.size * 0.1, span)
    assert path.size % 2000 == 0

    # Pure noise must yield no windows at all, so a mock of empty sky stays empty.
    assert fitting_windows(wave, 1.0 + rng.normal(0.0, sigma), sigma, good) == []

    # Masking a line removes its window, so a mock loses the same path the data loses.
    masked = good.copy()
    masked[:150] = False
    kept_windows = fitting_windows(wave, flux, sigma, masked)
    assert len(kept_windows) == 1, kept_windows
    assert kept_windows[0][0] > 150

    # Two lines closer than the merge distance are one window, because a blend has to be
    # fitted together rather than as two independent regions.
    close = voigt_fit.convolve(
        np.exp(-sum(voigt_fit.voigt_tau(fine, z, 25.0, 14.0) for z in (0.0300, 0.0303))),
        kernel).reshape(wave.size, -1).mean(axis=1)
    assert len(fitting_windows(wave, close + rng.normal(0.0, sigma), sigma, good)) == 1

    # Each sightline goes to the snapshot nearest in redshift, ties to the lower index.
    picked = assign_snapshots([{'z': z} for z in (0.01, 0.12, 0.30, 0.55)],
                              [0.0, 0.1001, 0.2696, 0.5373])
    assert [t['isnap'] for t in picked] == [0, 1, 2, 3], picked

    print('sim_mock_lines self-test passed')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('spectra', nargs='*',
                   help='generated spectra HDF5, one per snapshot; each observed '
                        'sightline is measured against the nearest in redshift')
    p.add_argument('--templates', default='data/Obs/*_all.dat',
                   help='observed sightlines to borrow grids, errors and masks from')
    p.add_argument('--cos-list', default='data/Obs/cos.list')
    p.add_argument('--n', type=int, default=200, help='mock sightlines per cell')
    p.add_argument('--snr', action='append', default=[],
                   help="'data' for the template's own errors, or a number per "
                        "resolution element; repeatable")
    p.add_argument('--fwhm', type=float, default=17.0)
    p.add_argument('--lsf-table', help='two-column tabulated COS kernel')
    p.add_argument('--lsf-pixel-kms', type=float, default=None,
                   help='kernel spacing if the table is indexed in pixels')
    p.add_argument('--uvb', type=float, default=1.0, help='factor on tau (1/Gamma_HI)')
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

    kernel = None
    if args.lsf_table:
        import obs_compare
        kernel = obs_compare.load_lsf_table(args.lsf_table)

    templates = load_templates(args.templates, args.cos_list)
    if not templates:
        p.error(f'no usable templates matched {args.templates}')

    snrs = args.snr or ['data']
    run(args.spectra, templates, args.n, snrs, args.fwhm, kernel, args.uvb,
        args.continuum_error, args.workers, args.seed, args.out, args.tau_path,
        args.pool)
    return 0


if __name__ == '__main__':
    sys.exit(main())
