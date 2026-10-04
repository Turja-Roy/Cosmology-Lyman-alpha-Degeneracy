"""
Run the observed spectrum through the same analysis the simulated spectra get.

`analyze` takes optical depth on a grid uniform in velocity, one row per sightline, and
produces the flux PDF, the tau PDF, the flux power spectrum, the line-width distribution and
the flux statistics. An observed spectrum can be put in exactly that shape, and then every
number is measured by the same code, which is the only way the two are comparable.

Two things have to happen first:

* **Masking.** Airglow, Milky Way lines, the damped Galactic Lyman-alpha trough and the
  quasar's proximity zone are removed, which leaves the forest in pieces. PG1048 keeps 13
  contiguous runs totalling 37 200 km/s, the longest 12 600 km/s.
* **Segmenting.** The power spectrum needs unbroken stretches, so the usable runs are cut
  into fixed-length segments containing no masked pixels, and each is resampled onto a grid
  uniform in velocity (the file is uniform in wavelength, so the pixel width in km/s drifts
  across the band). At the default 1000 km/s that gives 32 segments, 32 000 km/s of path.

Optical depth is -ln(F), and noise puts some observed pixels at or below zero flux, where
that diverges. Those pixels are clipped to `--tau-max`. The same clip has to be applied to
any simulation this is compared against, which is what obs_compare's forward model is for:
comparing a noisy observed flux PDF against a noiseless simulated one measures the noise.

    python scripts/obs_analyze.py data/Obs/pg1048_all.dat --z-em 0.167
    python scripts/obs_analyze.py --self-test
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

# analysis.py imports `scripts` as a package, so the repo root has to be importable when this
# file is run directly as `python scripts/obs_analyze.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import observations as obs
from analysis import (compute_flux_statistics, compute_flux_tau_pdf, compute_power_spectrum,
                      compute_effective_optical_depth, compute_line_width_distribution)
from data_export import save_analysis_results
from plotting import (setup_plot_style, plot_flux_power_spectrum, plot_flux_statistics,
                      plot_line_width_distribution)


def clean_segments(wave, flux, err, good, segment_kms=1000.0, dv_out=None):
    """Cut the usable spectrum into equal-length segments free of masked pixels.

    Returns (flux array [n_segments, n_pixels], dv in km/s, segment mid redshifts). Each
    segment is resampled onto a uniform velocity grid, because the analysis code assumes a
    constant pixel width and the observed grid is uniform in wavelength instead.
    """
    dv_pixel = obs.velocity_width(wave)
    dv_out = float(np.median(dv_pixel)) if dv_out is None else float(dv_out)
    n_pixels = int(segment_kms // dv_out)

    runs, start = [], None
    for i, ok in enumerate(np.append(good, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None

    rows, z_mid = [], []
    for a, b in runs:
        w, f = wave[a:b], flux[a:b]
        v = np.concatenate([[0.0], np.cumsum(dv_pixel[a:b])[:-1]])
        for k in range(int(v[-1] // segment_kms) if v.size else 0):
            v_grid = v[0] + k * segment_kms + np.arange(n_pixels) * dv_out
            rows.append(np.interp(v_grid, v, f))
            z_mid.append(np.interp(v_grid.mean(), v, w) / obs.LYA - 1.0)
    if not rows:
        return np.empty((0, n_pixels)), dv_out, np.array([])
    return np.vstack(rows), dv_out, np.array(z_mid)


def optical_depth(flux, tau_max=5.0):
    """tau = -ln(F), with noise-driven non-positive flux clipped rather than diverging."""
    return np.clip(-np.log(np.clip(flux, np.exp(-tau_max), None)), 0.0, tau_max).astype(np.float32)


def analyse(wave, flux, err, good, segment_kms=1000.0, tau_max=5.0, dv_out=None):
    """The same measurements `analyze` makes on simulated spectra."""
    seg_flux, dv, z_mid = clean_segments(wave, flux, err, good, segment_kms, dv_out)
    if seg_flux.shape[0] == 0:
        raise ValueError('no clean segments: lower --segment-kms')
    tau = optical_depth(seg_flux, tau_max)

    results = {
        'flux_stats': compute_flux_statistics(tau),
        'pdfs': compute_flux_tau_pdf(tau),
        'power_spectrum': compute_power_spectrum(np.exp(-tau), dv),
        'line_widths': compute_line_width_distribution(tau, dv),
    }
    results['flux_stats'].update(compute_effective_optical_depth(tau))
    results['segments'] = {'n_segments': int(seg_flux.shape[0]), 'segment_kms': segment_kms,
                           'dv_kms': dv, 'n_pixels': int(seg_flux.shape[1]),
                           'path_kms': float(seg_flux.shape[0] * segment_kms),
                           'z_min': float(z_mid.min()), 'z_max': float(z_mid.max()),
                           'z_median': float(np.median(z_mid))}
    return results, tau, dv


def run_one(path, z_em, windows, out_root, plot_root, segment_kms, tau_max, dv,
            make_plots=True):
    """Analyse one sightline and write its CSVs and plots. Returns the summary row."""
    name = Path(path).name.replace('_all.dat', '').replace('.dat', '')
    wave, flux, err = obs.load_spectrum(path)
    good = (obs.mask_from_windows(wave, err, windows, z_em) if windows
            else obs.usable_mask(wave, err, z_em))
    results, tau, dv_used = analyse(wave, flux, err, good, segment_kms, tau_max, dv)

    save_analysis_results(results, Path(out_root) / name)
    seg, stats = results['segments'], results['flux_stats']
    if make_plots:
        plot_dir = Path(plot_root) / name
        plot_dir.mkdir(parents=True, exist_ok=True)
        setup_plot_style()
        plot_flux_power_spectrum(results['power_spectrum'], seg['z_median'],
                                 plot_dir / 'power_spectrum.png',
                                 title=f'{name}: flux power spectrum')
        plot_flux_statistics(results['pdfs'], stats, plot_dir / 'flux_statistics.png', tau=tau)
        plot_line_width_distribution(results['line_widths'], seg['z_median'],
                                     plot_dir / 'line_widths.png',
                                     title=f'{name}: line widths')
    return {
        'name': name, 'z_em': z_em, 'n_segments': seg['n_segments'],
        'path_kms': seg['path_kms'], 'dv_kms': round(seg['dv_kms'], 3),
        'z_min': round(seg['z_min'], 4), 'z_max': round(seg['z_max'], 4),
        'z_median': round(seg['z_median'], 4),
        'mean_flux': stats.get('mean_flux', float('nan')),
        'tau_eff': stats.get('tau_eff', stats.get('effective_tau', float('nan'))),
        'tau_eff_std': stats.get('tau_eff_std', float('nan')),
        'median_snr': float(np.median(np.abs(flux[good]) / np.maximum(err[good], 1e-9))),
    }


def write_summary(rows, path):
    """One row per sightline: the population table the single-spectrum runs cannot give."""
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    rng = np.random.default_rng(0)
    wave = np.arange(1250.0, 1400.0, 0.0367)
    flux = np.ones_like(wave)
    v = np.cumsum(obs.velocity_width(wave))
    for centre in rng.uniform(v[0], v[-1], 60):
        flux *= 1 - 0.5 * np.exp(-0.5 * ((v - centre) / 20.0) ** 2)
    err = np.full_like(wave, 0.05)
    flux = flux + rng.normal(0, 1, wave.size) * err
    good = np.ones_like(wave, dtype=bool)
    good[2000:2300] = False                      # a mask gap in the middle

    seg, dv, z_mid = clean_segments(wave, flux, err, good, 1000.0)
    assert seg.shape[0] > 10 and seg.shape[1] == int(1000.0 // dv)
    assert 5.0 < dv < 10.0, dv
    assert np.all(np.diff(np.sort(z_mid)) >= 0)

    # The gap must cost path but not corrupt it: no segment may straddle the masked pixels.
    wide = clean_segments(wave, flux, err, np.ones_like(good), 1000.0)[0]
    assert wide.shape[0] > seg.shape[0], (wide.shape, seg.shape)

    # Clipping keeps tau finite where noise drove the flux non-positive.
    tau = optical_depth(seg, tau_max=5.0)
    assert np.isfinite(tau).all() and tau.max() <= 5.0 and tau.min() >= 0.0

    results, tau, dv = analyse(wave, flux, err, good)
    assert results['flux_stats']['mean_flux'] > 0
    assert results['power_spectrum']['k'].size > 1
    assert results['segments']['n_segments'] == seg.shape[0]

    # The measured mean flux must match a direct average of the segments.
    direct = float(np.mean(np.exp(-tau)))
    assert abs(results['flux_stats']['mean_flux'] - direct) < 1e-3, (
        results['flux_stats']['mean_flux'], direct)

    print('obs_analyze self-test passed '
          f"({results['segments']['n_segments']} segments, dv = {dv:.1f} km/s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spectrum', nargs='*',
                        help='three-column observed .dat files, or a glob')
    parser.add_argument('--z-em', type=float,
                        help='emission redshift; only for a single file without --cos-list')
    parser.add_argument('--cos-list', default=None,
                        help="the collaborator's table: z_em and usable windows per sightline")
    parser.add_argument('--no-plots', action='store_true',
                        help='CSVs only, which is what a 50-sightline batch usually wants')
    parser.add_argument('--segment-kms', type=float, default=1000.0,
                        help='length of each mask-free segment (default 1000)')
    parser.add_argument('--dv', type=float, default=None,
                        help='velocity pixel of the resampled grid (default: observed median)')
    parser.add_argument('--tau-max', type=float, default=5.0,
                        help='cap on -ln(F) where noise drives the flux to zero')
    parser.add_argument('--out-dir', default=None,
                        help='default output/analysis/Obs/<name>')
    parser.add_argument('--plot-dir', default=None, help='default plots/obs/<name>')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    paths = sorted(p for pattern in args.spectrum for p in glob.glob(pattern)) or args.spectrum
    if not paths:
        parser.error('need at least one spectrum (or --self-test)')

    table = obs.load_cos_list(args.cos_list) if args.cos_list else {}
    if not table and args.z_em is None:
        parser.error('need --cos-list, or --z-em for a single spectrum')

    out_root = Path(args.out_dir or 'output/analysis/Obs')
    plot_root = Path(args.plot_dir or 'plots/obs')

    rows, skipped = [], []
    for path in paths:
        name = Path(path).name.replace('_all.dat', '').replace('.dat', '')
        entry = table.get(name)
        if entry is None and args.z_em is None:
            skipped.append((name, 'no z_em'))
            continue
        z_em = entry['z_em'] if entry else args.z_em
        windows = entry['windows'] if entry else None
        try:
            row = run_one(path, z_em, windows, out_root, plot_root,
                          args.segment_kms, args.tau_max, args.dv,
                          make_plots=not args.no_plots)
        except ValueError as exc:
            skipped.append((name, str(exc)))
            continue
        rows.append(row)
        print(f"{row['name']:<12} z_em {z_em:.3f}  {row['n_segments']:>3} seg  "
              f"{row['path_kms']:>7.0f} km/s  z {row['z_min']:.3f}-{row['z_max']:.3f}  "
              f"<F> {row['mean_flux']:.4f}  tau_eff {row['tau_eff']:.4f}  "
              f"S/N {row['median_snr']:.0f}")

    if not rows:
        print('nothing analysed')
        return 1

    summary = out_root / 'sightline_summary.csv'
    write_summary(rows, summary)
    tau = np.array([r['tau_eff'] for r in rows], dtype=float)
    print(f"\n{len(rows)} sightlines, {sum(r['path_kms'] for r in rows):.0f} km/s of path, "
          f"{sum(r['n_segments'] for r in rows)} segments")
    print(f"tau_eff across sightlines: median {np.nanmedian(tau):.4f}, "
          f"16-84% {np.nanpercentile(tau, 16):.4f}-{np.nanpercentile(tau, 84):.4f}")
    if skipped:
        print(f"skipped {len(skipped)}: " + ', '.join(f'{n} ({why})' for n, why in skipped))
    print(f'summary: {summary}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
