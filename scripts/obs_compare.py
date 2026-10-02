"""
Observed forest scatter against simulated sightlines, over a grid of instrument states.

One box length of velocity is the unit of comparison: the observed spectrum is cut into
chunks that long, and each chunk is matched against simulated sightlines forward-modelled
onto that chunk's own wavelength grid, errors and mask. PG 1048+342 contributes fifteen
chunks and about forty absorbers, so nothing here is averaged into an ensemble mean: the
products are the per-chunk and per-absorber rows themselves, which scripts/obs_scatter.py
draws as clouds with the observation overlaid.

Each chunk is compared against the snapshot nearest its own redshift, since the forest thins
measurably between z = 0.15 and z = 0.

Because the question is what survives the instrument, the whole comparison runs over a grid:
    S/N per resolution element : the observed errors, or inf / 80 / 50 / 10
    LSF                        : none / Gaussian / a tabulated COS kernel
The simulated tau arrays are large, so they are loaded once and every cell reuses them.

    python scripts/obs_compare.py data/Obs/pg1048_all.dat --z-em 0.167 \
        --spectra 'spectra/IllustrisTNG/1P/1P_p1_0/camel_lya_lya_h_spectra_snap_08*_n10000.hdf5' \
        --snr data --snr inf --snr 80 --snr 50 --snr 10

    python scripts/obs_compare.py --self-test
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import observations as obs
from statistical_tests import kolmogorov_smirnov_test, uniformity_test

STATS = ['tau_eff', 'n_absorbers', 'ew_sum', 'gap_max']
ABSORBER_FIELDS = ['ew', 'z', 'width', 'depth', 'ew_snr']


def load_sim_spectra(path, tau_path='tau/H/1/1215'):
    """Read tau, redshift and pixel width from a generated spectra file."""
    with h5py.File(path, 'r') as f:
        if tau_path not in f:
            raise KeyError(f"{path}: no {tau_path}")
        tau = f[tau_path][:]
        header = f['Header'].attrs
        z = float(header.get('redshift', header.get('Redshift')))
        if 'dvbin' in header:
            dv = float(header['dvbin'])
        else:
            # Same construction analyze uses: box length in velocity over the pixel count.
            box = float(header['box']) / 1000.0                     # ckpc/h -> cMpc/h
            om = float(header.get('omega_m', header.get('Omega0', 0.3089)))
            hz = 100.0 * np.sqrt(om * (1 + z) ** 3 + (1 - om))      # km/s/(Mpc/h)
            dv = box * hz / (1 + z) / tau.shape[1]
    return tau, z, dv


def load_lsf_table(path):
    """Read a two-column LSF table: velocity offset [km/s] or pixel index, and weight.

    STScI publishes the COS kernels per grating and lifetime position. A table in pixels is
    converted with --lsf-pixel-kms. Returns (weights, spacing in km/s).
    """
    table = np.loadtxt(path)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"{path}: expected two columns (offset, weight)")
    offset, weight = table[:, 0], table[:, 1]
    spacing = float(np.median(np.diff(offset)))
    return weight, spacing


def chunk_row(wave, flux, err, good):
    """The per-chunk scalars, computed identically for the observation and for a mock."""
    recs = obs.absorber_properties(wave, flux, err, good)
    ew = np.array([r['ew'] for r in recs])
    gaps = obs.gap_lengths(wave, flux, err, good)
    return {
        'tau_eff': obs.tau_eff(flux, good),
        'n_absorbers': float(len(recs)),
        'ew_sum': float(ew.sum()),
        'gap_max': float(gaps.max()) if len(gaps) else np.nan,
    }, recs


def mock_rows(tau, dv_sim, z_sim, wave, err, good, n_mocks, snr_resel=None, fwhm_kms=17.0,
              kernel=None, uvb_factor=1.0, continuum_error=0.0, rng=None):
    """Statistics of `n_mocks` simulated sightlines observed the way this chunk was.

    The chunk's own errors and mask are reused, so the mock loses the same path and carries
    the same noise the data has at that wavelength, unless `snr_resel` asks for a rung of the
    ladder instead.
    """
    rng = np.random.default_rng() if rng is None else rng
    picks = rng.choice(len(tau), size=min(n_mocks, len(tau)), replace=False)
    chunks, absorbers = [], []
    for i in picks:
        flux, sigma = obs.forward_model(
            tau[i], dv_sim, z_sim, wave,
            snr_resel=np.inf if snr_resel is None else snr_resel,
            sigma=err if snr_resel is None else None,
            fwhm_kms=fwhm_kms, kernel=kernel, uvb_factor=uvb_factor,
            continuum_error=continuum_error, rng=rng)
        row, recs = chunk_row(wave, flux, sigma, good)
        chunks.append(row)
        absorbers.extend(recs)
    return chunks, absorbers


def plot_spectra(wave, flux, err, good, sims, out_path, n_panels=4, chunk=None,
                 snr_resel=None, fwhm_kms=17.0, kernel=None, uvb_factor=1.0, seed=11):
    """The observed spectrum with forward-modelled sightlines stacked below it.

    Mocks are different patches of universe, so no line matches line for line. What this
    shows is whether they are the same kind of spectrum: same pixel scale, same noise, same
    line density and depth. It is the check every summary statistic is a compression of.
    """
    rng = np.random.default_rng(seed)
    z_sims = np.array([s[1] for s in sims])
    dv_chunk = obs.box_velocity_length(25000.0, np.median(wave[good]) / obs.LYA - 1.0)
    slices = obs.chunk_slices(wave, good, dv_chunk)
    sl = slices[len(slices) // 2 if chunk is None else chunk]

    w, f, e, g = wave[sl], flux[sl], err[sl], good[sl]
    z_mid = 0.5 * (w[0] + w[-1]) / obs.LYA - 1.0
    tau_sim, z_sim, dv_sim = sims[int(np.argmin(np.abs(z_sims - z_mid)))]

    fig, axes = plt.subplots(n_panels + 1, 1, figsize=(11, 1.7 * (n_panels + 1)),
                             sharex=True, sharey=True)
    axes[0].step(w, f, where='mid', color='C3', lw=0.8)
    axes[0].fill_between(w, -e, e, step='mid', color='C3', alpha=0.25, lw=0)
    axes[0].fill_between(w, 0, 1.6, where=~g, color='0.85', step='mid', lw=0)
    axes[0].set_ylabel('observed', fontsize=9)

    for ax, i in zip(axes[1:], rng.choice(len(tau_sim), size=n_panels, replace=False)):
        mock, sigma = obs.forward_model(
            tau_sim[i], dv_sim, z_sim, w,
            snr_resel=np.inf if snr_resel is None else snr_resel,
            sigma=e if snr_resel is None else None,
            fwhm_kms=fwhm_kms, kernel=kernel, uvb_factor=uvb_factor, rng=rng)
        ax.step(w, mock, where='mid', color='C0', lw=0.8)
        ax.fill_between(w, -sigma, sigma, step='mid', color='C0', alpha=0.25, lw=0)
        ax.set_ylabel(f'sightline {i}', fontsize=9)

    for ax in axes:
        ax.axhline(1.0, color='0.5', lw=0.6, ls=':')
        ax.set_ylim(-0.15, 1.6)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel(r'observed wavelength [$\AA$]')
    axes[0].set_title(f'PG1048 chunk at z = {z_mid:.3f} and mocks from snapshot z = {z_sim:.3f}'
                      f'  ({"observed errors" if snr_resel is None else f"S/N {snr_resel:g}"})')
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}')


def compare(wave, flux, err, good, sims, n_mocks=500, snr_resel=None, uvb_factor=1.0,
            continuum_error=0.0, fwhm_kms=17.0, kernel=None, seed=42):
    """Per-chunk observed rows, the matching mock rows, and the observed percentiles.

    `sims` is a list of (tau, z, dv), one per snapshot; each chunk uses the nearest in z.
    Returns (chunk_rows, sim_chunk_rows, obs_absorbers, sim_absorbers).
    """
    rng = np.random.default_rng(seed)
    z_sims = np.array([s[1] for s in sims])
    dv_chunk = obs.box_velocity_length(25000.0, np.median(wave[good]) / obs.LYA - 1.0)

    rows, sim_rows, obs_abs, sim_abs = [], [], [], []
    for index, sl in enumerate(obs.chunk_slices(wave, good, dv_chunk)):
        w, f, e, g = wave[sl], flux[sl], err[sl], good[sl]
        z_mid = 0.5 * (w[0] + w[-1]) / obs.LYA - 1.0
        tau_sim, z_sim, dv_sim = sims[int(np.argmin(np.abs(z_sims - z_mid)))]

        observed, recs = chunk_row(w, f, e, g)
        for r in recs:
            obs_abs.append({**r, 'chunk': index, 'z_chunk': z_mid})

        mocks, m_abs = mock_rows(tau_sim, dv_sim, z_sim, w, e, g, n_mocks,
                                 snr_resel=snr_resel, fwhm_kms=fwhm_kms, kernel=kernel,
                                 uvb_factor=uvb_factor, continuum_error=continuum_error,
                                 rng=rng)
        for j, m in enumerate(mocks):
            sim_rows.append({'chunk': index, 'z_chunk': z_mid, 'z_sim': z_sim,
                             'mock': j, **m})
        for r in m_abs:
            sim_abs.append({**r, 'chunk': index, 'z_chunk': z_mid})

        row = {'chunk': index, 'z_mid': z_mid, 'z_sim': z_sim,
               'good_frac': float(g.mean()), 'n_mocks': len(mocks)}
        for key in STATS:
            sim_vals = np.array([m[key] for m in mocks], dtype=float)
            sim_vals = sim_vals[np.isfinite(sim_vals)]
            row[f'obs_{key}'] = observed[key]
            row[f'sim_{key}_median'] = np.median(sim_vals) if sim_vals.size else np.nan
            row[f'sim_{key}_lo'] = np.percentile(sim_vals, 16) if sim_vals.size else np.nan
            row[f'sim_{key}_hi'] = np.percentile(sim_vals, 84) if sim_vals.size else np.nan
            row[f'pct_{key}'] = (100.0 * np.mean(sim_vals < observed[key])
                                 if sim_vals.size and np.isfinite(observed[key]) else np.nan)
        rows.append(row)
    return rows, sim_rows, obs_abs, sim_abs


def summarise(rows, sim_rows):
    """Per-statistic verdicts.

    The headline is whether the observed percentiles are uniform: with fifteen chunks that is
    the question the sample size can answer. The pooled KS is kept as a secondary number, but
    it stacks chunks from four redshifts and so tests a null nobody believes.
    """
    out = {}
    for key in STATS:
        obs_vals = np.array([r[f'obs_{key}'] for r in rows], dtype=float)
        sim_vals = np.array([r[key] for r in sim_rows], dtype=float)
        pcts = np.array([r[f'pct_{key}'] for r in rows], dtype=float)
        obs_vals, sim_vals = obs_vals[np.isfinite(obs_vals)], sim_vals[np.isfinite(sim_vals)]
        if obs_vals.size < 3 or sim_vals.size < 3:
            continue
        uni = uniformity_test(pcts)
        ks = kolmogorov_smirnov_test(obs_vals, sim_vals)
        out[key] = {'uniform_p': uni['pvalue'], 'pct_median': np.nanmedian(pcts),
                    'pooled_ks_p': ks['pvalue'],
                    'obs_median': np.median(obs_vals), 'sim_median': np.median(sim_vals)}
    return out


def write_csv(rows, path, fields=None):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cell_tag(snr):
    return 'data' if snr is None else ('inf' if not np.isfinite(snr) else f'{snr:g}')


def self_test():
    rng = np.random.default_rng(3)
    dv_sim, n_pix, z_sim = 2.0, 1200, 0.1

    def sightlines(n, rate=6):
        v = (np.arange(n_pix) - n_pix / 2) * dv_sim
        out = np.zeros((n, n_pix))
        for i in range(n):
            for _ in range(rng.poisson(rate)):
                out[i] += rng.exponential(0.7) * np.exp(
                    -0.5 * ((v - rng.uniform(v[0], v[-1])) / 25.0) ** 2)
        return out

    tau_sim = sightlines(200)
    wave = np.arange(1290.0, 1320.0, 0.0367)
    err = np.full_like(wave, 0.05)
    truth, _ = obs.forward_model(sightlines(1)[0], dv_sim, z_sim, wave, sigma=err, rng=rng)
    good = np.ones_like(wave, dtype=bool)

    rows, sim_rows, obs_abs, sim_abs = compare(
        wave, truth, err, good, [(tau_sim, z_sim, dv_sim)], n_mocks=60, seed=1)
    assert rows and sim_rows and sim_abs
    row = rows[0]
    assert row['sim_tau_eff_lo'] <= row['sim_tau_eff_median'] <= row['sim_tau_eff_hi']
    assert row['sim_tau_eff_median'] > 0, row['sim_tau_eff_median']
    assert 1.0 < row['pct_tau_eff'] < 99.0, row['pct_tau_eff']
    assert all(set(ABSORBER_FIELDS) <= set(r) for r in sim_abs)

    # The regression test for the mask bug: masking a third of the pixels must give the same
    # mock tau_eff as deleting them, because the mock has to lose the path the data lost.
    masked = good.copy()
    masked[::3] = False
    err_hostile = err.copy()
    err_hostile[~masked] = 1e3      # what the old code let into the mean flux
    m_masked, _ = mock_rows(tau_sim, dv_sim, z_sim, wave, err_hostile, masked, 80,
                            rng=np.random.default_rng(5))
    m_kept, _ = mock_rows(tau_sim, dv_sim, z_sim, wave, err, good, 80,
                          rng=np.random.default_rng(5))
    t_masked = np.median([m['tau_eff'] for m in m_masked])
    t_kept = np.median([m['tau_eff'] for m in m_kept])
    assert abs(t_masked - t_kept) < 0.25 * t_kept, (t_masked, t_kept)

    # A simulation with more optical depth pushes the observation low in its distribution.
    biased = compare(wave, truth, err, good, [(3.0 * tau_sim, z_sim, dv_sim)],
                     n_mocks=60, seed=1)[0][0]
    assert biased['pct_tau_eff'] < row['pct_tau_eff']

    # Noise ladder: noise can only hide absorbers, never create detectable ones.
    noisy = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)],
                    n_mocks=60, snr_resel=10.0, seed=1)[0][0]
    clean = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)],
                    n_mocks=60, snr_resel=np.inf, seed=1)[0][0]
    assert clean['sim_n_absorbers_median'] >= noisy['sim_n_absorbers_median']

    # LSF: smearing conserves equivalent width but widens features and fills cores.
    sharp = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)], n_mocks=40,
                    snr_resel=np.inf, fwhm_kms=0.0, seed=2)[3]
    smeared = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)], n_mocks=40,
                      snr_resel=np.inf, fwhm_kms=25.0, seed=2)[3]
    assert np.median([r['width'] for r in smeared]) > np.median([r['width'] for r in sharp])
    assert np.median([r['depth'] for r in smeared]) < np.median([r['depth'] for r in sharp])

    print('obs_compare self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spectrum', nargs='?', help='three-column observed .dat')
    parser.add_argument('--z-em', type=float, help='quasar emission redshift')
    parser.add_argument('--spectra', action='append', default=[],
                        help='generated sim spectra HDF5 or glob; repeat for more snapshots')
    parser.add_argument('--n-mocks', type=int, default=500,
                        help='simulated sightlines per chunk')
    parser.add_argument('--snr', action='append', default=[],
                        help="'data' for the observed errors, or a S/N per resolution "
                             "element ('inf', '80', ...); repeat to run the ladder")
    parser.add_argument('--lsf', choices=['none', 'gauss', 'cos'], default='gauss')
    parser.add_argument('--lsf-table', help='two-column kernel table, required for --lsf cos')
    parser.add_argument('--fwhm', type=float, default=17.0, help='Gaussian LSF FWHM, km/s')
    parser.add_argument('--uvb', type=float, default=1.0, help='factor on tau (1/Gamma_HI)')
    parser.add_argument('--continuum-error', type=float, default=0.0)
    parser.add_argument('--out-dir', default='output/obs')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.spectrum or args.z_em is None or not args.spectra:
        parser.error('need a spectrum, --z-em and at least one --spectra (or --self-test)')

    kernel, fwhm = None, args.fwhm
    if args.lsf == 'none':
        fwhm = 0.0
    elif args.lsf == 'cos':
        if not args.lsf_table:
            parser.error('--lsf cos needs --lsf-table (STScI kernel for the grating and LP)')
        kernel = load_lsf_table(args.lsf_table)
        print(f"LSF: tabulated, {len(kernel[0])} points at {kernel[1]:.2f} km/s spacing")

    wave, flux, err = obs.load_spectrum(args.spectrum)
    good = obs.usable_mask(wave, err, args.z_em)

    paths = sorted(p for pattern in args.spectra for p in glob.glob(pattern)) or args.spectra
    sims = []
    for path in paths:
        tau, z, dv = load_sim_spectra(path)
        sims.append((tau, z, dv))
        print(f"{Path(path).name}: {tau.shape[0]} sightlines, z = {z:.4f}, dv = {dv:.2f} km/s")

    rungs = []
    for raw in (args.snr or ['data']):
        rungs.append(None if raw.lower() == 'data' else float(raw))

    stem = Path(args.spectrum).stem
    for snr in rungs:
        tag = f'{stem}_lsf-{args.lsf}_snr-{cell_tag(snr)}'
        rows, sim_rows, obs_abs, sim_abs = compare(
            wave, flux, err, good, sims, n_mocks=args.n_mocks, snr_resel=snr,
            uvb_factor=args.uvb, continuum_error=args.continuum_error,
            fwhm_kms=fwhm, kernel=kernel)

        plot_spectra(wave, flux, err, good, sims, f'{args.out_dir}/{tag}_spectra.png',
                     snr_resel=snr, fwhm_kms=fwhm, kernel=kernel, uvb_factor=args.uvb)
        write_csv(rows, f'{args.out_dir}/{tag}_chunks.csv')
        write_csv(sim_rows, f'{args.out_dir}/{tag}_sim_chunks.csv')
        write_csv(obs_abs, f'{args.out_dir}/{tag}_obs_absorbers.csv')
        write_csv(sim_abs, f'{args.out_dir}/{tag}_sim_absorbers.csv')

        label = 'observed errors' if snr is None else f'S/N = {cell_tag(snr)} per resel'
        print(f"\n=== {label}, LSF {args.lsf} — {len(rows)} chunks x {args.n_mocks} mocks, "
              f"{len(obs_abs)} observed and {len(sim_abs)} simulated absorbers")
        print(f"{'stat':<14}{'obs med':>10}{'sim med':>10}{'pct med':>9}"
              f"{'uniform p':>11}{'pooled p':>10}")
        for key, s in summarise(rows, sim_rows).items():
            print(f"{key:<14}{s['obs_median']:>10.4g}{s['sim_median']:>10.4g}"
                  f"{s['pct_median']:>9.0f}{s['uniform_p']:>11.3g}{s['pooled_ks_p']:>10.3g}")
        print(f"written: {args.out_dir}/{tag}_*.csv")

    print(f"\nnext: python scripts/obs_scatter.py '{args.out_dir}/{stem}_lsf-*' "
          f"--out-dir plots/obs")
    return 0


if __name__ == '__main__':
    sys.exit(main())
