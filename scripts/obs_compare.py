"""
Observed forest chunks against simulated sightlines, compared as distributions.

One box length of velocity is the unit of comparison: an observed spectrum is cut into
chunks that long, and each chunk is matched against simulated sightlines forward-modelled
onto that chunk's own wavelength grid, noise and mask. The observation contributes about
fifteen chunks, the simulation thousands of sightlines, so the question asked of every
statistic is not "do the means agree" but "where does each observed chunk fall inside the
simulated distribution".

Each chunk is compared against the snapshot nearest its own redshift, since the forest
thins measurably between z = 0.15 and z = 0.

    python scripts/obs_compare.py data/Obs/pg1048_all.dat --z-em 0.167 \
        --spectra spectra/IllustrisTNG/1P/1P_p1_0/camel_lya_lya_h_spectra_snap_08{4,6,8}_n10000.hdf5 \
        --spectra spectra/IllustrisTNG/1P/1P_p1_0/camel_lya_lya_h_spectra_snap_090_n10000.hdf5

    python scripts/obs_compare.py --self-test

`--snr` replaces the observed errors with a fixed signal-to-noise per resolution element,
which is the noise ladder: what survives at S/N = inf, 80, 50, 10.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import observations as obs
from statistical_tests import kolmogorov_smirnov_test

STATS = ['tau_eff', 'n_absorbers', 'ew_sum', 'gap_max']


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
            h = float(header.get('hubble', 0.6774))
            om = float(header.get('omega_m', header.get('Omega0', 0.3089)))
            hz = 100.0 * np.sqrt(om * (1 + z) ** 3 + (1 - om))      # km/s/(Mpc/h)
            dv = box * hz / (1 + z) / tau.shape[1]
    return tau, z, dv


def mock_chunk_statistics(tau, dv_sim, z_sim, wave, err, n_mocks, snr_resel=None,
                          fwhm_kms=17.0, uvb_factor=1.0, continuum_error=0.0, rng=None):
    """Statistics of `n_mocks` simulated sightlines observed the way this chunk was.

    The chunk's own error array is reused, so the mock carries the same noise the data has
    at that wavelength, unless `snr_resel` asks for a rung of the noise ladder instead.
    """
    rng = np.random.default_rng() if rng is None else rng
    picks = rng.choice(len(tau), size=min(n_mocks, len(tau)), replace=False)
    good = np.ones_like(wave, dtype=bool)
    rows = []
    for i in picks:
        flux, sigma = obs.forward_model(
            tau[i], dv_sim, z_sim, wave,
            snr_resel=np.inf if snr_resel is None else snr_resel,
            sigma=err if snr_resel is None else None,
            fwhm_kms=fwhm_kms, uvb_factor=uvb_factor,
            continuum_error=continuum_error, rng=rng)
        ew, _ = obs.equivalent_widths(wave, flux, sigma, good)
        gaps = obs.gap_lengths(wave, flux, sigma, good)
        rows.append({
            'tau_eff': obs.tau_eff(flux, good),
            'n_absorbers': float(len(ew)),
            'ew_sum': float(ew.sum()),
            'gap_max': float(gaps.max()) if len(gaps) else np.nan,
        })
    return rows


def compare(wave, flux, err, good, sims, n_mocks=500, snr_resel=None, uvb_factor=1.0,
            continuum_error=0.0, fwhm_kms=17.0, seed=42):
    """Per-chunk observed statistics, the matching mock distributions, and percentiles.

    `sims` is a list of (tau, z, dv), one per snapshot; each chunk uses the nearest in z.
    """
    rng = np.random.default_rng(seed)
    z_sims = np.array([s[1] for s in sims])
    z_mid_all = np.median(wave[good]) / obs.LYA - 1.0
    dv_chunk = obs.box_velocity_length(25000.0, z_mid_all)

    results = []
    for sl in obs.chunk_slices(wave, good, dv_chunk):
        w, f, e, g = wave[sl], flux[sl], err[sl], good[sl]
        z_mid = 0.5 * (w[0] + w[-1]) / obs.LYA - 1.0
        tau_sim, z_sim, dv_sim = sims[int(np.argmin(np.abs(z_sims - z_mid)))]

        ew, _ = obs.equivalent_widths(w, f, e, g)
        gaps = obs.gap_lengths(w, f, e, g)
        observed = {
            'tau_eff': obs.tau_eff(f, g),
            'n_absorbers': float(len(ew)),
            'ew_sum': float(ew.sum()),
            'gap_max': float(gaps.max()) if len(gaps) else np.nan,
        }

        # The mock has no mask, so it would cover more path than the chunk does. Give it the
        # chunk's own gaps by handing it the masked error array: masked pixels get an error
        # large enough that nothing is detected there and they drop out of the statistics.
        e_mock = np.where(g, e, 1e3)
        mocks = mock_chunk_statistics(tau_sim, dv_sim, z_sim, w, e_mock, n_mocks,
                                      snr_resel=snr_resel, fwhm_kms=fwhm_kms,
                                      uvb_factor=uvb_factor,
                                      continuum_error=continuum_error, rng=rng)

        row = {'z_mid': z_mid, 'z_sim': z_sim, 'good_frac': float(g.mean()),
               'n_mocks': len(mocks)}
        for key in STATS:
            sim_vals = np.array([m[key] for m in mocks], dtype=float)
            sim_vals = sim_vals[np.isfinite(sim_vals)]
            row[f'obs_{key}'] = observed[key]
            row[f'sim_{key}_median'] = np.median(sim_vals) if sim_vals.size else np.nan
            row[f'sim_{key}_lo'] = np.percentile(sim_vals, 16) if sim_vals.size else np.nan
            row[f'sim_{key}_hi'] = np.percentile(sim_vals, 84) if sim_vals.size else np.nan
            row[f'pct_{key}'] = (100.0 * np.mean(sim_vals < observed[key])
                                 if sim_vals.size and np.isfinite(observed[key]) else np.nan)
        results.append((row, mocks))
    return results


def summarise(results):
    """Whether the observed chunks, as a sample, look drawn from the mock distribution."""
    out = {}
    for key in STATS:
        obs_vals = np.array([r[f'obs_{key}'] for r, _ in results], dtype=float)
        sim_vals = np.array([m[key] for _, mocks in results for m in mocks], dtype=float)
        obs_vals, sim_vals = obs_vals[np.isfinite(obs_vals)], sim_vals[np.isfinite(sim_vals)]
        if obs_vals.size < 3 or sim_vals.size < 3:
            continue
        ks = kolmogorov_smirnov_test(obs_vals, sim_vals)
        out[key] = {'ks': ks['statistic'], 'p': ks['pvalue'],
                    'obs_median': np.median(obs_vals), 'sim_median': np.median(sim_vals)}
    return out


def write_csv(results, path):
    rows = [r for r, _ in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(results, path, label=''):
    """Observed chunks over the simulated 16-84 percentile band, one panel per statistic."""
    z = np.array([r['z_mid'] for r, _ in results])
    fig, axes = plt.subplots(len(STATS), 1, figsize=(7, 3 * len(STATS)), sharex=True)
    for ax, key in zip(np.atleast_1d(axes), STATS):
        lo = np.array([r[f'sim_{key}_lo'] for r, _ in results])
        hi = np.array([r[f'sim_{key}_hi'] for r, _ in results])
        med = np.array([r[f'sim_{key}_median'] for r, _ in results])
        ax.fill_between(z, lo, hi, alpha=0.3, color='C0', label='simulated 16-84%')
        ax.plot(z, med, color='C0', lw=1, label='simulated median')
        ax.plot(z, [r[f'obs_{key}'] for r, _ in results], 'o', color='C3', label='observed')
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel('z')
    fig.suptitle(label or 'observed chunks vs forward-modelled sightlines')
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def self_test():
    rng = np.random.default_rng(3)

    # A synthetic observation and a synthetic "simulation" drawn from the same process:
    # Gaussian absorbers of random depth sprinkled along each sightline.
    dv_sim, n_pix, z_sim = 2.0, 1200, 0.1
    def sightlines(n, rate=6):
        v = (np.arange(n_pix) - n_pix / 2) * dv_sim
        out = np.zeros((n, n_pix))
        for i in range(n):
            for _ in range(rng.poisson(rate)):
                centre = rng.uniform(v[0], v[-1])
                out[i] += rng.exponential(0.7) * np.exp(-0.5 * ((v - centre) / 25.0) ** 2)
        return out

    tau_sim = sightlines(200)
    wave = np.arange(1290.0, 1320.0, 0.0367)
    err = np.full_like(wave, 0.05)
    truth, _ = obs.forward_model(sightlines(1)[0], dv_sim, z_sim, wave, sigma=err, rng=rng)
    good = np.ones_like(wave, dtype=bool)

    results = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)], n_mocks=60, seed=1)
    assert results, 'no chunks produced'
    row, mocks = results[0]
    assert len(mocks) == 60
    assert row['sim_tau_eff_lo'] <= row['sim_tau_eff_median'] <= row['sim_tau_eff_hi']

    # Drawn from the same process, so the observation should not sit in a tail.
    assert 1.0 < row['pct_tau_eff'] < 99.0, row['pct_tau_eff']

    # A simulation with twice the optical depth should push the observation low.
    biased = compare(wave, truth, err, good, [(3.0 * tau_sim, z_sim, dv_sim)],
                     n_mocks=60, seed=1)[0][0]
    assert biased['pct_tau_eff'] < row['pct_tau_eff'], (biased['pct_tau_eff'], row['pct_tau_eff'])

    # Noise ladder: less noise cannot hide absorbers, so more of them are found.
    noisy = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)],
                    n_mocks=60, snr_resel=10.0, seed=1)[0][0]
    clean = compare(wave, truth, err, good, [(tau_sim, z_sim, dv_sim)],
                    n_mocks=60, snr_resel=np.inf, seed=1)[0][0]
    assert clean['sim_n_absorbers_median'] >= noisy['sim_n_absorbers_median'], \
        (clean['sim_n_absorbers_median'], noisy['sim_n_absorbers_median'])

    print('obs_compare self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('spectrum', nargs='?', help='three-column observed .dat')
    parser.add_argument('--z-em', type=float, help='quasar emission redshift')
    parser.add_argument('--spectra', action='append', default=[],
                        help='generated sim spectra HDF5; repeat for several snapshots')
    parser.add_argument('--n-mocks', type=int, default=500,
                        help='simulated sightlines per chunk')
    parser.add_argument('--snr', type=float, default=None,
                        help='fixed S/N per resolution element instead of the observed errors')
    parser.add_argument('--uvb', type=float, default=1.0, help='factor on tau (1/Gamma_HI)')
    parser.add_argument('--continuum-error', type=float, default=0.0)
    parser.add_argument('--fwhm', type=float, default=17.0, help='LSF FWHM in km/s')
    parser.add_argument('--out-dir', default='output/obs')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.spectrum or args.z_em is None or not args.spectra:
        parser.error('need a spectrum, --z-em and at least one --spectra (or --self-test)')

    wave, flux, err = obs.load_spectrum(args.spectrum)
    good = obs.usable_mask(wave, err, args.z_em)

    sims = []
    for path in args.spectra:
        tau, z, dv = load_sim_spectra(path)
        sims.append((tau, z, dv))
        print(f"{Path(path).name}: {tau.shape[0]} sightlines, z = {z:.4f}, dv = {dv:.2f} km/s")

    results = compare(wave, flux, err, good, sims, n_mocks=args.n_mocks,
                      snr_resel=args.snr, uvb_factor=args.uvb,
                      continuum_error=args.continuum_error, fwhm_kms=args.fwhm)

    tag = Path(args.spectrum).stem + ('' if args.snr is None else f'_snr{args.snr:g}')
    write_csv(results, f'{args.out_dir}/{tag}_chunks.csv')
    plot(results, f'{args.out_dir}/{tag}_chunks.png', label=tag)

    print(f"\n{len(results)} chunks, {args.n_mocks} mocks each"
          f"{'' if args.snr is None else f', S/N = {args.snr:g} per resel'}")
    print(f"{'stat':<14}{'obs median':>12}{'sim median':>12}{'KS':>8}{'p':>10}")
    for key, s in summarise(results).items():
        print(f"{key:<14}{s['obs_median']:>12.4g}{s['sim_median']:>12.4g}"
              f"{s['ks']:>8.3f}{s['p']:>10.3g}")

    pct = np.array([r['pct_tau_eff'] for r, _ in results], dtype=float)
    pct = pct[np.isfinite(pct)]
    print(f"\ntau_eff percentiles of the observed chunks: "
          f"{np.percentile(pct, 25):.0f} / {np.median(pct):.0f} / {np.percentile(pct, 75):.0f}"
          f"  (25 / 50 / 75 expected near 25 / 50 / 75)")
    print(f"written: {args.out_dir}/{tag}_chunks.csv and .png")
    return 0


if __name__ == '__main__':
    sys.exit(main())
