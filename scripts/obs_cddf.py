"""
Observed column-density distribution from the VPFIT catalogue, against the EX variants.

The simulated CDDF counts absorbers per unit column density per unit absorption distance,
f(N) = d2n / dN dX. The same quantity can be built from the collaborator's fits: every H I
component is an absorber with a measured log N, and the absorption distance comes from the
usable wavelength windows in cos.list. No new simulation runs are needed -- the EX cddf.csv
files already exist, and this puts the observation on their axes.

Three cuts decide what the observed side means, and all three are reported:

* **Well-measured components only.** Blended or saturated fits carry b and log N errors
  larger than the values themselves; they are real rows in the fit but not measurements.
* **Forest redshifts only.** Components outside the usable windows, blueward of them, or
  inside the quasar's proximity zone are dropped, which also removes Galactic H I at z ~ 0.
* **Lyman-beta contamination.** Above z_em ~ 0.25 the band contains Lyman-beta from higher
  redshift systems. `--lyb-safe` restricts each sightline to lambda > 1025.7 (1 + z_em),
  where every absorption line must be Lyman-alpha.

    python scripts/obs_cddf.py --components output/analysis/Obs/vpfit_components.csv
    python scripts/obs_cddf.py --lyb-safe --snap snap-090
    python scripts/obs_cddf.py --self-test
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import observations as obs
from ex_obs_overlay import EX_SIMS, EX_LABEL, EX_COLOR, OBS_COLOR, read_table, _setup_style, _save

LYB = 1025.7222
OMEGA_M = 0.3089


def dx_dz(z, omega_m=OMEGA_M):
    """Absorption distance per unit redshift, (1+z)^2 / E(z)."""
    return (1 + z) ** 2 / np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))


def absorption_distance(z_lo, z_hi, omega_m=OMEGA_M, steps=512):
    """Integral of dX/dz over a redshift interval."""
    if z_hi <= z_lo:
        return 0.0
    z = np.linspace(z_lo, z_hi, steps)
    return float(np.trapezoid(dx_dz(z, omega_m), z))


def forest_ranges(entry, proximity=3000.0, lyb_safe=False, z_min=0.0):
    """Redshift intervals over which this sightline's Lyman-alpha forest is usable.

    Her windows already exclude the Galactic lines; this converts them to redshift, trims the
    quasar proximity zone, and optionally drops everything blueward of Lyman-beta at the
    emission redshift, where Lyman-beta from higher redshift systems can masquerade as
    Lyman-alpha.
    """
    z_em = entry['z_em']
    lam_max = obs.LYA * (1 + z_em) * (1 - proximity / obs.C_KMS)
    lam_min = obs.LYA * (1 + z_min)
    if lyb_safe:
        lam_min = max(lam_min, LYB * (1 + z_em))

    out = []
    for lo, hi in entry['windows']:
        lo, hi = max(lo, lam_min), min(hi, lam_max)
        if hi > lo:
            out.append((lo / obs.LYA - 1.0, hi / obs.LYA - 1.0))
    return out


def read_components(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def select_hi(rows, ranges_by_sightline, max_b_err=10.0, max_logN_err=0.5):
    """Well-measured H I components that fall inside their sightline's usable forest."""
    kept = []
    for r in rows:
        if r.get('ion_key') != 'HI':
            continue
        try:
            z, b_err, logN_err = float(r['z']), float(r['b_err']), float(r['logN_err'])
            b, logN = float(r['b']), float(r['logN'])
        except (KeyError, ValueError):
            continue
        if b_err > max_b_err or logN_err > max_logN_err or b <= 0:
            continue
        for lo, hi in ranges_by_sightline.get(r['sightline'], []):
            if lo <= z <= hi:
                kept.append({'sightline': r['sightline'], 'z': z, 'b': b, 'logN': logN})
                break
    return kept


def observed_cddf(logN, total_dx, edges):
    """f(N) = counts / (Delta N * Delta X), with Poisson errors."""
    counts, _ = np.histogram(logN, bins=edges)
    n_lin = 10.0 ** edges
    delta_n = np.diff(n_lin)
    f_n = counts / (delta_n * total_dx) if total_dx > 0 else np.zeros_like(counts, dtype=float)
    err = np.sqrt(counts) / (delta_n * total_dx) if total_dx > 0 else np.zeros_like(f_n)
    return counts, f_n, err


def sim_bins(table, default=(13.0, 16.0, 0.2)):
    """Bin edges matching the simulated CDDF, so the two are histogrammed alike."""
    if table is not None and 'log10_N_HI' in table and 'delta_log_N' in table:
        centres = table['log10_N_HI']
        width = float(np.median(table['delta_log_N']))
        return np.append(centres - width / 2, centres[-1] + width / 2)
    lo, hi, width = default
    return np.arange(lo, hi + width, width)


def plot_cddf(ex, obs_curve, out_path, snap, title_extra=''):
    """Observed f(N) over the EX variants."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for sim in EX_SIMS:
        table = ex.get(sim)
        if table is None:
            continue
        ok = table['f_N_HI'] > 0
        ax.plot(table['log10_N_HI'][ok], table['f_N_HI'][ok], color=EX_COLOR[sim], lw=1.6,
                ls='--' if sim == 'EX_0' else '-', label=EX_LABEL[sim])

    centres, f_n, err, counts = obs_curve
    ok = f_n > 0
    ax.errorbar(centres[ok], f_n[ok], yerr=err[ok], fmt='o', color=OBS_COLOR, ms=5,
                capsize=3, label='observed (VPFIT H I)')
    for x, y, n in zip(centres[ok], f_n[ok], counts[ok]):
        ax.annotate(str(int(n)), (x, y), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=7, color=OBS_COLOR)

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10} N_{\rm HI}$ [cm$^{-2}$]')
    ax.set_ylabel(r'$f(N_{\rm HI}) = d^2n / dN\, dX$')
    ax.set_title(f'column density distribution, EX set at {snap}{title_extra}')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def plot_b(ex_lw, b_obs, out_path, snap, logN_lo=13.0):
    """Doppler parameters: VPFIT values against the simulated line widths."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bins = np.linspace(0, 120, 31)
    for sim in EX_SIMS:
        table = ex_lw.get(sim)
        if table is None or 'b_param_km_s' not in table:
            continue
        b = table['b_param_km_s']
        if 'N_HI' in table:
            b = b[table['N_HI'] >= 10.0 ** logN_lo]
        counts, edges = np.histogram(b[np.isfinite(b)], bins=bins, density=True)
        ax.plot(0.5 * (edges[:-1] + edges[1:]), counts, color=EX_COLOR[sim], lw=1.6,
                ls='--' if sim == 'EX_0' else '-', label=EX_LABEL[sim])
    counts, edges = np.histogram(b_obs, bins=bins, density=True)
    ax.plot(0.5 * (edges[:-1] + edges[1:]), counts, color=OBS_COLOR, lw=2.0, marker='o', ms=3,
            label=f'observed (VPFIT, {len(b_obs)} lines)')
    ax.set_xlabel('b [km/s]')
    ax.set_ylabel('probability density')
    ax.set_title(f'Doppler parameters, EX set at {snap}')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_path)


# Matches src/cpp/analysis/constants.h, so simulated counts mean the same thing here as in
# compute_line_width_distribution.
TAU_TO_COLDEN = 8.51e11


def sightline_absorbers(tau, dv, tau_threshold=0.5, logN_min=13.5):
    """Absorbers per simulated sightline, counted the way the C++ line finder counts them.

    A feature is a run of pixels above `tau_threshold`; its column density is the integral of
    tau across it. Counting here rather than in C++ is what keeps the sightline identity,
    which compute_line_width_distribution flattens away and which per-sightline scatter needs.
    """
    n_min = 10.0 ** logN_min
    counts = np.zeros(tau.shape[0], dtype=int)
    for i in range(tau.shape[0]):
        above = tau[i] > tau_threshold
        if not above.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([False], above, [False]))))
        for a, b in zip(edges[::2], edges[1::2]):
            if TAU_TO_COLDEN * float(tau[i, a:b].sum()) * dv >= n_min:
                counts[i] += 1
    return counts


def mock_dndz(sim_counts, dz_box, obs_dz, n_realisations=500, rng=None):
    """Mock dn/dz values matched to each observed sightline's path length.

    One simulated sightline covers a box length, far less than a real sightline, so a mock
    sightline is several of them concatenated until the paths match. Comparing the observed
    spread against mocks of the same path and sample size is the point: a longer path averages
    its own scatter away.
    """
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((n_realisations, len(obs_dz)))
    for r in range(n_realisations):
        for j, dz in enumerate(obs_dz):
            k = max(int(round(dz / dz_box)), 1)
            picks = rng.choice(len(sim_counts), size=k, replace=True)
            out[r, j] = sim_counts[picks].sum() / (k * dz_box)
    return out


def plot_dndz_scatter(obs_values, mocks, out_path, label, logN_min):
    """Observed dn/dz distribution against mocks, plus the spread of the spread."""
    _setup_style()
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11, 4.4))

    grid = np.linspace(0, max(obs_values.max(), np.percentile(mocks, 99)) * 1.05, 200)
    curves = np.array([[np.mean(row <= g) for g in grid] for row in mocks])
    ax.fill_between(grid, np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0),
                    color='C0', alpha=0.2, label='simulated, matched paths (95%)')
    ax.fill_between(grid, np.percentile(curves, 16, axis=0), np.percentile(curves, 84, axis=0),
                    color='C0', alpha=0.35, label='68%')
    ax.plot(grid, np.median(curves, axis=0), color='C0', lw=1)
    x = np.sort(obs_values)
    ax.step(x, np.arange(1, x.size + 1) / x.size, where='post', color=OBS_COLOR, lw=1.8,
            label=f'observed ({x.size} sightlines)')
    ax.set_xlabel(r'$dn/dz$ above $\log N = %.1f$' % logN_min)
    ax.set_ylabel('cumulative fraction')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')

    # The scatter itself: one observed number against its mock distribution.
    spreads = mocks.std(axis=1)
    axd.hist(spreads, bins=30, color='C0', alpha=0.6, label='simulated')
    axd.axvline(obs_values.std(), color=OBS_COLOR, lw=2, label='observed')
    pct = 100.0 * np.mean(spreads < obs_values.std())
    axd.set_xlabel(r'std of $dn/dz$ across sightlines')
    axd.set_ylabel('realisations')
    axd.set_title(f'observed spread at the {pct:.0f}th percentile')
    axd.grid(alpha=0.25)
    axd.legend(fontsize=8)

    fig.suptitle(label)
    fig.tight_layout()
    _save(fig, out_path)
    return pct


def self_test():
    # dX/dz is 1 at z = 0 by construction, and the integral is monotone.
    assert abs(dx_dz(0.0) - 1.0) < 1e-12
    assert absorption_distance(0.0, 0.1) > 0
    assert absorption_distance(0.1, 0.0) == 0.0
    near = absorption_distance(0.0, 0.1)
    far = absorption_distance(0.3, 0.4)
    assert far > near, (near, far)     # dX/dz grows with z

    # The red window runs past the quasar, so the proximity zone has to trim it.
    entry = {'z_em': 0.4, 'windows': [(1219.0, 1500.0), (1520.0, 1700.0)]}
    ranges = forest_ranges(entry)
    assert ranges[0][0] > 0.002 and ranges[-1][1] < 0.4
    assert ranges[-1][1] < 1700.0 / obs.LYA - 1.0
    assert abs(ranges[-1][1] - (0.4 - 1.4 * 3000.0 / obs.C_KMS)) < 1e-3, ranges[-1]

    safe = forest_ranges(entry, lyb_safe=True)
    lam_lo = LYB * (1 + 0.4)
    assert all(lo >= lam_lo / obs.LYA - 1.0 - 1e-9 for lo, _ in safe), safe
    assert sum(hi - lo for lo, hi in safe) < sum(hi - lo for lo, hi in ranges)

    rows = [
        {'sightline': 'a', 'ion_key': 'HI', 'z': '0.10', 'b': '25', 'b_err': '3',
         'logN': '13.5', 'logN_err': '0.1'},
        {'sightline': 'a', 'ion_key': 'HI', 'z': '0.10', 'b': '25', 'b_err': '99',
         'logN': '13.5', 'logN_err': '0.1'},          # badly constrained
        {'sightline': 'a', 'ion_key': 'HI', 'z': '0.90', 'b': '25', 'b_err': '3',
         'logN': '13.5', 'logN_err': '0.1'},          # outside the forest
        {'sightline': 'a', 'ion_key': 'CIV', 'z': '0.10', 'b': '25', 'b_err': '3',
         'logN': '13.5', 'logN_err': '0.1'},          # not H I
    ]
    kept = select_hi(rows, {'a': [(0.05, 0.20)]})
    assert len(kept) == 1 and kept[0]['logN'] == 13.5, kept

    edges = np.arange(13.0, 14.4, 0.2)
    counts, f_n, err = observed_cddf(np.array([13.1, 13.3, 13.3]), 2.0, edges)
    assert counts[0] == 1 and counts[1] == 2
    # f(N) falls with N at fixed counts because the linear bin width grows.
    assert f_n[1] < counts[1] / counts[0] * f_n[0]
    assert np.all(err[counts > 0] > 0)

    table = {'log10_N_HI': np.array([13.1, 13.3]), 'delta_log_N': np.array([0.2, 0.2])}
    b = sim_bins(table)
    assert np.allclose(b, [13.0, 13.2, 13.4]), b

    # Per-sightline counting: one clean feature per sightline, found and thresholded.
    tau = np.zeros((3, 400))
    v = np.arange(400) * 5.0
    tau[0] += 2.0 * np.exp(-0.5 * ((v - 500) / 30.0) ** 2)      # strong
    tau[1] += 2.0 * np.exp(-0.5 * ((v - 500) / 30.0) ** 2)
    tau[1] += 2.0 * np.exp(-0.5 * ((v - 1500) / 30.0) ** 2)     # two features
    counts = sightline_absorbers(tau, 5.0, logN_min=13.0)
    assert list(counts) == [1, 2, 0], counts
    # A high threshold rejects everything.
    assert sightline_absorbers(tau, 5.0, logN_min=20.0).sum() == 0

    # Matched-path mocks: more path per sightline means less scatter between them.
    rng = np.random.default_rng(0)
    sim_counts = rng.poisson(1.0, 500)
    short = mock_dndz(sim_counts, 0.01, np.full(20, 0.05), 200, rng)
    long = mock_dndz(sim_counts, 0.01, np.full(20, 0.40), 200, rng)
    assert short.std(axis=1).mean() > long.std(axis=1).mean()

    print('obs_cddf self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--components', default='output/analysis/Obs/vpfit_components.csv')
    parser.add_argument('--cos-list', default='data/Obs/cos.list')
    parser.add_argument('--ex-root', default='output/analysis/IllustrisTNG/EX')
    parser.add_argument('--snap', default='snap-090')
    parser.add_argument('--max-b-err', type=float, default=10.0)
    parser.add_argument('--max-logN-err', type=float, default=0.5)
    parser.add_argument('--proximity', type=float, default=3000.0, help='km/s')
    parser.add_argument('--lyb-safe', action='store_true',
                        help='restrict to lambda > 1025.7 (1 + z_em), free of Lyman-beta')
    parser.add_argument('--out-dir', default=None, help='default plots/ex_obs/<snap>')
    parser.add_argument('--per-sightline', metavar='SPECTRA',
                        help='sim spectra HDF5: adds the per-sightline dn/dz scatter')
    parser.add_argument('--logN-min', type=float, default=13.5,
                        help='column density threshold for dn/dz (default 13.5)')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    table = obs.load_cos_list(args.cos_list)
    rows = read_components(args.components)
    sightlines = sorted({r['sightline'] for r in rows})

    ranges, total_dx, per_line = {}, 0.0, []
    for name in sightlines:
        entry = table.get(name)
        if entry is None:
            continue
        ranges[name] = forest_ranges(entry, args.proximity, args.lyb_safe)
        dx = sum(absorption_distance(lo, hi) for lo, hi in ranges[name])
        total_dx += dx
        per_line.append((name, entry['z_em'], dx,
                         sum(hi - lo for lo, hi in ranges[name])))

    kept = select_hi(rows, ranges, args.max_b_err, args.max_logN_err)
    if not kept:
        print('no H I components survive the cuts')
        return 1
    logN = np.array([r['logN'] for r in kept])
    b = np.array([r['b'] for r in kept])
    z = np.array([r['z'] for r in kept])

    ex = {sim: read_table(Path(args.ex_root) / sim / args.snap / 'cddf.csv')
          for sim in EX_SIMS}
    ex_lw = {sim: read_table(Path(args.ex_root) / sim / args.snap / 'line_widths.csv')
             for sim in EX_SIMS}
    edges = sim_bins(ex.get('EX_0'))
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, f_n, err = observed_cddf(logN, total_dx, edges)

    total_dz = sum(p[3] for p in per_line)
    print(f"{len(ranges)} sightlines, dz = {total_dz:.3f}, dX = {total_dx:.3f}"
          f"{' (Lyman-beta safe)' if args.lyb_safe else ''}")
    print(f"{len(kept)} H I components in the forest "
          f"(b_err <= {args.max_b_err}, logN_err <= {args.max_logN_err})")
    print(f"  log N median {np.median(logN):.2f}, b median {np.median(b):.1f} km/s, "
          f"z {z.min():.3f}-{z.max():.3f}")
    print(f"  dn/dz above log N = 13: "
          f"{np.sum(logN >= 13.0) / total_dz:.0f} per unit redshift")

    print(f"\n{'log N':>8}{'count':>7}{'f(N) obs':>12}{'f(N) EX_0':>12}{'ratio':>8}")
    ex0 = ex.get('EX_0')
    for i, c in enumerate(centres):
        if counts[i] == 0:
            continue
        sim_f = np.nan
        if ex0 is not None:
            j = int(np.argmin(np.abs(ex0['log10_N_HI'] - c)))
            sim_f = ex0['f_N_HI'][j]
        ratio = f_n[i] / sim_f if sim_f and np.isfinite(sim_f) and sim_f > 0 else np.nan
        print(f"{c:>8.2f}{counts[i]:>7d}{f_n[i]:>12.3e}{sim_f:>12.3e}{ratio:>8.2f}")

    out_dir = Path(args.out_dir or f'plots/ex_obs/{args.snap}')
    tag = '_lybsafe' if args.lyb_safe else ''

    if args.per_sightline:
        from obs_compare import load_sim_spectra
        tau_sim, z_sim, dv_sim = load_sim_spectra(args.per_sightline)
        dz_box = tau_sim.shape[1] * dv_sim * (1 + z_sim) / obs.C_KMS
        sim_counts = sightline_absorbers(tau_sim, dv_sim, logN_min=args.logN_min)

        obs_dz, obs_values = [], []
        for name, _, _, dz in per_line:
            if dz <= 0:
                continue
            n = sum(1 for r in kept
                    if r['sightline'] == name and r['logN'] >= args.logN_min)
            obs_dz.append(dz)
            obs_values.append(n / dz)
        obs_values = np.array(obs_values)
        obs_dz = np.array(obs_dz)

        mocks = mock_dndz(sim_counts, dz_box, obs_dz, rng=np.random.default_rng(3))
        label = (f'{Path(args.per_sightline).parent.name} at z = {z_sim:.3f}, '
                 f'dn/dz above log N = {args.logN_min}')
        pct = plot_dndz_scatter(obs_values, mocks, out_dir / f'dndz_scatter{tag}.png',
                                label, args.logN_min)

        print(f"\nper-sightline dn/dz (log N >= {args.logN_min}), "
              f"{len(obs_values)} observed sightlines")
        print(f"  simulated box: dz = {dz_box:.4f}, "
              f"{sim_counts.mean():.2f} absorbers per sightline")
        print(f"  observed  median {np.median(obs_values):>6.1f}  "
              f"mean {obs_values.mean():>6.1f}  std {obs_values.std():>6.1f}")
        print(f"  simulated median {np.median(mocks):>6.1f}  "
              f"mean {mocks.mean():>6.1f}  std {mocks.std(axis=1).mean():>6.1f}")
        print(f"  observed spread sits at the {pct:.0f}th percentile of the mock spreads")
    plot_cddf(ex, (centres, f_n, err, counts), out_dir / f'cddf_obs{tag}.png', args.snap,
              ' (Lyman-beta safe)' if args.lyb_safe else '')
    plot_b(ex_lw, b, out_dir / f'b_params_obs{tag}.png', args.snap)
    return 0


if __name__ == '__main__':
    sys.exit(main())
