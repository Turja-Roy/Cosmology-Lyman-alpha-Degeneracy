"""
Overlay plots: the observed scatter on top of the simulated scatter, cell by cell.

Nothing here bins or averages along the comparison axis. Each simulated sightline is a point
and each detected absorber is a point; the fifteen observed chunks and forty-odd observed
absorbers go on top of that cloud. Where the observation sits inside the cloud is the
measurement, and how the cloud moves across the S/N and LSF grid is the answer to what
survives the instrument.

Three families, matching scripts/obs_compare.py's output:
  chunk clouds      tau_eff vs absorber count, vs longest gap, EW sum vs count
  absorber clouds   EW vs velocity width (the fitting-free stand-in for b-N), vs depth, vs z
  sample-size band  observed ECDF over the band of ECDFs from bootstrapped 15-chunk draws

Reads the CSVs written by obs_compare and needs no simulation files of its own:

    python scripts/obs_scatter.py 'output/obs/pg1048_all_lsf-*' --out-dir plots/obs
    python scripts/obs_scatter.py --self-test
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# (x, y, xlabel, ylabel) for the clouds. Pairs, not single statistics: a pair shows whether
# the observation is off in a direction, which a one-dimensional histogram hides.
CHUNK_PAIRS = [
    ('n_absorbers', 'tau_eff', 'absorbers per chunk', r'$\tau_{\rm eff}$'),
    ('gap_max', 'tau_eff', 'longest transmitted gap [km/s]', r'$\tau_{\rm eff}$'),
    ('n_absorbers', 'ew_sum', 'absorbers per chunk', r'$\sum W_{\rm rest}$ [m$\AA$]'),
]
ABSORBER_PAIRS = [
    ('width', 'ew', 'velocity width [km/s]', r'$W_{\rm rest}$ [m$\AA$]'),
    ('depth', 'ew', 'line depth $1 - F_{\\rm min}$', r'$W_{\rm rest}$ [m$\AA$]'),
    ('z', 'ew', 'absorption redshift', r'$W_{\rm rest}$ [m$\AA$]'),
]
CELL_RE = re.compile(r'_lsf-(?P<lsf>[^_]+)_snr-(?P<snr>[^_]+)_')


def _setup_style():
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def read_csv(path):
    with open(path, newline='') as fh:
        return [{k: (float(v) if v not in ('', 'nan') else np.nan) for k, v in row.items()}
                for row in csv.DictReader(fh)]


def column(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def find_cells(patterns):
    """Group obs_compare's CSVs into cells keyed by (lsf, snr)."""
    cells = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern + '*_chunks.csv') + glob.glob(pattern)):
            name = Path(path).name
            if name.endswith('_sim_chunks.csv') or not name.endswith('_chunks.csv'):
                continue
            match = CELL_RE.search(name)
            if not match:
                continue
            stem = path[: path.index('_chunks.csv')]
            key = (match.group('lsf'), match.group('snr'))
            cells[key] = {
                'chunks': read_csv(path),
                'sim_chunks': read_csv(f'{stem}_sim_chunks.csv'),
                'obs_absorbers': read_csv(f'{stem}_obs_absorbers.csv'),
                'sim_absorbers': read_csv(f'{stem}_sim_absorbers.csv'),
            }
    return cells


def snr_order(key):
    """Sort order for the ladder: observed errors first, then decreasing S/N."""
    lsf, snr = key
    rank = {'data': -1, 'inf': 0}.get(snr)
    return (lsf, rank if rank is not None else -1.0 / max(float(snr), 1e-9))


def cloud(ax, sim_x, sim_y, obs_x, obs_y, contours=True):
    """Simulated points with density contours, observed points on top."""
    ok = np.isfinite(sim_x) & np.isfinite(sim_y)
    sim_x, sim_y = sim_x[ok], sim_y[ok]
    if sim_x.size:
        ax.plot(sim_x, sim_y, '.', ms=1.5, alpha=0.15, color='C0', rasterized=True,
                label='simulated')
        if contours and sim_x.size > 50:
            # A 2D histogram only sets the contour levels; the points themselves stay unbinned.
            counts, xe, ye = np.histogram2d(sim_x, sim_y, bins=28)
            counts = gaussian_filter(counts, 1.0)   # otherwise the contours trace shot noise
            levels = np.unique(np.percentile(counts[counts > 0], [50, 84, 97.5]))
            ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), counts.T,
                       levels=levels, colors='C0', linewidths=0.8, alpha=0.8)
    ok = np.isfinite(obs_x) & np.isfinite(obs_y)
    ax.plot(obs_x[ok], obs_y[ok], 'o', ms=5, mfc='none', mew=1.4, color='C3',
            label='observed')


def plot_clouds(cell, pairs, sim_key, obs_key, out_path, title):
    """One row of clouds for a single cell of the grid."""
    _setup_style()
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.8))
    for ax, (xk, yk, xl, yl) in zip(np.atleast_1d(axes), pairs):
        cloud(ax, column(cell[sim_key], xk), column(cell[sim_key], yk),
              column(cell[obs_key], xk if obs_key.startswith('obs_abs') else f'obs_{xk}'),
              column(cell[obs_key], yk if obs_key.startswith('obs_abs') else f'obs_{yk}'))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].legend(loc='best')
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path)


def plot_grid(cells, pairs, sim_key, obs_key, out_path, title):
    """The same cloud repeated across the grid: columns are S/N rungs, rows are pairs."""
    _setup_style()
    keys = sorted(cells, key=snr_order)
    fig, axes = plt.subplots(len(pairs), len(keys),
                             figsize=(3.4 * len(keys), 3.2 * len(pairs)),
                             squeeze=False)
    for col, key in enumerate(keys):
        cell = cells[key]
        for row, (xk, yk, xl, yl) in enumerate(pairs):
            ax = axes[row][col]
            cloud(ax, column(cell[sim_key], xk), column(cell[sim_key], yk),
                  column(cell[obs_key], xk if obs_key.startswith('obs_abs') else f'obs_{xk}'),
                  column(cell[obs_key], yk if obs_key.startswith('obs_abs') else f'obs_{yk}'),
                  contours=False)
            if row == 0:
                ax.set_title(f'LSF {key[0]}, S/N {key[1]}')
            if row == len(pairs) - 1:
                ax.set_xlabel(xl)
            if col == 0:
                ax.set_ylabel(yl)
            ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path)


def ecdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    return v, np.arange(1, v.size + 1) / max(v.size, 1)


def plot_sample_band(cell, stat, out_path, title, n_boot=400, seed=4):
    """Observed ECDF against the band of ECDFs from draws of the observed sample size.

    The simulation has thousands of sightlines and the observation has fifteen chunks, so the
    honest comparison is against repeated fifteen-sightline draws, not against the full
    simulated ensemble.
    """
    _setup_style()
    rng = np.random.default_rng(seed)
    obs_vals = column(cell['chunks'], f'obs_{stat}')
    sim_vals = column(cell['sim_chunks'], stat)
    sim_vals = sim_vals[np.isfinite(sim_vals)]
    n = int(np.isfinite(obs_vals).sum())
    if n < 3 or sim_vals.size < n:
        return

    grid = np.linspace(np.nanmin(sim_vals), np.nanmax(sim_vals), 200)
    draws = np.empty((n_boot, grid.size))
    for i in range(n_boot):
        sample = rng.choice(sim_vals, size=n, replace=False)
        draws[i] = np.searchsorted(np.sort(sample), grid, side='right') / n

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.fill_between(grid, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0),
                    color='C0', alpha=0.2, label=f'simulated, {n}-chunk draws (95%)')
    ax.fill_between(grid, np.percentile(draws, 16, axis=0), np.percentile(draws, 84, axis=0),
                    color='C0', alpha=0.35, label='68%')
    ax.plot(grid, np.median(draws, axis=0), color='C0', lw=1)
    x, y = ecdf(obs_vals)
    ax.step(x, y, where='post', color='C3', lw=1.8, label='observed')
    ax.set_xlabel(stat)
    ax.set_ylabel('cumulative fraction')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save(fig, out_path)


def cloud_distance(cell, pairs, sim_key, obs_key, n_boot=200, seed=7):
    """How far the observed cloud sits from simulated clouds of the same sample size.

    Mean nearest-neighbour distance from the observed points to the simulated cloud, compared
    against the same distance for simulated subsamples. The p-value is the fraction of
    subsamples that are further out, so small means the observation is unusually far.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for xk, yk, _, _ in pairs:
        prefix = '' if obs_key.startswith('obs_abs') else 'obs_'
        o = np.column_stack([column(cell[obs_key], f'{prefix}{xk}'),
                             column(cell[obs_key], f'{prefix}{yk}')])
        s = np.column_stack([column(cell[sim_key], xk), column(cell[sim_key], yk)])
        o = o[np.isfinite(o).all(axis=1)]
        s = s[np.isfinite(s).all(axis=1)]
        if o.shape[0] < 3 or s.shape[0] < 20 * o.shape[0]:
            continue
        scale = np.std(s, axis=0)
        scale[scale == 0] = 1.0

        def mean_nn(points, cloud_points):
            d = np.linalg.norm((points[:, None, :] - cloud_points[None, :, :]) / scale, axis=2)
            return float(np.mean(np.min(d, axis=1)))

        # The reference cloud and the null draws must be disjoint: a point measured against a
        # cloud that contains it has nearest-neighbour distance zero, which would make every
        # simulated draw look infinitely closer than the observation.
        n = o.shape[0]
        order = rng.permutation(s.shape[0])
        ref_idx, pool_idx = order[: s.shape[0] // 2], order[s.shape[0] // 2:]
        reference = s[ref_idx[:2000]]
        observed = mean_nn(o, reference)
        null = np.array([mean_nn(s[rng.choice(pool_idx, size=n, replace=False)], reference)
                         for _ in range(n_boot)])
        out[f'{yk}_vs_{xk}'] = {'observed': observed,
                                'null_median': float(np.median(null)),
                                'p': float(np.mean(null >= observed))}
    return out


def self_test():
    rng = np.random.default_rng(0)

    def chunk_rows(n, shift=0.0):
        return [{'tau_eff': abs(rng.normal(0.03 + shift, 0.01)),
                 'n_absorbers': float(rng.poisson(3)),
                 'ew_sum': abs(rng.normal(150, 50)),
                 'gap_max': abs(rng.normal(1300, 300))} for _ in range(n)]

    def absorber_rows(n, width=60.0):
        return [{'ew': abs(rng.normal(80, 30)), 'z': rng.uniform(0.02, 0.15),
                 'width': abs(rng.normal(width, 15)), 'depth': rng.uniform(0.2, 0.9),
                 'ew_snr': rng.uniform(3, 20)} for _ in range(n)]

    sim = chunk_rows(600)
    obs = [{f'obs_{k}': v for k, v in r.items()} for r in chunk_rows(15)]
    cell = {'chunks': obs, 'sim_chunks': sim,
            'obs_absorbers': absorber_rows(40), 'sim_absorbers': absorber_rows(1500)}

    # ECDF of a sorted sample is monotone and ends at 1.
    x, y = ecdf([3, 1, 2, np.nan])
    assert np.all(np.diff(x) > 0) and y[-1] == 1.0

    # A cloud drawn from the simulated distribution is not flagged; a displaced one is.
    same = cloud_distance(cell, CHUNK_PAIRS, 'sim_chunks', 'chunks', n_boot=60)
    assert same, 'no pairs evaluated'
    assert all(v['p'] > 0.01 for v in same.values()), same

    shifted = {**cell, 'chunks': [{f'obs_{k}': v for k, v in r.items()}
                                  for r in chunk_rows(15, shift=0.25)]}
    away = cloud_distance(shifted, CHUNK_PAIRS, 'sim_chunks', 'chunks', n_boot=60)
    assert away['tau_eff_vs_n_absorbers']['p'] < 0.05, away['tau_eff_vs_n_absorbers']
    assert (away['tau_eff_vs_n_absorbers']['observed']
            > same['tau_eff_vs_n_absorbers']['observed'])

    # Figures render without touching the project's output directories.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        plot_clouds(cell, CHUNK_PAIRS, 'sim_chunks', 'chunks', f'{tmp}/chunks.png', 'test')
        plot_clouds(cell, ABSORBER_PAIRS, 'sim_absorbers', 'obs_absorbers',
                    f'{tmp}/absorbers.png', 'test')
        plot_sample_band(cell, 'tau_eff', f'{tmp}/band.png', 'test', n_boot=50)
        plot_grid({('gauss', 'data'): cell, ('gauss', '10'): cell},
                  CHUNK_PAIRS, 'sim_chunks', 'chunks', f'{tmp}/grid.png', 'test')
        assert Path(f'{tmp}/grid.png').exists()

    print('obs_scatter self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('patterns', nargs='*',
                        help="path stem or glob of obs_compare output, e.g. "
                             "'output/obs/pg1048_all_lsf-*'")
    parser.add_argument('--out-dir', default='plots/obs')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.patterns:
        parser.error('need a path stem or glob (or --self-test)')

    cells = find_cells(args.patterns)
    if not cells:
        print('no obs_compare CSVs matched')
        return 1
    print(f"{len(cells)} cell(s): " + ', '.join(f'LSF {k[0]} / S/N {k[1]}'
                                                for k in sorted(cells, key=snr_order)))

    for key in sorted(cells, key=snr_order):
        cell = cells[key]
        tag = f'lsf-{key[0]}_snr-{key[1]}'
        label = f'LSF {key[0]}, S/N {key[1]}'
        plot_clouds(cell, CHUNK_PAIRS, 'sim_chunks', 'chunks',
                    f'{args.out_dir}/chunk_clouds_{tag}.png', f'chunk scatter — {label}')
        plot_clouds(cell, ABSORBER_PAIRS, 'sim_absorbers', 'obs_absorbers',
                    f'{args.out_dir}/absorber_clouds_{tag}.png',
                    f'absorber scatter — {label}')
        for stat in ['tau_eff', 'n_absorbers', 'ew_sum', 'gap_max']:
            plot_sample_band(cell, stat, f'{args.out_dir}/ecdf_{stat}_{tag}.png',
                             f'{stat} — {label}')

        print(f'\n{label}: mean nearest-neighbour distance of the observed cloud')
        print(f"  {'pair':<28}{'observed':>10}{'null med':>10}{'p':>8}")
        for name, v in {**cloud_distance(cell, CHUNK_PAIRS, 'sim_chunks', 'chunks'),
                        **cloud_distance(cell, ABSORBER_PAIRS, 'sim_absorbers',
                                         'obs_absorbers')}.items():
            print(f"  {name:<28}{v['observed']:>10.3f}{v['null_median']:>10.3f}{v['p']:>8.3f}")

    if len(cells) > 1:
        plot_grid(cells, CHUNK_PAIRS, 'sim_chunks', 'chunks',
                  f'{args.out_dir}/ladder_chunk_clouds.png',
                  'chunk scatter across the S/N and LSF grid')
        plot_grid(cells, ABSORBER_PAIRS, 'sim_absorbers', 'obs_absorbers',
                  f'{args.out_dir}/ladder_absorber_clouds.png',
                  'absorber scatter across the S/N and LSF grid')
    return 0


if __name__ == '__main__':
    sys.exit(main())
