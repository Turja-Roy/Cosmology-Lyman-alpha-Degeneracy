"""
Every EX variant against the observation, for each S/N rung, TNG beside SIMBA.

Reads the mock catalogues from scripts/sim_mock_lines.py
(<mock-root>/<suite>/EX_k/[<sub>/]mock_{lines,sightlines}_snr-<rung>.csv) and the observed
spectra and VPFIT fits from data/Obs/, so it needs no simulation files and runs in seconds.

Both sides get the same cuts: H I only, b_err <= 10, logN_err <= 0.5, inside the cos.list
windows, outside the proximity zone, z <= z_max. The absorption path dX comes from the same
windows on both sides, since a mock covers exactly its template's path; the flux pixels come
from the same masks.

Rows are S/N rungs. In the 'data' row every mock carries its template's own errors and the
observation is the whole sample. In a fixed-S/N row the observation is the subset of
sightlines whose own S/N per pixel falls in that rung's bin (edges at the geometric midpoints
between rungs), and the mocks are restricted to the same templates, so path, redshift mix and
masking match as well as noise. The full observed sample is drawn dashed for reference.

Bands: simulation, 16-84% over one-survey draws (one mock per template); observation,
16-84% over a bootstrap of sightlines. Variant-to-variant differences share initial
conditions and need a paired bootstrap, which these bands are not.

    python scripts/ex_snr_plots.py --mock-root output/obs/mock_lines --out plots/obs/ex_snr
    python scripts/ex_snr_plots.py --sub uvb_match --out plots/obs/ex_snr_uvb_match
    python scripts/ex_snr_plots.py --self-test
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observations as obs                                       # noqa: E402
import vpfit_catalog as vp                                       # noqa: E402
from obs_cddf import absorption_distance, forest_ranges          # noqa: E402
from mock_lines_report import SIMS, MAX_B_ERR, MAX_LOGN_ERR      # noqa: E402
from sim_mock_lines import FLUX_EDGES, Z_MAX, load_templates     # noqa: E402

SUITES = ['IllustrisTNG', 'SIMBA']
RUNGS = ['data', '10', '20', '40', '80']
# Fixed-order categorical slots that pass the colour-blind separation check; the tab10
# red/green pair used elsewhere does not.
COLOURS = {'EX_0': '#2a78d6', 'EX_1': '#eb6834', 'EX_2': '#1baf7a', 'EX_3': '#eda100'}

LOGN_EDGES = np.round(np.arange(12.0, 17.0 + 1e-9, 0.05), 2)
B_EDGES = np.arange(0.0, 201.0, 1.0)
BN_EDGES = np.arange(12.5, 15.5 + 1e-9, 0.25)
HIST_KEYS = ('logN', 'b', 'bN', 'flux')
N_DRAWS = 200


def unit_hist(unit, values, edges, n_units):
    """Histogram of `values` per unit (one row per mock or per observed sightline)."""
    k = np.digitize(values, edges) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    out = np.zeros((n_units, len(edges) - 1))
    np.add.at(out, (unit[ok], k[ok]), 1.0)
    return out


def build_units(names, unit, z, b, logN, flux_counts, dx, b_logn_min):
    """Per-unit histograms; every statistic is rebuilt from sums of these."""
    n = len(names)
    unit, b, logN = np.asarray(unit, int), np.asarray(b, float), np.asarray(logN, float)
    strong = logN >= b_logn_min
    bn = np.zeros((n, len(BN_EDGES) - 1, len(B_EDGES) - 1))
    kn, kb = np.digitize(logN, BN_EDGES) - 1, np.digitize(b, B_EDGES) - 1
    ok = (kn >= 0) & (kn < bn.shape[1]) & (kb >= 0) & (kb < bn.shape[2])
    np.add.at(bn, (unit[ok], kn[ok], kb[ok]), 1.0)
    return {'names': np.asarray(names), 'dx': np.asarray(dx, float),
            'logN': unit_hist(unit, logN, LOGN_EDGES, n),
            'b': unit_hist(unit[strong], b[strong], B_EDGES, n),
            'bN': bn, 'flux': np.asarray(flux_counts, float)}


def _grouped(counts, edges, group):
    c = counts.reshape(*counts.shape[:-1], -1, group).sum(-1)
    e = edges[::group]
    return 0.5 * (e[1:] + e[:-1]), c, np.diff(e)


def _density(counts, edges, group):
    x, c, w = _grouped(counts, edges, group)
    return x, c / np.maximum(c.sum(-1, keepdims=True), 1) / w


def _cdf(counts, edges):
    c = np.cumsum(counts, -1)
    return edges[1:], c / np.maximum(c[..., -1:], 1)


def cddf(h, dx):
    """log f(N) = log d2n / dN dX, in 0.25 dex bins."""
    x, c, _ = _grouped(h['logN'], LOGN_EDGES, 5)
    e = LOGN_EDGES[::5]
    with np.errstate(divide='ignore', invalid='ignore'):
        f = c / (np.diff(10.0 ** e) * np.asarray(dx, float)[..., None])
        return x, np.where(c > 0, np.log10(f), np.nan)


def median_b(h, dx):
    """Median b in each log N bin, from the per-unit b histograms; NaN below 5 lines."""
    cum = np.cumsum(h['bN'], -1)
    tot = cum[..., -1]
    k = np.argmax(cum >= 0.5 * tot[..., None], axis=-1)
    x = 0.5 * (BN_EDGES[1:] + BN_EDGES[:-1])
    return x, np.where(tot >= 5, B_EDGES[k] + 0.5, np.nan)


# name -> (curve, x label, y label, log y, x range)
STATS = {
    'cddf': (cddf, r'$\log N_{\rm HI}$ [cm$^{-2}$]',
             r'$\log f(N_{\rm HI}, X)$', False, (12.0, 17.0)),
    'logN_pdf': (lambda h, dx: _density(h['logN'], LOGN_EDGES, 2),
                 r'$\log N_{\rm HI}$', 'PDF', False, (12.0, 16.5)),
    'logN_cdf': (lambda h, dx: _cdf(h['logN'], LOGN_EDGES),
                 r'$\log N_{\rm HI}$', 'CDF', False, (12.0, 16.5)),
    'b_pdf': (lambda h, dx: _density(h['b'], B_EDGES, 5),
              r'$b$ [km s$^{-1}$]', 'PDF', False, (0.0, 120.0)),
    'b_cdf': (lambda h, dx: _cdf(h['b'], B_EDGES),
              r'$b$ [km s$^{-1}$]', 'CDF', False, (0.0, 120.0)),
    'b_vs_logN': (median_b, r'$\log N_{\rm HI}$', r'median $b$ [km s$^{-1}$]', False,
                  (12.5, 15.5)),
    'flux_pdf': (lambda h, dx: _density(h['flux'], FLUX_EDGES, 2),
                 'normalised flux', 'PDF', True, (-0.1, 1.2)),
    'flux_cdf': (lambda h, dx: _cdf(h['flux'], FLUX_EDGES),
                 'normalised flux', 'CDF', False, (-0.1, 1.2)),
}


def evaluate(units, central, draws):
    """Central curves over the `central` units and 16-84% bands over the `draws`."""
    def summed(idx):
        return {k: units[k][idx].sum(axis=0) for k in HIST_KEYS}, units['dx'][idx].sum()

    h, dx = summed(central)
    centre = {s: spec[0](h, dx) for s, spec in STATS.items()}
    samples = {s: [] for s in STATS}
    for idx in draws:
        h, dx = summed(idx)
        for s, spec in STATS.items():
            samples[s].append(spec[0](h, dx)[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        band = {s: np.nanpercentile(np.array(v), [16, 84], axis=0) for s, v in samples.items()}
    return {'centre': centre, 'band': band}


def mock_draws(names, keep, rng, n=N_DRAWS):
    """One-survey draws: one mock per selected template."""
    groups = [np.flatnonzero(names == t) for t in keep]
    groups = [g for g in groups if g.size]
    central = np.concatenate(groups) if groups else np.array([], int)
    return central, [np.array([rng.choice(g) for g in groups]) for _ in range(n)]


def bootstrap_draws(names, keep, rng, n=N_DRAWS):
    central = np.flatnonzero(np.isin(names, list(keep)))
    return central, [rng.choice(central, size=central.size) for _ in range(n)]


def snr_bin(rung, snr_pixel):
    """Observed sightlines matching a fixed rung: S/N per pixel between the geometric
    midpoints to the neighbouring rungs."""
    levels = [float(r) for r in RUNGS if r != 'data']
    i = levels.index(float(rung))
    lo = np.sqrt(levels[i - 1] * levels[i]) if i > 0 else 0.0
    hi = np.sqrt(levels[i] * levels[i + 1]) if i + 1 < len(levels) else np.inf
    return (snr_pixel >= lo) & (snr_pixel < hi)


def in_ranges(z, ranges):
    return any(lo <= z <= hi for lo, hi in ranges)


def template_ranges(templates, cos_list, z_max):
    """Usable forest redshift intervals per sightline, the same ones the masks use."""
    meta = obs.load_cos_list(cos_list)
    out = {}
    for t in templates:
        r = forest_ranges(meta[t['name']]) if t['name'] in meta else []
        out[t['name']] = [(lo, min(hi, z_max)) for lo, hi in r if lo < z_max]
        if not out[t['name']]:
            print(f"warning: {t['name']} has no cos.list windows; it adds no path")
    return out


def observed_units(templates, ranges, dx, b_logn_min):
    names = [t['name'] for t in templates]
    fits = [str(t['fit']) for t in templates]
    unit, z, b, logN = [], [], [], []
    for r in vp.catalogue(fits):
        if r['ion_key'] != 'HI' or r['b'] <= 0:
            continue
        if r['b_err'] > MAX_B_ERR or r['logN_err'] > MAX_LOGN_ERR:
            continue
        if not in_ranges(r['z'], ranges[r['sightline']]):
            continue
        unit.append(names.index(r['sightline']))
        z.append(r['z'])
        b.append(r['b'])
        logN.append(r['logN'])
    flux = [np.histogram(t['flux'][t['good']], FLUX_EDGES)[0] for t in templates]
    return build_units(names, unit, z, b, logN, flux, [dx[n] for n in names], b_logn_min)


def mock_units(lines_path, sightlines_path, ranges, dx, b_logn_min):
    with open(sightlines_path) as fh:
        sl = list(csv.DictReader(fh))
    if sl and 'flux_counts' not in sl[0]:
        raise SystemExit(f'{sightlines_path} predates the flux histograms; rerun sim_mock_lines')
    index = {r['mock']: i for i, r in enumerate(sl)}
    names = [r['template'] for r in sl]
    unit, z, b, logN = [], [], [], []
    with open(lines_path) as fh:
        for r in csv.DictReader(fh):
            if r['ion_key'] != 'HI' or float(r['b']) <= 0:
                continue
            if float(r['b_err']) > MAX_B_ERR or float(r['logN_err']) > MAX_LOGN_ERR:
                continue
            if not in_ranges(float(r['z']), ranges[r['template']]):
                continue
            unit.append(index[r['mock']])
            z.append(float(r['z']))
            b.append(float(r['b']))
            logN.append(float(r['logN']))
    flux = [np.array(r['flux_counts'].split(), dtype=float) for r in sl]
    return build_units(names, unit, z, b, logN, flux, [dx[n] for n in names], b_logn_min)


def plot_stat(name, mocks, observed, out_dir):
    _, xlab, ylab, logy, xlim = STATS[name]
    fig, axes = plt.subplots(len(RUNGS), len(SUITES), figsize=(10.0, 2.4 * len(RUNGS)),
                             sharex=True, sharey=True, squeeze=False)
    for i, rung in enumerate(RUNGS):
        o = observed[rung]
        for j, suite in enumerate(SUITES):
            ax = axes[i, j]
            drawn = 0
            for sim in SIMS:
                r = mocks.get((suite, sim, rung))
                if r is None:
                    continue
                x, y = r['centre'][name]
                lo, hi = r['band'][name]
                ax.fill_between(x, lo, hi, color=COLOURS[sim], alpha=0.18, lw=0)
                ax.plot(x, y, color=COLOURS[sim], lw=1.5)
                drawn += 1
            if not drawn:
                ax.text(0.5, 0.5, 'no mocks', transform=ax.transAxes, ha='center',
                        va='center', color='0.45')
            if rung != 'data':
                ax.plot(*observed['data']['centre'][name], color='0.35', lw=1.0, ls='--')
            x, y = o['centre'][name]
            lo, hi = o['band'][name]
            ax.fill_between(x, lo, hi, color='0.55', alpha=0.25, lw=0)
            ax.plot(x, y, color='k', lw=1.8)

            noise = 'own error arrays' if rung == 'data' else f'S/N {rung} per pixel'
            ax.set_title(f'{suite}: {noise}, {o["n"]} observed sightlines', fontsize=9)
            ax.set_xlim(*xlim)
            if logy:
                ax.set_yscale('log')
            if name.endswith('_cdf'):
                ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2, lw=0.5)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == len(RUNGS) - 1:
                ax.set_xlabel(xlab)

    handles = [Line2D([], [], color=COLOURS[s], lw=1.5, label=f'{s} {SIMS[s]}') for s in SIMS]
    handles += [Line2D([], [], color='k', lw=1.8, label='observed, S/N-matched'),
                Line2D([], [], color='0.35', lw=1.0, ls='--', label='observed, all')]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = Path(out_dir) / f'{name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def self_test():
    rng = np.random.default_rng(0)

    h = unit_hist(np.array([0, 0, 1, 1, 1]), np.array([0.5, 1.5, 1.5, 9.0, 1.2]),
                  np.array([0.0, 1.0, 2.0]), 2)
    assert h.tolist() == [[1, 1], [0, 2]], h

    # Ten lines in one 0.25 dex bin over dX = 2.
    u = build_units(['a'], np.zeros(10), np.zeros(10), np.full(10, 25.0), np.full(10, 13.6),
                    [np.zeros(FLUX_EDGES.size - 1)], [2.0], 13.2)
    x, f = cddf({k: u[k][0] for k in HIST_KEYS}, 2.0)
    k = np.argmin(np.abs(x - 13.625))
    assert abs(10 ** f[k] - 10 / ((10 ** 13.75 - 10 ** 13.5) * 2.0)) < 1e-6 * 10 ** f[k]
    assert np.isnan(f[0])

    # Medians from the histograms land within a bin of the true median.
    u = build_units(['a'], np.zeros(5), np.zeros(5), np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
                    np.full(5, 13.1), [np.zeros(FLUX_EDGES.size - 1)], [1.0], 13.2)
    x, m = median_b({k: u[k][0] for k in HIST_KEYS}, 1.0)
    assert abs(m[np.argmin(np.abs(x - 13.125))] - 30.0) <= 1.0, m
    assert u['b'].sum() == 0              # below the b floor, kept for the b-N medians only

    # One-survey draws take exactly one mock per template.
    names = np.array(['p', 'q', 'p', 'q', 'r'])
    central, draws = mock_draws(names, ['p', 'q'], rng, n=20)
    assert sorted(central) == [0, 1, 2, 3]
    assert all(sorted(names[d]) == ['p', 'q'] for d in draws)

    # Rung bins meet at the geometric midpoints and cover every S/N once.
    s = np.array([5.0, 14.0, 14.2, 28.0, 30.0, 56.0, 57.0, 200.0])
    hits = np.array([snr_bin(r, s) for r in RUNGS[1:]])
    assert (hits.sum(axis=0) == 1).all() and hits[0].tolist()[:2] == [True, True]
    assert hits[3].tolist()[-2:] == [True, True]

    # CDFs end at 1 and densities integrate to 1.
    counts = rng.integers(0, 10, size=FLUX_EDGES.size - 1).astype(float)
    assert abs(_cdf(counts, FLUX_EDGES)[1][-1] - 1.0) < 1e-12
    xx, d = _density(counts, FLUX_EDGES, 2)
    assert abs(np.sum(d * np.diff(FLUX_EDGES[::2])) - 1.0) < 1e-9

    print('ex_snr_plots self-test passed')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mock-root', default='output/obs/mock_lines')
    p.add_argument('--sub', default='', help="mock subdirectory, e.g. 'uvb_match' or 'gauss'")
    p.add_argument('--templates', default='data/Obs/*_all.dat')
    p.add_argument('--cos-list', default='data/Obs/cos.list')
    p.add_argument('--z-max', type=float, default=Z_MAX, help='must match the mocks')
    p.add_argument('--b-logn-min', type=float, default=13.2,
                   help='log N floor for the b PDF and CDF')
    p.add_argument('--out', default='plots/obs/ex_snr')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return 0

    rng = np.random.default_rng(args.seed)
    templates = load_templates(args.templates, args.cos_list, args.z_max)
    ranges = template_ranges(templates, args.cos_list, args.z_max)
    dx = {n: sum(absorption_distance(lo, hi) for lo, hi in r) for n, r in ranges.items()}
    names = np.array([t['name'] for t in templates])
    snr = np.array([t['snr_pixel'] for t in templates])
    print(f'{len(templates)} observed sightlines, total dX = {sum(dx.values()):.2f} '
          f'at z <= {args.z_max}')

    obs_units = observed_units(templates, ranges, dx, args.b_logn_min)
    selection, observed = {}, {}
    for rung in RUNGS:
        keep = names if rung == 'data' else names[snr_bin(rung, snr)]
        if not keep.size:
            print(f'S/N {rung}: no observed sightline in the bin; using all of them')
            keep = names
        selection[rung] = keep
        if rung != 'data':
            print(f'S/N {rung} per pixel: {", ".join(keep)}')
        central, draws = bootstrap_draws(obs_units['names'], keep, rng)
        observed[rung] = {**evaluate(obs_units, central, draws), 'n': len(keep)}

    mocks = {}
    for suite in SUITES:
        for sim in SIMS:
            base = Path(args.mock_root) / suite / sim / args.sub
            for rung in RUNGS:
                lines = base / f'mock_lines_snr-{rung}.csv'
                sightlines = base / f'mock_sightlines_snr-{rung}.csv'
                if not (lines.exists() and sightlines.exists()):
                    print(f'missing: {lines}')
                    continue
                units = mock_units(lines, sightlines, ranges, dx, args.b_logn_min)
                central, draws = mock_draws(units['names'], selection[rung], rng)
                mocks[(suite, sim, rung)] = evaluate(units, central, draws)

    for name in STATS:
        plot_stat(name, mocks, observed, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
