"""
Every EX variant against the observation, for each S/N rung.

Reads the mock catalogues from scripts/sim_mock_lines.py
(<mock-root>/<suite>/EX_k/[<sub>/]mock_{lines,sightlines}_snr-<rung>.csv) and the observed
spectra and VPFIT fits from data/Obs/, so it needs no simulation files and runs in seconds.

Both sides get the same cuts: H I only, b_err <= 10, logN_err <= 0.5, inside the cos.list
windows, outside the proximity zone, z <= z_max. The absorption path dX comes from the same
windows on both sides, since a mock covers exactly its template's path; the flux pixels come
from the same masks.

Rows are S/N rungs, per resolution element. In the 'data' row every mock carries its
template's observed error array and the observation is the whole sample. In a fixed-S/N row
the observation is the subset of sightlines whose own S/N per resel falls in that rung's bin
(edges at the geometric midpoints between rungs); the mocks are all of them, so only the noise
changes down a column. The full observed sample is drawn dashed for reference.

Bands: simulation, 16-84% over one-survey draws (one mock per template); observation,
16-84% over a bootstrap of sightlines. Variant-to-variant differences share initial
conditions and need a paired bootstrap, which these bands are not.

Output, per suite with mocks:
    panel_<suite>.png                rows S/N, columns CDDF | flux PDF | b PDF
    b_N_plane_<suite>.png            rows S/N, columns EX models
    <sub>_vs_raw/<stat>_<suite>.png  the panel statistics with and without --sub
    single/<stat>/...                every statistic and rung as its own figure

    python scripts/ex_snr_plots.py                     # uvb_match, compared to raw
    python scripts/ex_snr_plots.py --sub '' --out plots/obs/ex_snr_raw
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
from matplotlib.patches import Patch

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
BN_LOGN = np.arange(12.0, 16.0 + 1e-9, 0.1)
BN_B = np.arange(0.0, 100.0 + 1e-9, 2.0)
HIST_KEYS = ('logN', 'b', 'flux')
N_DRAWS = 200


def unit_hist(unit, values, edges, n_units):
    """Histogram of `values` per unit (one row per mock or per observed sightline)."""
    k = np.digitize(values, edges) - 1
    ok = (k >= 0) & (k < len(edges) - 1)
    out = np.zeros((n_units, len(edges) - 1))
    np.add.at(out, (unit[ok], k[ok]), 1.0)
    return out


def build_units(names, unit, z, b, logN, flux_counts, dx, b_logn_min):
    """Per-unit histograms, rebuilt into every statistic by summing, plus the raw lines."""
    n = len(names)
    unit, b, logN = np.asarray(unit, int), np.asarray(b, float), np.asarray(logN, float)
    strong = logN >= b_logn_min
    return {'names': np.asarray(names), 'dx': np.asarray(dx, float),
            'logN': unit_hist(unit, logN, LOGN_EDGES, n),
            'b': unit_hist(unit[strong], b[strong], B_EDGES, n),
            'flux': np.asarray(flux_counts, float),
            'lines': (unit, logN, b)}


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


# name -> (curve, x label, y label, log y, x range)
STATS = {
    'cddf': (cddf, r'$\log N_{\rm HI}$ [cm$^{-2}$]',
             r'$\log f(N_{\rm HI}, X)$', False, (12.0, 17.0)),
    'flux_pdf': (lambda h, dx: _density(h['flux'], FLUX_EDGES, 2),
                 'normalised flux', 'PDF', True, (-0.1, 1.2)),
    'b_pdf': (lambda h, dx: _density(h['b'], B_EDGES, 5),
              r'$b$ [km s$^{-1}$]', 'PDF', False, (0.0, 120.0)),
    'b_cdf': (lambda h, dx: _cdf(h['b'], B_EDGES),
              r'$b$ [km s$^{-1}$]', 'CDF', False, (0.0, 120.0)),
    'logN_pdf': (lambda h, dx: _density(h['logN'], LOGN_EDGES, 2),
                 r'$\log N_{\rm HI}$', 'PDF', False, (12.0, 16.5)),
    'logN_cdf': (lambda h, dx: _cdf(h['logN'], LOGN_EDGES),
                 r'$\log N_{\rm HI}$', 'CDF', False, (12.0, 16.5)),
    'flux_cdf': (lambda h, dx: _cdf(h['flux'], FLUX_EDGES),
                 'normalised flux', 'CDF', False, (-0.1, 1.2)),
}
PANEL = ['cddf', 'flux_pdf', 'b_pdf']


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
    unit, logN, b = units['lines']
    sel = np.isin(unit, central)
    return {'centre': centre, 'band': band, 'n': len(central), 'bN': (logN[sel], b[sel])}


def mock_draws(names, rng, n=N_DRAWS):
    """All mocks, and one-survey draws of one mock per template."""
    groups = [np.flatnonzero(names == t) for t in np.unique(names)]
    return np.arange(names.size), [np.array([rng.choice(g) for g in groups]) for _ in range(n)]


def bootstrap_draws(names, keep, rng, n=N_DRAWS):
    central = np.flatnonzero(np.isin(names, list(keep)))
    return central, [rng.choice(central, size=central.size) for _ in range(n)]


def snr_bin(rung, snr_resel):
    """Observed sightlines matching a fixed rung: S/N per resel between the geometric
    midpoints to the neighbouring rungs."""
    levels = [float(r) for r in RUNGS if r != 'data']
    i = levels.index(float(rung))
    lo = np.sqrt(levels[i - 1] * levels[i]) if i > 0 else 0.0
    hi = np.sqrt(levels[i] * levels[i + 1]) if i + 1 < len(levels) else np.inf
    return (snr_resel >= lo) & (snr_resel < hi)


def hpd_levels(h, fracs=(0.95, 0.68)):
    """Histogram heights above which the densest cells hold each fraction of the counts."""
    s = np.sort(h.ravel())[::-1]
    c = np.cumsum(s) / max(s.sum(), 1)
    return [s[min(np.searchsorted(c, f), s.size - 1)] for f in fracs]


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


def load_mocks(root, sub, ranges, dx, b_logn_min, rng):
    mocks = {}
    for suite in SUITES:
        for sim in SIMS:
            base = Path(root) / suite / sim / sub
            for rung in RUNGS:
                lines = base / f'mock_lines_snr-{rung}.csv'
                sightlines = base / f'mock_sightlines_snr-{rung}.csv'
                if not (lines.exists() and sightlines.exists()):
                    print(f'missing: {lines}')
                    continue
                units = mock_units(lines, sightlines, ranges, dx, b_logn_min)
                mocks[(suite, sim, rung)] = evaluate(units, *mock_draws(units['names'], rng))
    return mocks


def noise_label(rung):
    return 'observed errors' if rung == 'data' else f'S/N {rung} per resel'


def title(ax, suite, rung, observed, mocks, sim=None):
    n_mock = next((m['n'] for (su, s, r), m in mocks.items()
                   if su == suite and r == rung and sim in (None, s)), 0)
    head = suite if sim is None else f'{suite} {sim}'
    ax.set_title(f'{head}: {noise_label(rung)}\n{observed[rung]["n"]} observed / '
                 f'{n_mock} mock sightlines', fontsize=8)


def style(ax, name, first_col=True, last_row=True):
    _, xlab, ylab, logy, xlim = STATS[name]
    ax.set_xlim(*xlim)
    if logy:
        ax.set_yscale('log')
    if name.endswith('_cdf'):
        ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, lw=0.5)
    if first_col:
        ax.set_ylabel(ylab)
    if last_row:
        ax.set_xlabel(xlab)


def draw_observed(ax, name, rung, observed):
    if rung != 'data':
        ax.plot(*observed['data']['centre'][name], color='0.35', lw=1.0, ls='--')
    x, y = observed[rung]['centre'][name]
    lo, hi = observed[rung]['band'][name]
    ax.fill_between(x, lo, hi, color='0.55', alpha=0.25, lw=0)
    ax.plot(x, y, color='k', lw=1.8)


def draw_stat(ax, name, suite, rung, mocks, observed):
    for sim in SIMS:
        r = mocks.get((suite, sim, rung))
        if r is None:
            continue
        x, y = r['centre'][name]
        lo, hi = r['band'][name]
        ax.fill_between(x, lo, hi, color=COLOURS[sim], alpha=0.18, lw=0)
        ax.plot(x, y, color=COLOURS[sim], lw=1.5)
    draw_observed(ax, name, rung, observed)


def draw_compare(ax, name, suite, sim, rung, mocks, raw, observed):
    for m, ls in ((raw, '-'), (mocks, '--')):
        r = m.get((suite, sim, rung))
        if r is not None:
            ax.plot(*r['centre'][name], color=COLOURS[sim], lw=1.5, ls=ls)
    draw_observed(ax, name, rung, observed)


def draw_bn(ax, suite, sim, rung, mocks, observed):
    r = mocks.get((suite, sim, rung))
    if r is not None and r['bN'][0].size:
        h = np.histogram2d(*r['bN'], [BN_LOGN, BN_B])[0]
        x, y = 0.5 * (BN_LOGN[1:] + BN_LOGN[:-1]), 0.5 * (BN_B[1:] + BN_B[:-1])
        for level, alpha in zip(hpd_levels(h), (0.25, 0.45)):
            ax.contourf(x, y, h.T, levels=[level, h.max() + 1], colors=[COLOURS[sim]],
                        alpha=alpha)
    ax.plot(*observed[rung]['bN'], '.', color='k', ms=2.5)
    ax.set_xlim(BN_LOGN[0], BN_LOGN[-1])
    ax.set_ylim(BN_B[0], BN_B[-1])
    ax.grid(alpha=0.2, lw=0.5)


def stat_handles():
    h = [Line2D([], [], color=COLOURS[s], lw=1.5, label=f'{s} {SIMS[s]}') for s in SIMS]
    return h + [Line2D([], [], color='k', lw=1.8, label='observed, S/N-matched'),
                Line2D([], [], color='0.35', lw=1.0, ls='--', label='observed, all')]


def save(fig, path, handles=None, ncol=3):
    if handles:
        fig.legend(handles=handles, loc='upper center', ncol=ncol, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def grid(ncols, width):
    return plt.subplots(len(RUNGS), ncols, figsize=(width, 2.5 * len(RUNGS)), squeeze=False)


def plot_suite(suite, mocks, raw, observed, out, sub):
    last = len(RUNGS) - 1

    fig, axes = grid(len(PANEL), 4.0 * len(PANEL))
    for i, rung in enumerate(RUNGS):
        for j, name in enumerate(PANEL):
            draw_stat(axes[i, j], name, suite, rung, mocks, observed)
            title(axes[i, j], suite, rung, observed, mocks)
            style(axes[i, j], name, True, i == last)
    save(fig, out / f'panel_{suite}.png', stat_handles())

    bn_handles = [Patch(color=COLOURS[s], alpha=0.5, label=f'{s} {SIMS[s]} (68, 95%)')
                  for s in SIMS]
    bn_handles.append(Line2D([], [], ls='', marker='.', color='k',
                             label='observed lines, S/N-matched'))
    fig, axes = grid(len(SIMS), 3.2 * len(SIMS))
    for i, rung in enumerate(RUNGS):
        for j, sim in enumerate(SIMS):
            ax = axes[i, j]
            draw_bn(ax, suite, sim, rung, mocks, observed)
            title(ax, suite, rung, observed, mocks, sim)
            if j == 0:
                ax.set_ylabel(r'$b$ [km s$^{-1}$]')
            if i == last:
                ax.set_xlabel(r'$\log N_{\rm HI}$')
    save(fig, out / f'b_N_plane_{suite}.png', bn_handles, ncol=5)

    if raw:
        handles = [Line2D([], [], color='0.3', lw=1.5, label='raw'),
                   Line2D([], [], color='0.3', lw=1.5, ls='--', label=sub)]
        handles += stat_handles()[len(SIMS):]
        for name in PANEL:
            fig, axes = grid(len(SIMS), 3.2 * len(SIMS))
            for i, rung in enumerate(RUNGS):
                for j, sim in enumerate(SIMS):
                    draw_compare(axes[i, j], name, suite, sim, rung, mocks, raw, observed)
                    title(axes[i, j], suite, rung, observed, mocks, sim)
                    style(axes[i, j], name, j == 0, i == last)
            save(fig, out / f'{sub}_vs_raw' / f'{name}_{suite}.png', handles, ncol=4)

    single = out / 'single'
    for rung in RUNGS:
        for name in STATS:
            fig, ax = plt.subplots(figsize=(5.0, 3.6))
            draw_stat(ax, name, suite, rung, mocks, observed)
            title(ax, suite, rung, observed, mocks)
            style(ax, name)
            save(fig, single / name / f'{suite}_snr-{rung}.png', stat_handles(), ncol=2)
        for sim in SIMS:
            fig, ax = plt.subplots(figsize=(5.0, 3.6))
            draw_bn(ax, suite, sim, rung, mocks, observed)
            title(ax, suite, rung, observed, mocks, sim)
            ax.set_xlabel(r'$\log N_{\rm HI}$')
            ax.set_ylabel(r'$b$ [km s$^{-1}$]')
            save(fig, single / 'b_N' / f'{suite}_{sim}_snr-{rung}.png')


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

    # Lines below the b floor stay out of the b histograms but stay in the b-N plane.
    u = build_units(['a', 'b'], np.array([0, 0, 1]), np.zeros(3), np.array([20.0, 30.0, 40.0]),
                    np.array([12.5, 13.5, 13.5]), [np.zeros(FLUX_EDGES.size - 1)] * 2,
                    [1.0, 1.0], 13.2)
    assert u['b'].sum() == 2
    r = evaluate(u, np.array([0]), [np.array([0])])
    assert r['n'] == 1 and r['bN'][1].tolist() == [20.0, 30.0]

    # Every mock is central, and one-survey draws take exactly one mock per template.
    names = np.array(['p', 'q', 'p', 'q', 'r'])
    central, draws = mock_draws(names, rng, n=20)
    assert central.tolist() == [0, 1, 2, 3, 4]
    assert all(sorted(names[d]) == ['p', 'q', 'r'] for d in draws)

    # Rung bins meet at the geometric midpoints and cover every S/N once.
    s = np.array([5.0, 14.0, 14.2, 28.0, 30.0, 56.0, 57.0, 200.0])
    hits = np.array([snr_bin(r, s) for r in RUNGS[1:]])
    assert (hits.sum(axis=0) == 1).all() and hits[0].tolist()[:2] == [True, True]
    assert hits[3].tolist()[-2:] == [True, True]

    # Contour levels enclose at least the requested fractions, the inner one higher.
    h = np.array([[10.0, 5.0], [3.0, 2.0]])
    l95, l68 = hpd_levels(h)
    assert h[h >= l68].sum() >= 0.68 * h.sum() and h[h >= l95].sum() >= 0.95 * h.sum()
    assert l68 > l95

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
    p.add_argument('--sub', default='uvb_match',
                   help="mock subdirectory, '' for raw tau; anything else is also drawn "
                        "against raw")
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
    snr = np.array([t['snr_resel'] for t in templates])
    print(f'{len(templates)} observed sightlines, total dX = {sum(dx.values()):.2f} '
          f'at z <= {args.z_max}')

    obs_units = observed_units(templates, ranges, dx, args.b_logn_min)
    observed = {}
    for rung in RUNGS:
        keep = names if rung == 'data' else names[snr_bin(rung, snr)]
        if not keep.size:
            print(f'S/N {rung}: no observed sightline in the bin; using all of them')
            keep = names
        if rung != 'data':
            print(f'S/N {rung} per resel: {", ".join(keep)}')
        observed[rung] = evaluate(obs_units, *bootstrap_draws(obs_units['names'], keep, rng))
        observed[rung]['n'] = len(keep)

    mocks = load_mocks(args.mock_root, args.sub, ranges, dx, args.b_logn_min, rng)
    raw = load_mocks(args.mock_root, '', ranges, dx, args.b_logn_min, rng) if args.sub else {}
    for suite in SUITES:
        if any(k[0] == suite for k in mocks):
            plot_suite(suite, mocks, raw, observed, Path(args.out), args.sub)
        else:
            print(f'{suite}: no mocks, skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
