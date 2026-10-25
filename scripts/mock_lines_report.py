"""Compare the mock Voigt catalogues against the observed one, at matched column density.

The mock catalogues come from scripts/sim_mock_lines.py: simulated optical depth put through
the COS line spread function, resampled onto each observed sightline's pixel grid, given an
error array, and fitted by the same code that fits the real spectra. That makes the simulated
widths comparable to the collaborator's Voigt decompositions in a way the noiseless
0.1 km/s measurement never was.

The one thing that still has to be matched by hand is column density: the mock fits reach
weaker lines than her by-eye regions do, so the catalogues overlap only above a cut. Every
comparison here is therefore reported as a function of a log N floor.

Writes tables.tex fragments and three figures.

    python scripts/mock_lines_report.py --mock-root <cluster>/output/obs/mock_lines
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vpfit_catalog as vp

SIMS = {'EX_0': 'fiducial', 'EX_1': 'extreme AGN', 'EX_2': 'extreme SN',
        'EX_3': 'no feedback'}
SNR_LEVELS = ['80', '40', '20', 'data', '10']
LOGN_CUTS = [None, 13.0, 13.2, 13.5]
# Windows, not floors: the observed catalogue runs to log N = 19.3, and a handful of damped
# systems dominate any unbounded sum of N.
LOGN_WINDOWS = [(13.0, 14.0), (13.0, 14.5), (13.2, 15.0), (13.5, 15.0)]
B_NARROW = 30.0
MAX_B_ERR = 10.0
MAX_LOGN_ERR = 0.5
N_OBS_SIGHTLINES = 51     # 4c25 has no VPFIT fit
N_MOCKS = 255             # reset from the mock ids in main()
COLOURS = {'EX_0': '#1f77b4', 'EX_1': '#d62728', 'EX_2': '#2ca02c', 'EX_3': '#7f3f9f'}


def load_mock(path):
    """Mock rows under the same quality cuts applied to the observed catalogue.

    Without the cuts the mock catalogue carries components whose b error exceeds b, which the
    observed comparison sample has already had removed.
    """
    b, logn, mock, snap, tmpl = [], [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r['ion_key'] != 'HI':
                continue
            if float(r['b_err']) > MAX_B_ERR or float(r['logN_err']) > MAX_LOGN_ERR:
                continue
            b.append(float(r['b']))
            logn.append(float(r['logN']))
            mock.append(int(r['mock']))
            snap.append(r['snap'])
            tmpl.append(r['template'])
    return {'b': np.array(b), 'logN': np.array(logn), 'mock': np.array(mock),
            'snap': np.array(snap), 'template': np.array(tmpl)}


def load_observed(template_glob):
    paths = [str(Path(p).with_name(Path(p).stem.replace('_all', '') + '.all'))
             for p in sorted(glob.glob(template_glob))]
    rows = vp.hi_rows(vp.catalogue([p for p in paths if Path(p).exists()]),
                      max_b_err=MAX_B_ERR, max_logN_err=MAX_LOGN_ERR)
    return {'b': np.array([r['b'] for r in rows]),
            'logN': np.array([r['logN'] for r in rows]),
            'sightline': np.array([r['sightline'] for r in rows])}


def cut(cat, logn_min):
    if logn_min is None:
        return np.ones(cat['b'].size, dtype=bool)
    return cat['logN'] >= logn_min


def stats(cat, logn_min, n_units):
    m = cut(cat, logn_min)
    b = cat['b'][m]
    if b.size == 0:
        return {'n': 0, 'per': 0.0, 'median': np.nan, 'frac': np.nan}
    return {'n': int(b.size), 'per': b.size / n_units,
            'median': float(np.median(b)), 'frac': float((b < B_NARROW).mean())}


def summed_column(cat, lo, hi, n_units):
    """Components and total column density per sightline within a log N window.

    Summed N is insensitive to how a feature was divided into components, where the count is
    not, so the two together separate a decomposition difference from a difference in the
    amount of absorbing gas.
    """
    m = (cat['logN'] >= lo) & (cat['logN'] < hi)
    return {'per': int(m.sum()) / n_units, 'sumN': float((10.0 ** cat['logN'][m]).sum()) / n_units}


def cv_width_scatter(cv_root, snap='090'):
    """Seed-to-seed scatter in the narrow-line fraction across the CV set.

    The EX variants share initial conditions, so this does not limit a comparison between
    them. It is the error on an absolute simulation-to-observation ratio drawn from one box.
    """
    paths = sorted(glob.glob(str(Path(cv_root) / '*' / f'snap-{snap}' / 'line_widths.csv')))
    fr = []
    for p in paths:
        b = []
        with open(p) as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                b.append(float(row[1]))
        if b:
            fr.append(float((np.array(b) < B_NARROW).mean()))
    if len(fr) < 2:
        return None
    fr = np.array(fr)
    return {'n': fr.size, 'mean': float(fr.mean()), 'std': float(fr.std(ddof=1)),
            'lo': float(fr.min()), 'hi': float(fr.max())}


def frac_by_mock(cat, logn_min, n_mocks):
    """frac(b < 30) resolved per mock realisation, so it can be resampled."""
    m = cut(cat, logn_min)
    b, mk = cat['b'][m], cat['mock'][m]
    num = np.bincount(mk, weights=(b < B_NARROW).astype(float), minlength=n_mocks)
    den = np.bincount(mk, minlength=n_mocks)
    return num, den


def paired_bootstrap(mocks, ref, logn_min, n_mocks, draws=2000, seed=0):
    """Difference in frac(b<30) from the reference variant, resampling mock ids jointly.

    The variants share mock id -> template assignment, so resampling the same ids for all of
    them removes the template-to-template scatter the comparison does not care about. This
    covers mock-to-mock scatter only: it carries no box-to-box cosmic variance.
    """
    rng = np.random.default_rng(seed)
    nd = {k: frac_by_mock(v, logn_min, n_mocks) for k, v in mocks.items()}
    idx = rng.integers(0, n_mocks, size=(draws, n_mocks))
    out = {}
    for k, (num, den) in nd.items():
        f = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
        out[k] = f
    base = {k: num.sum() / max(den.sum(), 1) for k, (num, den) in nd.items()}
    res = {}
    for k in mocks:
        if k == ref:
            continue
        d = out[k] - out[ref]
        res[k] = (base[k] - base[ref], float(np.std(d)))
    return base, res


def snapshot_assignment(cat):
    """How many distinct observed sightlines were sent to each snapshot."""
    out = {}
    for s, t in zip(cat['snap'], cat['template']):
        out.setdefault(s, set()).add(t)
    return {k: len(v) for k, v in out.items()}


def self_test():
    cat = {'b': np.array([10.0, 20.0, 40.0, 50.0]),
           'logN': np.array([12.0, 13.5, 13.5, 14.0]),
           'mock': np.array([0, 0, 1, 1]),
           'snap': np.array(['086'] * 4),
           'template': np.array(['q1', 'q1', 'q2', 'q2'])}
    s = stats(cat, None, 2)
    assert s['n'] == 4 and s['per'] == 2.0 and s['frac'] == 0.5, s
    s = stats(cat, 13.5, 2)
    assert s['n'] == 3 and abs(s['median'] - 40.0) < 1e-9 and abs(s['frac'] - 1 / 3) < 1e-9, s

    num, den = frac_by_mock(cat, None, 2)
    assert list(den) == [2, 2] and list(num) == [2.0, 0.0], (num, den)

    # The window excludes both ends, and the sum is over N rather than over log N.
    win = {'b': np.zeros(4), 'logN': np.array([12.5, 13.6, 14.2, 16.0]),
           'mock': np.zeros(4, dtype=int), 'snap': np.array(['086'] * 4),
           'template': np.array(['q1'] * 4)}
    s = summed_column(win, 13.5, 15.0, 2)
    assert s['per'] == 1.0, s
    assert abs(s['sumN'] - (10 ** 13.6 + 10 ** 14.2) / 2) < 1.0, s

    # A variant compared against itself must give exactly zero difference and zero spread.
    base, res = paired_bootstrap({'a': cat, 'b': cat}, 'a', None, 2, draws=50)
    assert abs(res['b'][0]) < 1e-12 and res['b'][1] < 1e-12, res

    assert snapshot_assignment(cat) == {'086': 2}
    print('mock_lines_report self-test passed')


def figure_b_vs_logn(obs, mocks, cuts, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(cuts))
    labels = ['all' if c is None else r'$\geq %.1f$' % c for c in cuts]

    for ax, key, ylab in ((axes[0], 'median', r'median $b$ [km s$^{-1}$]'),
                          (axes[1], 'frac', r'fraction $b < 30$ km s$^{-1}$')):
        for sim, label in SIMS.items():
            v = [stats(mocks[sim], c, N_MOCKS)[key] for c in cuts]
            ax.plot(x, v, 'o-', color=COLOURS[sim], label=label, ms=5)
        v = [stats(obs, c, N_OBS_SIGHTLINES)[key] for c in cuts]
        ax.plot(x, v, 'k^--', label='observed', ms=7, lw=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r'$\log N$ floor')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title('simulated widths rise with column density, observed widths do not',
                      fontsize=9)
    axes[1].set_title('S/N matched to each sightline\'s own error array', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_width_cdf(obs, mocks, logn_min, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for sim, label in SIMS.items():
        b = np.sort(mocks[sim]['b'][cut(mocks[sim], logn_min)])
        ax.plot(b, np.arange(1, b.size + 1) / b.size, color=COLOURS[sim], label=label, lw=1.6)
    b = np.sort(obs['b'][cut(obs, logn_min)])
    ax.plot(b, np.arange(1, b.size + 1) / b.size, 'k--', label='observed', lw=2.2)
    ax.axvline(B_NARROW, color='k', ls=':', lw=1.0)
    ax.set_xlim(0, 100)
    ax.set_xlabel(r'$b$ [km s$^{-1}$]')
    ax.set_ylabel('cumulative fraction')
    ax.set_title(r'$\log N \geq %.1f$, S/N matched to the data' % logn_min, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_counts(obs, mocks, mocks_snr10, cuts, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(cuts))
    for sim, label in SIMS.items():
        v = [stats(mocks[sim], c, N_MOCKS)['per'] for c in cuts]
        ax.plot(x, v, 'o-', color=COLOURS[sim], label=label, ms=5)
    v = [stats(mocks_snr10['EX_0'], c, N_MOCKS)['per'] for c in cuts]
    ax.plot(x, v, 's:', color=COLOURS['EX_0'], label='fiducial, S/N 10', ms=5)
    v = [stats(obs, c, N_OBS_SIGHTLINES)['per'] for c in cuts]
    ax.plot(x, v, 'k^--', label='observed', ms=7, lw=2.0)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(['all' if c is None else r'$\geq %.1f$' % c for c in cuts])
    ax.set_xlabel(r'$\log N$ floor')
    ax.set_ylabel('components per sightline')
    ax.set_title('the count excess narrows with the cut but does not close', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_column(obs, mocks, windows, out_path, cv_std=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(windows))
    labels = ['%.1f--%.1f' % w for w in windows]

    if cv_std is not None:
        ax.axhspan(1 - cv_std, 1 + cv_std, color='0.85',
                   label='cosmic variance, one box')
    ax.axhline(1.0, color='k', lw=1.0)
    for sim, label in SIMS.items():
        o = [summed_column(obs, lo, hi, N_OBS_SIGHTLINES) for lo, hi in windows]
        s = [summed_column(mocks[sim], lo, hi, N_MOCKS) for lo, hi in windows]
        ax.plot(x, [a['sumN'] / b['sumN'] for a, b in zip(s, o)], 'o-',
                color=COLOURS[sim], label=label, ms=5)
        ax.plot(x, [a['per'] / b['per'] for a, b in zip(s, o)], 's--',
                color=COLOURS[sim], ms=4, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r'$\log N$ window')
    ax.set_ylabel('simulated / observed, per sightline')
    ax.set_title('solid: summed column density.   dashed: component count', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _rows(lines):
    return ''.join(l + ' \\\\\n' for l in lines)


def write_tables(path, obs, by_snr, cuts, disc, windows, cv):
    fid = by_snr['data']['EX_0']
    ladder = []
    for snr in SNR_LEVELS:
        s = stats(by_snr[snr]['EX_0'], None, N_MOCKS)
        name = 'own error array' if snr == 'data' else snr
        ladder.append('%s & %.1f & %.1f & %.3f' % (name, s['per'], s['median'], s['frac']))

    matched = []
    for c in cuts:
        o = stats(obs, c, N_OBS_SIGHTLINES)
        d = stats(fid, c, N_MOCKS)
        t = stats(by_snr['10']['EX_0'], c, N_MOCKS)
        matched.append('%s & %d & %.1f & %.3f & %.1f & %.3f & %.1f & %.3f'
                       % ('all' if c is None else '$\\geq %.1f$' % c, o['n'], o['median'],
                          o['frac'], d['median'], d['frac'], t['median'], t['frac']))

    disc_rows = []
    for sim in ('EX_3', 'EX_1', 'EX_2'):
        cells = []
        for c in (None, 13.2, 13.5):
            d, sd = disc[c][1][sim]
            ref = disc[c][0]['EX_0']
            cells.append('$%+.1f\\%%$ & $%.1f\\sigma$' % (d / ref * 100, abs(d) / sd))
        disc_rows.append('%s & %s' % (SIMS[sim], ' & '.join(cells)))

    counts = []
    for c in cuts:
        o = stats(obs, c, N_OBS_SIGHTLINES)['per']
        d = stats(fid, c, N_MOCKS)['per']
        t = stats(by_snr['10']['EX_0'], c, N_MOCKS)['per']
        counts.append('%s & %.1f & %.1f & %.1f & %.1f'
                      % ('all' if c is None else '$\\geq %.1f$' % c, o, d, t, t / o))

    column = []
    for lo, hi in windows:
        o = summed_column(obs, lo, hi, N_OBS_SIGHTLINES)
        d = summed_column(fid, lo, hi, N_MOCKS)
        t = summed_column(by_snr['10']['EX_0'], lo, hi, N_MOCKS)
        column.append('%.1f--%.1f & %.2f & %.2e & %.2f & %.2f & %.2f & %.2f'
                      % (lo, hi, o['per'], o['sumN'], d['per'] / o['per'],
                         d['sumN'] / o['sumN'], t['per'] / o['per'], t['sumN'] / o['sumN']))

    assign = snapshot_assignment(fid)
    with open(path, 'w') as fh:
        fh.write('%% Generated by scripts/mock_lines_report.py -- do not edit by hand.\n')
        for name, body in (('mockLadder', ladder), ('mockMatched', matched),
                           ('mockDiscrim', disc_rows), ('mockCounts', counts),
                           ('mockColumn', column)):
            fh.write('\\newcommand{\\%s}{%%\n%s}\n\n' % (name, _rows(body)))
        fh.write('\\newcommand{\\mockAssign}{%s}\n'
                 % ', '.join('%s ($\\times %d$)' % (k, v) for k, v in sorted(assign.items())))
        fh.write('\\newcommand{\\mockNmocks}{%d}\n' % N_MOCKS)
        if cv is not None:
            fh.write('\\newcommand{\\cvSeeds}{%d}\n' % cv['n'])
            fh.write('\\newcommand{\\cvMean}{%.4f}\n' % cv['mean'])
            fh.write('\\newcommand{\\cvScatter}{%.4f}\n' % cv['std'])
            fh.write('\\newcommand{\\cvScatterPct}{%.1f}\n' % (cv['std'] / cv['mean'] * 100))
            fh.write('\\newcommand{\\cvRangePct}{%.0f}\n'
                     % ((cv['hi'] - cv['lo']) / cv['mean'] * 100))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mock-root', default='output/obs/mock_lines')
    p.add_argument('--cv-root', default='output/analysis/IllustrisTNG/CV',
                   help='CV set analysis outputs, for the box-to-box scatter')
    p.add_argument('--cv-snap', default='090')
    p.add_argument('--templates', default='data/Obs/*_all.dat')
    p.add_argument('--out', default='plots/obs/mock_lines')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.mock_root)

    obs = load_observed(args.templates)
    by_snr = {}
    for snr in SNR_LEVELS:
        by_snr[snr] = {sim: load_mock(root / sim / f'mock_lines_snr-{snr}.csv')
                       for sim in SIMS}

    global N_MOCKS
    n_mocks = N_MOCKS = int(by_snr['data']['EX_0']['mock'].max()) + 1
    disc = {c: paired_bootstrap(by_snr['data'], 'EX_0', c, n_mocks, draws=args.bootstrap)
            for c in (None, 13.2, 13.5)}

    cv = cv_width_scatter(args.cv_root, args.cv_snap)
    if cv is None:
        print(f'no CV outputs under {args.cv_root}; skipping the box-to-box scatter')

    write_tables(out / 'tables.tex', obs, by_snr, LOGN_CUTS, disc, LOGN_WINDOWS, cv)
    figure_b_vs_logn(obs, by_snr['data'], LOGN_CUTS, out / 'mock_b_vs_logN.png')
    figure_width_cdf(obs, by_snr['data'], 13.2, out / 'mock_width_cdf.png')
    figure_counts(obs, by_snr['data'], by_snr['10'], LOGN_CUTS, out / 'mock_counts.png')
    figure_column(obs, by_snr['data'], LOGN_WINDOWS, out / 'mock_column.png',
                  cv_std=None if cv is None else cv['std'] / cv['mean'])

    o = stats(obs, 13.2, N_OBS_SIGHTLINES)
    d = stats(by_snr['data']['EX_0'], 13.2, n_mocks)
    print(f'observed {obs["b"].size} H I components over {N_OBS_SIGHTLINES} sightlines; '
          f'at logN >= 13.2 median b {o["median"]:.1f}, fiducial {d["median"]:.1f}')
    for c in (None, 13.2, 13.5):
        base, res = disc[c]
        label = 'all' if c is None else f'>= {c}'
        bits = ', '.join('%s %+.1f%% (%.1f sigma)'
                         % (SIMS[k], v[0] / base['EX_0'] * 100, abs(v[0]) / v[1])
                         for k, v in res.items())
        print(f'  logN {label}: {bits}')
    for lo, hi in LOGN_WINDOWS:
        a = summed_column(obs, lo, hi, N_OBS_SIGHTLINES)
        s = summed_column(by_snr['data']['EX_0'], lo, hi, n_mocks)
        print(f'  logN {lo}-{hi}: fiducial count x{s["per"] / a["per"]:.2f}, '
              f'summed N x{s["sumN"] / a["sumN"]:.2f}')
    if cv is not None:
        print(f'  CV scatter over {cv["n"]} seeds at snap {args.cv_snap}: '
              f'{cv["mean"]:.4f} +/- {cv["std"]:.4f} ({cv["std"] / cv["mean"] * 100:.1f}%)')
    print(f'wrote {out}/tables.tex and four figures')


if __name__ == '__main__':
    main()
