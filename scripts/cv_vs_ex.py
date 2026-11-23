"""
Cosmic variance against the spread between the EX feedback models, from the noiseless
analysis CSVs (no COS forward model; scripts/ex_snr_plots.py does the mock side).

Per suite and snapshot where EX_0..EX_3 and at least two CV boxes are analysed, for the
degeneracy_test observables plus the median b of lines with N_HI >= 10^13.2:
    sigma_cv      std over the CV boxes (IllustrisTNG: the 24 on the EX code version)
    sigma_cv_all  std over every CV box analysed
    ex0_offset    (EX_0 - CV mean) / sigma_cv: is the EX seed a typical box?
    dEX_k         |EX_k - EX_0| / sigma_cv
    range         (max - min over EX) / sigma_cv; above 1 the feedback spread is larger
                  than the box-to-box scatter (range_all with sigma_cv_all)
sigma_cv is measured at fiducial physics and applied to every model.

Output, in --out:
    cv_vs_ex_<suite>.csv
    observables_vs_z_<suite>_{1,2}sigma.png   value / EX_0 against z, each model with a
                                              +-1 (2) sigma_cv band
    range_vs_z_<suite>.png                    range against z, per observable

    python scripts/cv_vs_ex.py                                # after cv_lowz.sbatch
    python scripts/cv_vs_ex.py --out plots/cv_vs_ex_preview   # TNG CSVs already there
    python scripts/cv_vs_ex.py --self-test
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hypothesis_test_p1 import load_snap_row, _setup_style, _save   # noqa: E402
from degeneracy_test import _obs_extractors                        # noqa: E402
from feedback_robustness import _obs_grid                          # noqa: E402
from cosmic_variance import CV_SIMS                                # noqa: E402
from ex_snr_plots import COLOURS, CV_SKIP, SUITES                  # noqa: E402
from mock_lines_report import SIMS                                 # noqa: E402

B_LOGN_MIN = 13.2


def median_b(row):
    lw = row['line_widths']
    if lw is None:
        return np.nan
    b = lw.loc[lw['N_HI'] >= 10 ** B_LOGN_MIN, 'b_param_km_s']
    return float(b.median()) if len(b) else np.nan


OBS = {n: (fn, label) for n, (fn, label, _lg) in _obs_extractors().items()}
OBS['median_b'] = (median_b, rf'median $b$ ($N_{{\rm HI}} \geq 10^{{{B_LOGN_MIN}}}$)')


def values(rows):
    return {n: np.array([fn(r) for r in rows], float) for n, (fn, _) in OBS.items()}


def _std(v):
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=1)) if v.size >= 2 else np.nan


def compare(ex, cv, keep):
    """ex: one observable in EX_0..EX_3; cv: in each CV box; keep: boxes in sigma_cv."""
    s, s_all = _std(cv[keep]), _std(cv)
    mean = float(np.nanmean(cv[keep])) if np.isfinite(cv[keep]).any() else np.nan
    spread = np.nanmax(ex) - np.nanmin(ex) if np.isfinite(ex).any() else np.nan
    out = {sim: float(v) for sim, v in zip(SIMS, ex)}
    with np.errstate(divide='ignore', invalid='ignore'):
        out.update({'cv_mean': mean, 'sigma_cv': s, 'sigma_cv_all': s_all,
                    'ex0_offset': (ex[0] - mean) / s})
        out.update({f'dEX_{k}': abs(ex[k] - ex[0]) / s for k in range(1, len(ex))})
        out.update({'range': spread / s, 'range_all': spread / s_all})
    return out


def snapshots(root, suite):
    """(snap, CV boxes analysed there) for each snapshot with all four EX models analysed."""
    def done(d):
        return (d / 'flux_stats.csv').exists()

    ex_root, cv_root = root / suite / 'EX', root / suite / 'CV'
    out = []
    for d in sorted((ex_root / 'EX_0').glob('snap-*')):
        if not all(done(ex_root / sim / d.name) for sim in SIMS):
            continue
        boxes = [b for b in CV_SIMS if done(cv_root / b / d.name)]
        if len(boxes) >= 2:
            out.append((d.name, boxes))
    return out


def record(root, suite, snap, boxes):
    ex_rows = [load_snap_row(root / suite / 'EX' / sim / snap) for sim in SIMS]
    ex = values(ex_rows)
    cv = values([load_snap_row(root / suite / 'CV' / b / snap) for b in boxes])
    keep = np.array([b not in CV_SKIP.get(suite, ()) for b in boxes])
    return {'snap': snap, 'z': ex_rows[0]['redshift'], 'n_cv': int(keep.sum()),
            'n_cv_all': len(boxes), 'obs': {n: compare(ex[n], cv[n], keep) for n in OBS}}


def write_table(path, suite, recs):
    rows = [{'suite': suite, 'snap': r['snap'], 'z': round(float(r['z']), 4),
             'n_cv': r['n_cv'], 'n_cv_all': r['n_cv_all'], 'observable': n, **c}
            for r in recs for n, c in r['obs'].items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'  saved {path}')


def plot_observables(suite, recs, n_sigma, path):
    z = np.array([r['z'] for r in recs])
    fig, axes = _obs_grid(len(OBS))
    for ax, (name, (_, label)) in zip(axes.ravel(), OBS.items()):
        ex = np.array([[r['obs'][name][sim] for sim in SIMS] for r in recs])
        s = np.array([r['obs'][name]['sigma_cv'] for r in recs])
        with np.errstate(divide='ignore', invalid='ignore'):
            ref = ex[:, 0]
            for k, sim in enumerate(SIMS):
                ax.fill_between(z, (ex[:, k] - n_sigma * s) / ref, (ex[:, k] + n_sigma * s) / ref,
                                color=COLOURS[sim], alpha=0.18, lw=0)
                ax.plot(z, ex[:, k] / ref, 'o-', ms=3.5, lw=1.3, color=COLOURS[sim],
                        label=f'{sim} {SIMS[sim]}')
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('value / EX_0')
    axes[0, 0].legend(fontsize=8, frameon=False)
    n = sorted({r['n_cv'] for r in recs})
    fig.suptitle(f'{suite}: shaded $\\pm{n_sigma}\\sigma_{{\\rm CV}}$ over '
                 f'{"/".join(map(str, n))} CV boxes, applied to every model', fontsize=10)
    _save(fig, path)


def plot_range(suite, recs, path):
    z = np.array([r['z'] for r in recs])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, (_, label)) in enumerate(OBS.items()):
        ax.plot(z, [r['obs'][name]['range'] for r in recs], '-', marker='os^vD<>p'[i % 8],
                ms=5, color=f'C{i}', label=label)
    for y in (1.0, 2.0):
        ax.axhline(y, color='k', ls='--' if y == 1.0 else ':', lw=1.0)
    ax.set_yscale('log')
    ax.set_xlabel('redshift')
    ax.set_ylabel(r'(max $-$ min over EX) / $\sigma_{\rm CV}$')
    ax.set_title(f'{suite}: above the dashed line feedback beats cosmic variance', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    _save(fig, path)


def print_range(suite, recs):
    print(f'{suite}: (max - min over EX) / sigma_cv')
    print('  ' + ' ' * 12 + ''.join(f'{r["z"]:7.2f}' for r in recs))
    for name in OBS:
        print(f'  {name:12s}' + ''.join(f'{r["obs"][name]["range"]:7.2f}' for r in recs))


def self_test():
    import pandas as pd

    rng = np.random.default_rng(1)
    cv = np.r_[rng.normal(10.0, 1.0, 4000), 50.0, 60.0]
    keep = np.r_[np.ones(4000, bool), False, False]
    c = compare(np.array([10.0, 12.0, 9.0, 13.0]), cv, keep)
    assert abs(c['sigma_cv'] - 1.0) < 0.05 and c['sigma_cv_all'] > 1.3
    assert abs(c['range'] - 4.0) < 0.2 and abs(c['dEX_2'] - 1.0) < 0.05
    assert abs(c['ex0_offset']) < 0.1
    assert np.isnan(compare(np.ones(4), np.array([1.0, np.nan]), np.ones(2, bool))['range'])

    lw = pd.DataFrame({'N_HI': [1e13, 2e13, 3e13, 1e14], 'b_param_km_s': [5.0, 20.0, 30.0, 40.0]})
    assert median_b({'line_widths': lw}) == 30.0
    assert np.isnan(median_b({'line_widths': None}))
    print('cv_vs_ex self-test passed')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('output/analysis'))
    p.add_argument('--out', type=Path, default=Path('plots/cv_vs_ex'))
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return 0

    _setup_style()
    for suite in SUITES:
        snaps = snapshots(args.root, suite)
        if not snaps:
            print(f'{suite}: no snapshot with all four EX models and two CV boxes, skipped')
            continue
        recs = sorted((record(args.root, suite, s, b) for s, b in snaps), key=lambda r: r['z'])
        print(f'{suite}: ' + ', '.join(f'{r["snap"]} ({r["n_cv"]}/{r["n_cv_all"]} boxes)'
                                       for r in recs))
        write_table(args.out / f'cv_vs_ex_{suite}.csv', suite, recs)
        for n_sigma in (1, 2):
            plot_observables(suite, recs, n_sigma,
                             args.out / f'observables_vs_z_{suite}_{n_sigma}sigma.png')
        plot_range(suite, recs, args.out / f'range_vs_z_{suite}.png')
        print_range(suite, recs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
