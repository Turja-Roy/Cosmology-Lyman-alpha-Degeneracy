"""
Does extreme feedback move an observable as much as cosmology does?

EX_0..EX_3 share cosmology (Omega_m = 0.3, sigma_8 = 0.8) and IC seed, and differ
only in feedback: EX_0 fiducial, EX_1 A_AGN1 = 100, EX_2 A_SN1 = 100, EX_3 none.

Per observable and snapshot, the fractional spread (max - min) / |fiducial| over
    S_fb   EX_0..EX_3
    S_p1   1P_p1_n2..1P_p1_2   (Omega_0)
    S_p2   1P_p2_n2..1P_p2_2   (sigma_8)
and R = S_fb / S_p1, S_fb / S_p2. R < 1: cosmology moves it more than feedback.

R compares two chosen parameter ranges, so it ranks observables; it is not an
error bar. Spreads are only taken within one set, because EX and 1P use different
IC seeds (for cosmic variance see cosmic_variance.py). A spread below the
fiducial's own 1-sigma error is flagged upper_limit.

EX_1 equals EX_0 at snaps 024 and 028: TNG's kinetic AGN mode is not active yet.

Run:
    python scripts/feedback_robustness.py --out-dir plots/ex_robustness
    python scripts/feedback_robustness.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hypothesis_test_p1 import load_snap_row, load_cosmo_table, _setup_style, _save
from hypothesis_test_p1 import build_scan_frame, FIDUCIAL
from degeneracy_test import (_obs_extractors, _obs_error_extractors,
                             CDDF_LOWN, CDDF_HIGHN)


EX_SIMS = ['EX_0', 'EX_1', 'EX_2', 'EX_3']
EX_FID = 0
EX_LABEL = {
    'EX_0': 'fiducial',
    'EX_1': r'extreme AGN ($A_{\rm AGN1}{=}100$)',
    'EX_2': r'extreme SN ($A_{\rm SN1}{=}100$)',
    'EX_3': 'no feedback',
}
EX_COLOR = {'EX_0': 'k', 'EX_1': 'C3', 'EX_2': 'C0', 'EX_3': 'C2'}

DEFAULT_SNAPS = ('snap-024 snap-028 snap-032 snap-038 snap-044 '
                 'snap-050 snap-060 snap-072 snap-080 snap-090').split()

COSMO_SCANS = ['p1', 'p2']
SCAN_LABEL = {'p1': r'$\Omega_0$ scan', 'p2': r'$\sigma_8$ scan'}


# =====================================================================
# Arithmetic
# =====================================================================

def frac_spread(vals, fid_idx=0):
    """(max - min) / |vals[fid_idx]|. NaN if any member is missing or the
    fiducial is zero."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return np.nan
    fid = v[fid_idx]
    if not np.isfinite(fid) or fid == 0:
        return np.nan
    return float((v.max() - v.min()) / abs(fid))


def ratio(num, den):
    """num / den, NaN instead of inf."""
    if not (np.isfinite(num) and np.isfinite(den)) or den == 0:
        return np.nan
    return float(num / den)


def n_distinct(vals):
    v = np.asarray(vals, dtype=float)
    return int(np.unique(v[np.isfinite(v)]).size)


def is_upper_limit(spread, fid_val, fid_err):
    """Spread smaller than the fiducial's fractional 1-sigma error."""
    if not np.isfinite(spread):
        return False
    if not (np.isfinite(fid_val) and np.isfinite(fid_err)) or fid_val == 0:
        return False
    return bool(spread < abs(fid_err / fid_val))


# =====================================================================
# Loading
# =====================================================================

def _observables(rows):
    return (
        {n: np.array([fn(r) for r in rows], float)
         for n, (fn, _l, _lg) in _obs_extractors().items()},
        {n: np.array([fn(r) for r in rows], float)
         for n, fn in _obs_error_extractors().items()},
    )


def ex_record(ex_root, snap):
    rows = [load_snap_row(Path(ex_root) / sim / snap) for sim in EX_SIMS]
    obs, err = _observables(rows)
    return {'snap': snap, 'rows': rows, 'z': rows[EX_FID]['redshift'],
            'obs': obs, 'obs_err': err}


def cosmo_record(p1_root, cosmo, scan, snap):
    """One 1P scan at one snapshot, in the same layout as ex_record."""
    rows = build_scan_frame(Path(p1_root), cosmo, scan, snap)
    obs, err = _observables(rows)
    fid_idx = next((i for i, r in enumerate(rows) if r['suffix'] == FIDUCIAL), None)
    return {'scan': scan, 'snap': snap, 'rows': rows, 'fid_idx': fid_idx,
            'z': rows[fid_idx]['redshift'] if fid_idx is not None else np.nan,
            'obs': obs, 'obs_err': err}


def build(ex_root, p1_root, cosmo_csv, snaps):
    cosmo = load_cosmo_table(cosmo_csv)
    ex = {s: ex_record(ex_root, s) for s in snaps}
    cos = {sc: {s: cosmo_record(p1_root, cosmo, sc, s) for s in snaps}
           for sc in COSMO_SCANS}
    return ex, cos


def find_missing(ex, cos, snaps):
    """Sim/snap dirs without tau_eff. A missing member would change max - min."""
    missing = []
    for s in snaps:
        for i, sim in enumerate(EX_SIMS):
            if not np.isfinite(ex[s]['obs']['tau_eff'][i]):
                missing.append(f'EX/{sim}/{s}')
        for sc in COSMO_SCANS:
            for r in cos[sc][s]['rows']:
                if not np.isfinite(r['tau_eff']):
                    missing.append(f"1P/{r['label']}/{s}")
    return missing


def robustness_table(ex, cos, snaps):
    out = {'snaps': list(snaps),
           'z': [float(ex[s]['z']) for s in snaps],
           'observables': {}}
    for name, (_fn, label, _log) in _obs_extractors().items():
        rec = {'label': label, 'S_fb': [], 'S_p1': [], 'S_p2': [],
               'R_p1': [], 'R_p2': [], 'n_distinct_ex': [], 'upper_limit': []}
        for s in snaps:
            v = ex[s]['obs'][name]
            e = ex[s]['obs_err'][name]
            s_fb = frac_spread(v, EX_FID)
            rec['S_fb'].append(s_fb)
            rec['n_distinct_ex'].append(n_distinct(v))
            rec['upper_limit'].append(is_upper_limit(s_fb, v[EX_FID], e[EX_FID]))
            for sc in COSMO_SCANS:
                c = cos[sc][s]
                s_c = frac_spread(c['obs'][name], c['fid_idx'])
                rec[f'S_{sc}'].append(s_c)
                rec[f'R_{sc}'].append(ratio(s_fb, s_c))
        out['observables'][name] = rec
    return out


# =====================================================================
# Figures
# =====================================================================

def _obs_style():
    return {n: (f'C{i}', 'os^vD<>p'[i % 8]) for i, n in enumerate(_obs_extractors())}


def _obs_grid(n):
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False, sharex=True)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('redshift')
    return fig, axes


def plot_robustness_ratio(tab, out_path):
    z = np.array(tab['z'])
    style = _obs_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, sc in zip(axes, COSMO_SCANS):
        for name, rec in tab['observables'].items():
            r = np.array(rec[f'R_{sc}'], float)
            ul = np.array(rec['upper_limit'], bool)
            col, mk = style[name]
            ax.plot(z, r, '-', color=col, lw=1.2, alpha=0.8, zorder=2)
            ax.plot(z[~ul], r[~ul], mk, color=col, ms=6, label=rec['label'], zorder=3)
            if ul.any():
                ax.plot(z[ul], r[ul], mk, mfc='none', color=col, ms=6, zorder=3)
        ax.axhline(1.0, color='k', ls='--', lw=1.2, zorder=1)
        ax.set_yscale('log')
        ax.set_xlabel('redshift')
        ax.set_title(f'feedback spread / {SCAN_LABEL[sc]} spread')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$R = S_{\rm fb}\,/\,S_{\rm cosmo}$')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle('Below the dashed line feedback moves the observable less than cosmology.\n'
                 'Open markers: spread below the 1$\\sigma$ error (upper limits).',
                 fontsize=9, y=1.06)
    _save(fig, out_path)


def plot_spreads(tab, out_path):
    z = np.array(tab['z'])
    names = list(tab['observables'])
    fig, axes = _obs_grid(len(names))
    for ax, name in zip(axes.ravel(), names):
        rec = tab['observables'][name]
        ax.plot(z, rec['S_fb'], 'ko-', ms=4, label=r'$S_{\rm fb}$ (EX)')
        ax.plot(z, rec['S_p1'], 'o-', color='C0', ms=4, label=r'$S_{p1}$ ($\Omega_0$)')
        ax.plot(z, rec['S_p2'], 's-', color='C3', ms=4, label=r'$S_{p2}$ ($\sigma_8$)')
        ax.set_yscale('log')
        ax.set_title(rec['label'], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('fractional spread')
    axes[0, 0].legend(fontsize=8, frameon=False)
    _save(fig, out_path)


def plot_ex_observables(ex, tab, snaps, out_path):
    """Each observable / EX_0 vs z, four sims overlaid."""
    names = list(tab['observables'])
    z = np.array(tab['z'])
    fig, axes = _obs_grid(len(names))
    for ax, name in zip(axes.ravel(), names):
        vals = np.array([ex[s]['obs'][name] for s in snaps], float)   # (n_snap, 4)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm = vals / vals[:, EX_FID][:, None]
        for i, sim in enumerate(EX_SIMS):
            ax.plot(z, norm[:, i], 'o-', ms=4, color=EX_COLOR[sim],
                    label=EX_LABEL[sim] if name == names[0] else None)
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.set_title(tab['observables'][name]['label'], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('value / EX_0')
    axes[0, 0].legend(fontsize=8, frameon=False)
    same = sorted({s for rec in tab['observables'].values()
                   for s, n in zip(snaps, rec['n_distinct_ex']) if n < len(EX_SIMS)})
    if same:
        fig.suptitle('EX_1 = EX_0 at ' + ', '.join(same) +
                     ' (kinetic AGN feedback not yet active)', fontsize=9, y=1.02)
    _save(fig, out_path)


def plot_cddf_grid(ex, snaps, out_path):
    ncol = 5
    nrow = int(np.ceil(len(snaps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for ax, snap in zip(axes.ravel(), snaps):
        rec = ex[snap]
        for sim, row in zip(EX_SIMS, rec['rows']):
            c = row['cddf']
            if c is None:
                continue
            m = c['f_N_HI'] > 0
            ax.plot(c['log10_N_HI'][m], c['f_N_HI'][m], '-', color=EX_COLOR[sim], lw=1.3,
                    label=EX_LABEL[sim] if snap == snaps[0] else None)
        ax.set_yscale('log')
        ax.set_title(f"{snap}  (z = {rec['z']:.2f})", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(snaps):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel(r'$\log_{10} N_{\rm HI}$')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$f(N_{\rm HI})$')
    axes[0, 0].legend(fontsize=8, frameon=False)
    _save(fig, out_path)


# =====================================================================

def self_test():
    assert ratio(frac_spread([1.0, 1.0, 1.2, 0.8]),
                 frac_spread([2.0, 2.0, 2.4, 1.6])) == 1.0

    # normalised by the fiducial member, not the mean
    assert abs(frac_spread([2.0, 3.0, 1.0], fid_idx=0) - 1.0) < 1e-12
    assert abs(frac_spread([2.0, 3.0, 1.0], fid_idx=1) - 2.0 / 3.0) < 1e-12

    assert np.isnan(frac_spread([1.0, np.nan, 1.5, 0.5]))
    assert np.isnan(frac_spread([0.0, 1.0, 2.0]))
    assert np.isnan(ratio(1.0, 0.0))
    assert np.isnan(ratio(np.nan, 1.0))

    assert n_distinct([1.0, 1.0, 2.0, 3.0]) == 3
    assert n_distinct([1.0, 2.0, 3.0, 4.0]) == 4

    assert is_upper_limit(0.01, 1.0, 0.05)
    assert not is_upper_limit(0.10, 1.0, 0.05)
    assert not is_upper_limit(np.nan, 1.0, 0.05)
    assert not is_upper_limit(0.01, 1.0, np.nan)

    # z = 0 tau_eff from the 2026-08-24 EX run
    assert abs(frac_spread([0.0275900, 0.0261500, 0.0299493, 0.0258625]) - 0.148) < 0.002

    # cddf_value does not extrapolate below the first bin centre (13.111)
    assert CDDF_HIGHN > CDDF_LOWN > 13.111
    assert _obs_extractors().keys() == _obs_error_extractors().keys()

    print('self-test OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--ex-root', type=Path, default=Path('output/analysis/IllustrisTNG/EX'))
    ap.add_argument('--p1-root', type=Path, default=Path('output/analysis/IllustrisTNG/1P'))
    ap.add_argument('--cosmo-csv', type=Path,
                    default=Path('data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv'))
    ap.add_argument('--snaps', default=','.join(DEFAULT_SNAPS))
    ap.add_argument('--out-dir', type=Path, default=Path('plots/ex_robustness'))
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    ex, cos = build(args.ex_root, args.p1_root, args.cosmo_csv, snaps)

    missing = find_missing(ex, cos, snaps)
    if missing:
        print(f'{len(missing)} sim/snap dirs missing:')
        for m in missing[:20]:
            print(f'  {m}')
        if len(missing) > 20:
            print(f'  ... and {len(missing) - 20} more')
        return 1
    print(f'complete: {len(EX_SIMS)} EX + {5 * len(COSMO_SCANS)} 1P x {len(snaps)} snapshots')

    tab = robustness_table(ex, cos, snaps)

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_robustness_ratio(tab, args.out_dir / 'robustness_ratio_vs_z.png')
    plot_spreads(tab, args.out_dir / 'spreads_vs_z.png')
    plot_ex_observables(ex, tab, snaps, args.out_dir / 'ex_observables_vs_z.png')
    plot_cddf_grid(ex, snaps, args.out_dir / 'ex_cddf_grid.png')
    jpath = args.out_dir / 'robustness.json'
    jpath.write_text(json.dumps(tab, indent=2, default=float))
    print(f'  saved {jpath}')

    # Largest R over snapshots, ignoring upper limits.
    print('\nLargest R over measured snapshots (lower = less affected by feedback):')
    rank = []
    for name, rec in tab['observables'].items():
        ul = np.array(rec['upper_limit'], bool)
        r = np.concatenate([np.array(rec['R_p1'], float)[~ul], np.array(rec['R_p2'], float)[~ul]])
        r = r[np.isfinite(r)]
        rank.append((r.max() if r.size else np.nan, int((~ul).sum()), name))
    for worst, n_meas, name in sorted(rank, key=lambda t: (np.isnan(t[0]), t[0])):
        if not np.isfinite(worst):
            print(f'  {name:12s} R_max =      n/a   (all snapshots are upper limits)')
        else:
            tag = 'robust' if worst < 1 else 'feedback-dominated'
            print(f'  {name:12s} R_max = {worst:8.3f}   {tag}   ({n_meas}/{len(snaps)} measured)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
