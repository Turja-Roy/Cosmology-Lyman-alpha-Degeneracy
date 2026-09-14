"""
Cosmic variance from the CAMELS CV set, and the matched-S_8 gaps in units of it.

CV_0..CV_26 share the fiducial cosmology and astrophysics and differ only in IC
seed, so the scatter across them is the cosmic variance of a 25 Mpc/h box.

Per observable and snapshot:
    sigma                std / |mean| over the 27 boxes
    pair_sigma           sqrt(2) * sigma, expected difference between two boxes
    significance         matched-S_8 gap (degeneracy_test) / pair_sigma
    sigma_robust         1.4826 MAD / |median|, insensitive to a few extreme boxes
    significance_robust  the same gap / (sqrt(2) * sigma_robust)
    outliers             boxes more than 5 robust sigmas from the median
    ex0_p1_offset        |EX_0 / 1P_p1_0 - 1|, the earlier two-box estimate
The same is done for the redshift-evolution index of tau_eff and <F>.

At z=4, TNG's AGN radiation field lowers the neutral fraction across much of CV_10 and CV_18
(scripts/check_snapshot.py); the robust sigma shows the scatter without them.

p1 and p2 share one seed, so their gap is not itself limited by cosmic variance;
the significance says whether the gap would hold in a different realisation.

Run:
    python scripts/cosmic_variance.py --out-dir plots/cosmic_variance
    python scripts/cosmic_variance.py --self-test
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
from degeneracy_test import _obs_extractors, scan_record, _norm_to_fid, _matched_s8_gap
from feedback_robustness import ratio, _obs_style, DEFAULT_SNAPS, COSMO_SCANS

# CV_2, CV_8, CV_17 ran Arepo commit fc131931 (2021); the other CV runs, 1P and EX ran 4ab97a2c (2016).
CV_SIMS = [f'CV_{i}' for i in range(27)]
EVO_OBS = ('tau_eff', 'mean_flux')


def frac_sigma(vals):
    """std / |mean|. NaN if any box is missing or the mean is zero."""
    v = np.asarray(vals, float)
    if v.size < 2 or not np.all(np.isfinite(v)) or v.mean() == 0:
        return np.nan
    return float(v.std(ddof=1) / abs(v.mean()))


def frac_sigma_robust(vals):
    """1.4826 MAD / |median|, equal to std / |mean| for Gaussian scatter. NaN as frac_sigma."""
    v = np.asarray(vals, float)
    if v.size < 2 or not np.all(np.isfinite(v)) or np.median(v) == 0:
        return np.nan
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)) / abs(med))


def outliers(vals, n_sigma=5.0):
    """CV boxes further than n_sigma robust sigmas from the median."""
    v = np.asarray(vals, float)
    s = frac_sigma_robust(v) * abs(np.median(v))
    if not np.isfinite(s) or s == 0:
        return []
    return [CV_SIMS[i] for i in np.flatnonzero(np.abs(v - np.median(v)) > n_sigma * s)]


def evo_index(z, y):
    """d ln(y) / d ln(1+z), fitted as in degeneracy_test.evolution_index_test."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    m = np.isfinite(z) & np.isfinite(y) & (y > 0) & (z > -1)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log1p(z[m]), np.log(y[m]), 1)[0])


def frac_offset(a, b):
    r = ratio(a, b)
    return abs(r - 1.0) if np.isfinite(r) else np.nan


def summarise(vals, gap, offset):
    """vals: one observable's value in each CV box."""
    sigma, sigma_r = frac_sigma(vals), frac_sigma_robust(vals)
    pair = np.sqrt(2.0) * sigma
    return {'sigma': sigma, 'pair_sigma': pair,
            'matched_s8_gap': gap, 'significance': ratio(gap, pair),
            'sigma_robust': sigma_r,
            'significance_robust': ratio(gap, np.sqrt(2.0) * sigma_r),
            'outliers': outliers(vals),
            'ex0_p1_offset': offset, 'ex0_p1_offset_in_pair_sigma': ratio(offset, pair)}


# =====================================================================
# Loading
# =====================================================================

def load_rows(root, sims, snap):
    rows = [load_snap_row(Path(root) / sim / snap) for sim in sims]
    return {'z': rows[0]['redshift'],
            'obs': {n: np.array([fn(r) for r in rows], float)
                    for n, (fn, _l, _lg) in _obs_extractors().items()}}


def find_missing(cv, cos, snaps):
    missing = []
    for s in snaps:
        missing += [f'CV/{sim}/{s}' for sim, t in zip(CV_SIMS, cv[s]['obs']['tau_eff'])
                    if not np.isfinite(t)]
        missing += [f"1P/{r['label']}/{s}" for sc in COSMO_SCANS for r in cos[sc][s]['rows']
                    if not np.isfinite(r['tau_eff'])]
    return missing


# =====================================================================
# Table
# =====================================================================

def matched_gap(cos, snap, name):
    return _matched_s8_gap({
        sc: (cos[sc][snap]['S8'], _norm_to_fid(cos[sc][snap]['obs'][name], cos[sc][snap]['fid_idx']))
        for sc in COSMO_SCANS})


def build_table(cv, cos, ex0, snaps):
    """cv, ex0: snap -> load_rows(); cos: scan -> snap -> scan_record()."""
    tab = {'snaps': list(snaps), 'z': [float(cv[s]['z']) for s in snaps],
           'observables': {}, 'evolution_index': {}}

    for name in _obs_extractors():
        per_snap = []
        for s in snaps:
            p1 = cos['p1'][s]
            p1_fid = p1['obs'][name][p1['fid_idx']] if p1['fid_idx'] is not None else np.nan
            per_snap.append(summarise(cv[s]['obs'][name],
                                      matched_gap(cos, s, name),
                                      frac_offset(ex0[s]['obs'][name][0], p1_fid)))
        tab['observables'][name] = per_snap

    for name in EVO_OBS:
        cv_z = [cv[s]['z'] for s in snaps]
        cv_idx = [evo_index(cv_z, [cv[s]['obs'][name][i] for s in snaps])
                  for i in range(len(CV_SIMS))]

        curves, p1_fid_idx = {}, np.nan
        for sc in COSMO_SCANS:
            recs = [cos[sc][s] for s in snaps]
            z = [r['z'] for r in recs]
            idx = np.array([evo_index(z, [r['obs'][name][j] for r in recs])
                            for j in range(len(recs[0]['rows']))])
            fid = recs[0]['fid_idx']
            curves[sc] = (recs[0]['S8'], _norm_to_fid(idx, fid))
            if sc == 'p1' and fid is not None:
                p1_fid_idx = idx[fid]
        ex0_idx = evo_index([ex0[s]['z'] for s in snaps], [ex0[s]['obs'][name][0] for s in snaps])

        tab['evolution_index'][name] = summarise(cv_idx, _matched_s8_gap(curves),
                                                 frac_offset(ex0_idx, p1_fid_idx))
    return tab


# =====================================================================
# Figures
# =====================================================================

def plot_sigma(tab, out_path):
    """Robust and standard pair sigma per observable vs z, with the earlier EX_0 / 1P_p1_0 offset."""
    z = np.array(tab['z'])
    labels = {n: lbl for n, (_f, lbl, _lg) in _obs_extractors().items()}
    names = list(tab['observables'])
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False, sharex=True)
    for ax, name in zip(axes.ravel(), names):
        rec = tab['observables'][name]
        ax.plot(z, [d['pair_sigma'] for d in rec], 'ko-', ms=4,
                label=r'$\sqrt{2}\,\sigma$ (27 CV boxes)')
        ax.plot(z, [np.sqrt(2.0) * d['sigma_robust'] for d in rec], 'o--', color='0.6', ms=3,
                label=r'$\sqrt{2}\,\sigma$, robust')
        ax.plot(z, [d['ex0_p1_offset'] for d in rec], 'x', color='C3', ms=7,
                label='|EX_0 / 1P_p1_0 - 1|')
        ax.set_yscale('log')
        ax.set_title(labels[name], fontsize=10)
        ax.grid(alpha=0.3, which='both')
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('redshift')
    for ax in axes[:, 0]:
        ax.set_ylabel('fractional difference')
    axes[0, 0].legend(fontsize=7, frameon=False)
    _save(fig, out_path)


def plot_significance(tab, out_path):
    z = np.array(tab['z'])
    labels = {n: lbl for n, (_f, lbl, _lg) in _obs_extractors().items()}
    style = _obs_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, rec in tab['observables'].items():
        col, mk = style[name]
        ax.plot(z, [d['significance'] for d in rec], '-' + mk, color=col, ms=6,
                label=labels[name])
    ax.axhline(1, color='k', ls='--', lw=1.1)
    ax.axhline(3, color='k', ls=':', lw=1.1)
    ax.set_yscale('log')
    ax.set_xlabel('redshift')
    ax.set_ylabel(r'matched-$S_8$ gap / $\sqrt{2}\,\sigma$')
    evo = ', '.join(f"{n} index: {d['significance']:.2f}"
                    for n, d in tab['evolution_index'].items())
    ax.set_title(rf'$\Omega_0$ vs $\sigma_8$ gap in units of cosmic variance' + f'\n({evo})',
                 fontsize=10)
    ax.grid(alpha=0.3, which='both')
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save(fig, out_path)


# =====================================================================

def self_test():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 0.1, 20000)
    assert abs(frac_sigma(x) - 0.05) < 0.002
    assert abs(frac_sigma_robust(x) - 0.05) < 0.003
    assert np.isnan(frac_sigma([1.0, np.nan, 1.2]))
    assert np.isnan(frac_sigma([-1.0, 1.0]))

    v = np.r_[np.linspace(0.97, 1.03, 25), 2.0, 1.5]      # two extreme boxes, like z=4
    assert frac_sigma(v) > 0.1 and frac_sigma_robust(v) < 0.04
    assert outliers(v) == ['CV_25', 'CV_26']

    z = np.array([3.0, 2.0, 1.0, 0.0])
    assert abs(evo_index(z, (1 + z) ** 2.3) - 2.3) < 1e-9
    assert abs(evo_index(z, np.r_[(1 + z[:3]) ** -0.8, 0.0]) + 0.8) < 1e-9   # y <= 0 dropped

    assert abs(frac_offset(1.1, 1.0) - 0.1) < 1e-12
    assert np.isnan(frac_offset(1.0, 0.0))
    d = summarise(v, 0.3, 0.1)
    assert abs(d['significance'] - 0.3 / (np.sqrt(2) * d['sigma'])) < 1e-12
    assert abs(d['significance_robust'] - 0.3 / (np.sqrt(2) * d['sigma_robust'])) < 1e-12

    cv = {'snap-080': {'obs': {'tau_eff': np.r_[np.ones(26), np.nan]}}}
    cos = {sc: {'snap-080': {'rows': []}} for sc in COSMO_SCANS}
    assert find_missing(cv, cos, ['snap-080']) == ['CV/CV_26/snap-080']

    print('self-test OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--cv-root', type=Path, default=Path('output/analysis/IllustrisTNG/CV'))
    ap.add_argument('--p1-root', type=Path, default=Path('output/analysis/IllustrisTNG/1P'))
    ap.add_argument('--ex-root', type=Path, default=Path('output/analysis/IllustrisTNG/EX'))
    ap.add_argument('--cosmo-csv', type=Path,
                    default=Path('data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv'))
    ap.add_argument('--snaps', default=','.join(DEFAULT_SNAPS))
    ap.add_argument('--out-dir', type=Path, default=Path('plots/cosmic_variance'))
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    cosmo = load_cosmo_table(args.cosmo_csv)
    cv = {s: load_rows(args.cv_root, CV_SIMS, s) for s in snaps}
    ex0 = {s: load_rows(args.ex_root, ['EX_0'], s) for s in snaps}
    cos = {sc: {s: scan_record(args.p1_root, cosmo, sc, s) for s in snaps}
           for sc in COSMO_SCANS}

    missing = find_missing(cv, cos, snaps)
    if missing:
        print(f'{len(missing)} sim/snap dirs missing:')
        for m in missing[:20]:
            print(f'  {m}')
        if len(missing) > 20:
            print(f'  ... and {len(missing) - 20} more')
        return 1
    print(f'complete: {len(CV_SIMS)} CV + {5 * len(COSMO_SCANS)} 1P x {len(snaps)} snapshots')
    if not all(np.isfinite(ex0[s]['obs']['tau_eff'][0]) for s in snaps):
        print('  WARNING: EX_0 incomplete; ex0_p1_offset is NaN there')

    tab = build_table(cv, cos, ex0, snaps)

    _setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_sigma(tab, args.out_dir / 'sigma_vs_z.png')
    plot_significance(tab, args.out_dir / 'significance_vs_z.png')
    jpath = args.out_dir / 'cosmic_variance.json'
    jpath.write_text(json.dumps(tab, indent=2, default=float))
    print(f'  saved {jpath}')

    print('\nLargest gap / pair_sigma over snapshots:')
    for name, rec in tab['observables'].items():
        sig = np.array([d['significance'] for d in rec], float)
        if not np.any(np.isfinite(sig)):
            print(f'  {name:12s} n/a')
            continue
        i = int(np.nanargmax(sig))
        old = np.nanmedian([d['ex0_p1_offset_in_pair_sigma'] for d in rec])
        print(f"  {name:12s} {sig[i]:6.2f} at z = {tab['z'][i]:.2f}   "
              f"(EX_0 vs 1P_p1_0: median {old:.2f} pair_sigma)")
    for name, d in tab['evolution_index'].items():
        print(f"  {name + ' index':12s} {d['significance']:6.2f}   "
              f"(pair_sigma {d['pair_sigma']:.4f}, EX_0 vs 1P_p1_0: {d['ex0_p1_offset_in_pair_sigma']:.2f})")

    print('\nOutlier boxes (> 5 robust sigmas, any observable):')
    for i, s in enumerate(tab['snaps']):
        boxes = sorted({b for rec in tab['observables'].values() for b in rec[i]['outliers']})
        if boxes:
            print(f"  {s} (z = {tab['z'][i]:.2f}): {', '.join(boxes)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
