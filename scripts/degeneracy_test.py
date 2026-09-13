"""
Omega_0 vs sigma_8 degeneracy tests on the per-snapshot analysis CSVs.

Both parameters raise the clustering amplitude, so most forest statistics depend
on them only through S_8 = sigma_8 * sqrt(Omega_m / 0.3). Each test asks whether
an observable separates the p1 (Omega_0) and p2 (sigma_8) scans at fixed S_8.

Caveats:
- p1 spans S_8 = 0.46-1.03 and p2 only 0.60-1.00, so the scans are compared at
  matched S_8 (_matched_s8_gap), never by their raw spreads.
- The 25 Mpc/h box (k_min = 0.25 h/Mpc) has no modes near k_eq ~ 0.015 h/Mpc,
  so nothing here probes the linear power-spectrum shape.
- Whether a gap exceeds cosmic variance is checked in cosmic_variance.py.
- Doppler b is not used: line_width.cpp caps b at 80 km/s, where most z >= 3
  lines sit.

Run:
    python scripts/degeneracy_test.py \\
        --analysis-root output/analysis/IllustrisTNG/1P \\
        --ex-root       output/analysis/IllustrisTNG/EX \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
                    data/IllustrisTNG/EX/CosmoAstroSeed_IllustrisTNG_L25n256_EX.txt \\
        --scans p1,p2,ex --snaps snap-080,snap-044 --out-dir plots/degeneracy_test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hypothesis_test_p1 import (
    SCANS, FIDUCIAL, members,
    build_scan_frame, load_cosmo_table,
    dXdz,
    _setup_style, _save,
)

DEGEN_SCANS = ['p1', 'p2']
SCAN_COLOR  = {'p1': 'C0', 'p2': 'C3', 'ex': 'C2',
               'p7': 'C4', 'p8': 'C5', 'p9': 'C6'}
SCAN_MARKER = {'p1': 'o',  'p2': 's',  'ex': '^',
               'p7': 'v',  'p8': 'D',  'p9': 'P'}
S8_FID = 0.8

# The matched-S_8 gap is defined only between these two scans.
COSMO_PAIR = {'p1', 'p2'}


def varies_omega0(rec):
    om = np.asarray(rec['Omega0'], float)
    om = om[np.isfinite(om)]
    return om.size > 1 and not np.allclose(om, om[0])


def has_numeric_param(scan):
    """False for EX, whose x-axis is only the order of the sims."""
    return SCANS[scan].get('column') is not None


# =====================================================================
# Scalar observables per variant
# =====================================================================

def _trapz(y, x):
    fn = getattr(np, 'trapezoid', None) or np.trapz
    return fn(y, x)


def scale_split_ratio(ps, k_large_max=0.01, k_small_min=0.05):
    """Integrals of k*P_F(k) over k <= k_large_max and k >= k_small_min [s/km],
    and their ratio small/large. The ratio ignores an overall amplitude change."""
    if ps is None:
        return np.nan, np.nan, np.nan
    k = ps['k_s_per_km'].values
    kP = k * ps['P_k_mean_km_per_s'].values
    mL = (k > 0) & (k <= k_large_max)
    mS = k >= k_small_min
    L = _trapz(kP[mL], k[mL]) if mL.sum() > 1 else np.nan
    S = _trapz(kP[mS], k[mS]) if mS.sum() > 1 else np.nan
    ratio = S / L if (np.isfinite(L) and L != 0) else np.nan
    return L, S, ratio


CDDF_LOWN = 13.5
CDDF_HIGHN = 15.0


def cddf_value(cddf, logN_ref):
    """f(N_HI) at log N = logN_ref, interpolated in log-log. No extrapolation."""
    if cddf is None:
        return np.nan
    m = cddf['f_N_HI'] > 0
    x = cddf['log10_N_HI'][m].values
    y = np.log10(cddf['f_N_HI'][m].values)
    if x.size < 2 or not (x.min() <= logN_ref <= x.max()):
        return np.nan
    return 10.0 ** np.interp(logN_ref, x, y)


def cddf_slope(cddf, logN_lo=CDDF_LOWN, logN_hi=CDDF_HIGHN):
    """d log f / d log N between two column densities."""
    flo = cddf_value(cddf, logN_lo)
    fhi = cddf_value(cddf, logN_hi)
    if not (np.isfinite(flo) and np.isfinite(fhi) and flo > 0 and fhi > 0):
        return np.nan
    return (np.log10(fhi) - np.log10(flo)) / (logN_hi - logN_lo)


def cddf_value_err(cddf, logN_ref):
    """Poisson error on f(N_HI) from the nearest bin."""
    if cddf is None:
        return np.nan
    m = cddf['f_N_HI'] > 0
    if not m.any():
        return np.nan
    x = cddf['log10_N_HI'][m].values
    if 'f_N_HI_err' in cddf.columns:
        e = cddf['f_N_HI_err'][m].values
    elif 'counts' in cddf.columns:
        c = cddf['counts'][m].values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            e = np.where(c > 0, cddf['f_N_HI'][m].values / np.sqrt(c), np.nan)
    else:
        return np.nan
    if x.size == 0 or not (x.min() <= logN_ref <= x.max()):
        return np.nan
    return float(e[np.argmin(np.abs(x - logN_ref))])


def cddf_slope_err(cddf, logN_lo=CDDF_LOWN, logN_hi=CDDF_HIGHN):
    flo, fhi = cddf_value(cddf, logN_lo), cddf_value(cddf, logN_hi)
    elo, ehi = cddf_value_err(cddf, logN_lo), cddf_value_err(cddf, logN_hi)
    if not all(np.isfinite(v) and v > 0 for v in (flo, fhi)):
        return np.nan
    if not (np.isfinite(elo) and np.isfinite(ehi)):
        return np.nan
    # d(log10 f) = df / (f ln 10)
    s_lo = elo / flo / np.log(10.0)
    s_hi = ehi / fhi / np.log(10.0)
    return float(np.sqrt(s_lo ** 2 + s_hi ** 2) / (logN_hi - logN_lo))


# name -> (extractor(row), axis label, log y-axis)
def _obs_extractors():
    return {
        'tau_eff':    (lambda r: r['tau_eff'],                         r'$\tau_{\rm eff}$',                False),
        'mean_flux':  (lambda r: r['mean_flux'],                       r'$\langle F\rangle$',              False),
        'T0':         (lambda r: r['T0'],                              r'$T_0$ [K]',                       False),
        'power_ratio':(lambda r: scale_split_ratio(r['power_spectrum'])[2], r'$P_F$ small/large ratio',    False),
        'cddf_lowN':  (lambda r: cddf_value(r['cddf'], CDDF_LOWN),     rf'$f(N_{{\rm HI}}{{=}}10^{{{CDDF_LOWN}}})$',  True),
        'cddf_highN': (lambda r: cddf_value(r['cddf'], CDDF_HIGHN),    rf'$f(N_{{\rm HI}}{{=}}10^{{{CDDF_HIGHN}}})$', True),
        'cddf_slope': (lambda r: cddf_slope(r['cddf']),                rf'CDDF log-log slope ({CDDF_LOWN}$\to${CDDF_HIGHN})', False),
    }


# 1-sigma errors from within one box (sightline scatter, Poisson counts).
# power_ratio has none: it would need the k-mode covariance, which is not stored.
def _obs_error_extractors():
    return {
        'tau_eff':    lambda r: r.get('tau_eff_err', np.nan),
        'mean_flux':  lambda r: r.get('mean_flux_err', np.nan),
        'T0':         lambda r: r.get('T0_err', np.nan),
        'power_ratio': lambda r: np.nan,
        'cddf_lowN':  lambda r: cddf_value_err(r['cddf'], CDDF_LOWN),
        'cddf_highN': lambda r: cddf_value_err(r['cddf'], CDDF_HIGHN),
        'cddf_slope': lambda r: cddf_slope_err(r['cddf']),
    }


# =====================================================================
# One scan at one snapshot
# =====================================================================

def _S8(omega0, sigma8):
    return sigma8 * np.sqrt(omega0 / 0.3)


def scan_record(analysis_root, cosmo, scan, snap):
    """Rows of one scan plus Omega_0, sigma_8, S_8 and observable arrays."""
    rows = build_scan_frame(analysis_root, cosmo, scan, snap)
    for r in rows:
        lab = r['label']
        om = cosmo.loc[lab, 'Omega0'] if lab in cosmo.index else np.nan
        s8 = cosmo.loc[lab, 'sigma8'] if lab in cosmo.index else np.nan
        r['Omega0'], r['sigma8'], r['S8'] = om, s8, _S8(om, s8)

    rec = {
        'scan': scan, 'snap': snap, 'rows': rows,
        'Omega0': np.array([r['Omega0'] for r in rows], float),
        'sigma8': np.array([r['sigma8'] for r in rows], float),
        'S8':     np.array([r['S8']     for r in rows], float),
        'param':  np.array([r['param_value'] for r in rows], float),
        'obs':    {name: np.array([fn(r) for r in rows], float)
                   for name, (fn, _lbl, _lg) in _obs_extractors().items()},
        'obs_err': {name: np.array([fn(r) for r in rows], float)
                    for name, fn in _obs_error_extractors().items()},
    }
    rec['fid_idx'] = next((i for i, r in enumerate(rows)
                           if r['suffix'] == FIDUCIAL), None)
    rec['z'] = rows[rec['fid_idx']]['redshift'] if rec['fid_idx'] is not None else np.nan
    return rec


def _norm_to_fid(arr, fid_idx):
    if fid_idx is None or not np.isfinite(arr[fid_idx]) or arr[fid_idx] == 0:
        return np.full_like(arr, np.nan)
    return arr / arr[fid_idx]


def _norm_err_to_fid(arr, err, fid_idx):
    """Error on arr / arr[fid], including the fiducial's own error."""
    if fid_idx is None or err is None:
        return None
    with np.errstate(divide='ignore', invalid='ignore'):
        v_fid, e_fid = arr[fid_idx], err[fid_idx]
        if not np.isfinite(v_fid) or v_fid == 0:
            return None
        out = np.abs(arr / v_fid) * np.sqrt((err / arr) ** 2 + (e_fid / v_fid) ** 2)
    return out if np.any(np.isfinite(out)) else None


def _matched_s8_gap(curves):
    """RMS difference between the p1 and p2 curves on their common S_8 range.

    curves: scan -> (S_8, obs / obs_fid). NaN if either scan is missing or has
    fewer than two finite points.
    """
    if not COSMO_PAIR.issubset(curves):
        return np.nan
    x1, y1 = curves['p1']; x2, y2 = curves['p2']
    lo = max(np.nanmin(x1), np.nanmin(x2))
    hi = min(np.nanmax(x1), np.nanmax(x2))
    m1 = np.isfinite(x1) & np.isfinite(y1)
    m2 = np.isfinite(x2) & np.isfinite(y2)
    if m1.sum() < 2 or m2.sum() < 2 or not hi > lo:
        return np.nan
    xs = np.linspace(lo, hi, 25)
    g1 = np.interp(xs, x1[m1], y1[m1])
    g2 = np.interp(xs, x2[m2], y2[m2])
    return float(np.sqrt(np.mean((g2 - g1) ** 2)))


# =====================================================================
# Single-snapshot figures. `records` maps scan -> scan_record.
# =====================================================================

def s8_collapse_test(records, out_path, snap_label):
    """Each observable / fiducial vs S_8. Overlapping scans = degenerate."""
    extr = _obs_extractors()
    names = list(extr)
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows),
                             squeeze=False)
    flat = list(axes.ravel())

    gaps = {}
    for ax, name in zip(flat, names):
        _, lbl, logy = extr[name]
        curves = {}
        for scan, rec in records.items():
            y = _norm_to_fid(rec['obs'][name], rec['fid_idx'])
            yerr = _norm_err_to_fid(rec['obs'][name], rec['obs_err'][name], rec['fid_idx'])
            ax.errorbar(rec['S8'], y, yerr=yerr, fmt=SCAN_MARKER[scan] + '-',
                        color=SCAN_COLOR[scan], lw=1.8, ms=6, capsize=3,
                        label=f'{scan} ({SCANS[scan]["label"]})')
            curves[scan] = (rec['S8'], y)
        gap = _matched_s8_gap(curves)
        if np.isfinite(gap):
            gaps[name] = gap

        ax.axvline(S8_FID, color='gray', lw=0.8, ls=':')
        ax.axhline(1.0,    color='gray', lw=0.8, ls=':')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(r'$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$')
        ax.set_ylabel(lbl + ' / fid')
        ax.set_title(name + (f'  (gap={gap:.3f})' if np.isfinite(gap) else ''), fontsize=10)
        ax.grid(alpha=0.3, which='both')
    for ax in flat[len(names):]:
        ax.axis('off')
    flat[0].legend(fontsize=9, loc='best')
    fig.suptitle(f'Observables vs $S_8$: overlapping scans are degenerate ({snap_label})',
                 fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return gaps


def path_length_geometry_test(records, out_path, snap_label):
    """Left: dX/dz / fiducial per scan. Right: spread of the low-N CDDF amplitude
    before and after dividing out dX/dz. Skipped if no scan varies Omega_0."""
    if not any(varies_omega0(rec) for rec in records.values()):
        print('  [path-length] no selected scan varies Omega_0 -- skipping')
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for scan, rec in records.items():
        z, fid_idx = rec['z'], rec['fid_idx']
        if fid_idx is None or not np.isfinite(z):
            continue
        dX = np.array([dXdz(z, o) if np.isfinite(o) else np.nan for o in rec['Omega0']])
        axL.plot(rec['param'] / rec['param'][fid_idx], dX / dX[fid_idx],
                 SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan], lw=2, ms=7,
                 label=f'{scan} ({SCANS[scan]["label"]})')
    axL.axhline(1.0, color='gray', lw=0.8, ls=':')
    axL.axvline(1.0, color='gray', lw=0.8, ls=':')
    axL.set_xlabel('parameter / fiducial')
    axL.set_ylabel(r'$dX/dz \,/\, (dX/dz)_{\rm fid}$')
    axL.set_title(r'Absorption path length (depends on $\Omega_0$ only)')
    axL.grid(alpha=0.3); axL.legend()

    width = 0.35
    xpos = np.arange(len(records))
    for j, (scan, rec) in enumerate(records.items()):
        fid_idx, z = rec['fid_idx'], rec['z']
        raw = _norm_to_fid(rec['obs']['cddf_lowN'], fid_idx)
        dX = np.array([dXdz(z, o) if np.isfinite(o) and np.isfinite(z) else np.nan
                       for o in rec['Omega0']])
        corr = dX[fid_idx] / dX if fid_idx is not None else np.ones_like(dX)
        cor = _norm_to_fid(rec['obs']['cddf_lowN'] * corr, fid_idx)
        axR.bar(xpos[j] - width / 2, np.nanmax(raw) - np.nanmin(raw),
                width, color=SCAN_COLOR[scan], alpha=0.55,
                label='raw' if j == 0 else None)
        axR.bar(xpos[j] + width / 2, np.nanmax(cor) - np.nanmin(cor),
                width, color=SCAN_COLOR[scan], hatch='//', alpha=0.85,
                label='dX/dz divided out' if j == 0 else None)
    axR.set_xticks(xpos)
    axR.set_xticklabels([f'{s}\n({SCANS[s]["label"]})' for s in records])
    axR.set_ylabel(rf'spread of $f(10^{{{CDDF_LOWN}}})$ / fid across variants')
    axR.set_title(r'Path-length correction shrinks only the $\Omega_0$ spread')
    axR.grid(alpha=0.3, axis='y'); axR.legend()

    fig.suptitle(f'Absorption path length ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


def power_scale_split_test(records, out_path, snap_label):
    """Small/large-scale flux power ratio / fiducial vs parameter / fiducial.
    Returns the fitted slope per scan with a numeric parameter."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    slopes = {}
    for scan, rec in records.items():
        fid_idx = rec['fid_idx']
        y = _norm_to_fid(rec['obs']['power_ratio'], fid_idx)
        x = rec['param'] / rec['param'][fid_idx] if fid_idx is not None else rec['param']
        ax.plot(x, y, SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan],
                lw=2, ms=7, label=f'{scan} ({SCANS[scan]["label"]})')
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2 and has_numeric_param(scan):
            slopes[scan] = float(np.polyfit(x[m], y[m], 1)[0])
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.axvline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('parameter / fiducial' if all(has_numeric_param(s) for s in records)
                  else 'variant')
    ax.set_ylabel(r'(small/large $P_F$ ratio) / fid')
    sub = ', '.join(f'{s} slope={v:.2f}' for s, v in slopes.items()) or 'no numeric parameter'
    ax.set_title(f'Small/large-scale flux power ratio ({snap_label})\n{sub}', fontsize=11)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return slopes


def observable_pair_map(records, out_path, snap_label):
    """One observable against another along each scan. Tracks that are not
    collinear separate the parameters even where each observable alone cannot."""
    pairs = [('tau_eff', 'T0'), ('tau_eff', 'power_ratio')]
    extr = _obs_extractors()
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 5.5))
    for ax, (xn, yn) in zip(np.atleast_1d(axes), pairs):
        for scan, rec in records.items():
            xv = _norm_to_fid(rec['obs'][xn], rec['fid_idx'])
            yv = _norm_to_fid(rec['obs'][yn], rec['fid_idx'])
            ax.plot(xv, yv, SCAN_MARKER[scan] + '-', color=SCAN_COLOR[scan],
                    lw=2, ms=7, label=f'{scan} ({SCANS[scan]["label"]})')
            for i, r in enumerate(rec['rows']):
                if np.isfinite(xv[i]) and np.isfinite(yv[i]):
                    ax.annotate(r['suffix'], (xv[i], yv[i]), xytext=(4, 4),
                                textcoords='offset points', fontsize=7,
                                color=SCAN_COLOR[scan])
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.axvline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(extr[xn][1] + ' / fid')
        ax.set_ylabel(extr[yn][1] + ' / fid')
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(f'Observable pairs ({snap_label})', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


# =====================================================================
# Multi-snapshot figures. `records_by_snap` maps snap -> {scan -> record}.
# =====================================================================

def cddf_amplitude_figure(records_by_snap, out_path,
                          panels=(('cddf_lowN', 3.0), ('cddf_highN', 0.0))):
    """One CDDF amplitude per panel, each at the snapshot nearest its target z."""
    extr = _obs_extractors()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.4),
                             squeeze=False)
    gaps = {}
    for ax, (name, z_target) in zip(axes.ravel(), panels):
        _, lbl, logy = extr[name]
        snap = min(records_by_snap,
                   key=lambda s: abs(next(iter(records_by_snap[s].values()))['z'] - z_target))
        records = records_by_snap[snap]
        z = next(iter(records.values()))['z']
        curves = {}
        for scan, rec in records.items():
            y = _norm_to_fid(rec['obs'][name], rec['fid_idx'])
            yerr = _norm_err_to_fid(rec['obs'][name], rec['obs_err'][name], rec['fid_idx'])
            ax.errorbar(rec['S8'], y, yerr=yerr, fmt=SCAN_MARKER[scan] + '-',
                        color=SCAN_COLOR[scan], lw=1.8, ms=6, capsize=3,
                        label=f'{scan} ({SCANS[scan]["label"]})')
            curves[scan] = (rec['S8'], y)
        gap = _matched_s8_gap(curves)
        gaps[name] = {'snap': snap, 'z': z, 'gap': gap}

        ax.axvline(S8_FID, color='gray', lw=0.8, ls=':')
        ax.axhline(1.0,    color='gray', lw=0.8, ls=':')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel(r'$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$')
        ax.set_ylabel(lbl + ' / fid')
        ax.set_title(f'{name} at $z = {z:.2f}$' + (f'  (gap={gap:.3f})' if np.isfinite(gap) else ''),
                     fontsize=11)
        ax.grid(alpha=0.3, which='both')
    axes.ravel()[0].legend(fontsize=9, loc='best')
    fig.suptitle('CDDF amplitudes vs $S_8$', fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return gaps


def evolution_index_test(records_by_snap, out_path, obs_names=('tau_eff', 'mean_flux')):
    """Observable vs z for every variant, one panel per scan, and the index
    d ln(obs) / d ln(1+z) per variant. A power law, because tau_eff spans three
    decades and a linear fit would be set by the end points."""
    snaps = list(records_by_snap)
    if len(snaps) < 2:
        print('  [evolution index] needs >= 2 snapshots -- skipping')
        return {}
    scans = list(records_by_snap[snaps[0]])

    fig, axes = plt.subplots(len(obs_names), len(scans),
                             figsize=(6.0 * len(scans), 4.4 * len(obs_names)), squeeze=False)
    slopes = {}
    for j, scan in enumerate(scans):
        snaps_z = sorted(snaps, key=lambda s: np.nan_to_num(records_by_snap[s][scan]['z'], nan=np.inf))
        zarr = np.array([records_by_snap[s][scan]['z'] for s in snaps_z], float)
        mem = members(scan)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(mem)))
        for i, name in enumerate(obs_names):
            ax = axes[i][j]
            for vi, suf in enumerate(mem):
                yv = np.array([records_by_snap[s][scan]['obs'][name][vi] for s in snaps_z], float)
                ev = np.array([records_by_snap[s][scan]['obs_err'][name][vi] for s in snaps_z], float)
                pv = records_by_snap[snaps_z[0]][scan]['param'][vi]
                ax.errorbar(zarr, yv, yerr=ev if np.any(np.isfinite(ev)) else None,
                            fmt='o-', color=colors[vi], lw=1.6, ms=5, capsize=2,
                            label=f'{suf} ({pv:.2f})')
                m = np.isfinite(zarr) & np.isfinite(yv) & (yv > 0) & (zarr > -1)
                if m.sum() >= 2:
                    slopes.setdefault(scan, {}).setdefault(name, {})[suf] = float(
                        np.polyfit(np.log(1.0 + zarr[m]), np.log(yv[m]), 1)[0])
            ax.invert_xaxis()
            ax.set_xlabel('redshift z')
            if name == 'tau_eff':
                ax.set_ylim(-0.5, 3)
                ax.set_ylabel(r'$\tau_{\rm eff}$')
            else:
                ax.set_ylabel(r'$\langle F \rangle$' if name == 'mean_flux' else name)
            ax.set_title(f'{scan} ({SCANS[scan]["label"]})')
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(title='variant', fontsize=7)
    fig.suptitle(r'Redshift evolution; index = $d\ln({\rm obs})/d\ln(1+z)$', fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return slopes


# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', required=True, type=Path,
                    help='directory holding 1P_p1_0/, 1P_p2_0/, ...')
    ap.add_argument('--ex-root', type=Path, default=None,
                    help='directory holding EX_0/, ...; required if --scans includes ex')
    ap.add_argument('--cosmo-csv', required=True, type=Path, nargs='+',
                    help='CosmoAstroSeed table(s); pass the 1P and EX ones for --scans p1,p2,ex')
    ap.add_argument('--scans', default=','.join(DEGEN_SCANS),
                    help='comma-separated, any of ' + ','.join(SCANS))
    ap.add_argument('--snaps', default='snap-080,snap-044',
                    help='comma-separated snapshot dirs')
    ap.add_argument('--out-dir', type=Path, default=Path('plots/degeneracy_test'))
    args = ap.parse_args()

    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        ap.error(f'unknown scan(s) {unknown}; known: {list(SCANS)}')
    if 'ex' in scans and args.ex_root is None:
        ap.error("--scans includes 'ex' but --ex-root was not given")
    root_for = {s: (args.ex_root if s == 'ex' else args.analysis_root) for s in scans}

    _setup_style()
    cosmo = pd.concat([load_cosmo_table(c) for c in args.cosmo_csv])
    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records_by_snap = {snap: {scan: scan_record(root_for[scan], cosmo, scan, snap)
                              for scan in scans}
                       for snap in snaps}
    for snap, recs in records_by_snap.items():
        for scan, rec in recs.items():
            if rec['fid_idx'] is None or not np.isfinite(rec['z']):
                print(f'  WARNING: {scan} has no fiducial at {snap} under {root_for[scan]}')

    summary = {'snaps': snaps, 'per_snap': {}}
    for snap in snaps:
        print(f'\n=== {snap} ===')
        recs = records_by_snap[snap]
        d = args.out_dir / snap
        d.mkdir(parents=True, exist_ok=True)
        zs = [r['z'] for r in recs.values() if np.isfinite(r['z'])]
        label = f'$z = {zs[0]:.2f}$' if zs else snap
        gaps = s8_collapse_test(recs, d / 's8_collapse.png', label)
        path_length_geometry_test(recs, d / 'path_length_geometry.png', label)
        slopes = power_scale_split_test(recs, d / 'power_scale_split.png', label)
        observable_pair_map(recs, d / 'observable_pair_map.png', label)
        summary['per_snap'][snap] = {
            'z': {s: r['z'] for s, r in recs.items()},
            'matched_s8_gap': gaps,
            'power_ratio_slope': slopes,
        }
        with open(d / 'summary.json', 'w') as fh:
            json.dump(summary['per_snap'][snap], fh, indent=2, default=float)

    print('\n=== CDDF amplitudes ===')
    summary['cddf_amplitudes'] = cddf_amplitude_figure(
        records_by_snap, args.out_dir / 'cddf_amplitudes.png')

    print('\n=== evolution index ===')
    summary['evolution_index'] = evolution_index_test(
        records_by_snap, args.out_dir / 'evolution_index.png')

    with open(args.out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
