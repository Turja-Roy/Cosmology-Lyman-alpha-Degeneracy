"""
Inversion hypothesis test: does 'less Omega_0 -> less feedback/reionization ->
more HI' explain the p1 CDDF / tau_eff inversion in the 1P scan?

Per-scan figures, all CSV-only:
    thermal trend            T_0 and gamma against the scanned parameter
    CDDF path-length control CDDF before/after dividing out dX/dz, so the
                             geometric term cannot be mistaken for a gas signal
    FGPA residual            measured tau_eff ratio minus the thermal+Hubble
                             prediction (T0_fid/T0)^0.7 * (H_fid/H). What is left
                             is the absorber population itself -- the quantity
                             the inversion hypothesis is actually about, and the
                             one thing here that is not just an amplitude
    cross-scan direction     tau_eff and T_0 responses overlaid across p1, p2,
                             p7, p8, p9, each normalised to its own fiducial
    across-snapshot grids    one panel per snapshot, plus the geometry-divided
                             delta-tau_eff figure (multiplying by E(z) separates
                             a 1/H(z) geometric difference from a gas one)

Dropped, and why:
  * Doppler-b figures. line_width.cpp clamps b to [2, 80] km/s and 99.4% of the
    measured values at z=4 sit exactly at that ceiling, so every variant returned
    the same median. Restore once the deblender is fixed.
  * Two-band flux-power plot. degeneracy_test.py plots the band RATIO, which is
    the discriminating part; this one normalised each band to its own maximum and
    threw the ratio away.

Consumes the per-variant CSVs that `analyze_spectra.py analyze` already writes
under output/analysis/<suite>/1P/1P_p{idx}_{n2,n1,0,1,2}/snap-{XXX}/:
    cddf.csv, flux_stats.csv, power_spectrum.csv, temp_density.csv

Reads the CosmoAstroSeed CSV to map variant label -> parameter value.

Does NOT touch raw .hdf5 snapshots. Designed to run on HPC in the directory tree
where these CSVs live. Run:

    python scripts/hypothesis_test_p1.py \\
        --analysis-root output/analysis/IllustrisTNG/1P \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
        --snaps snap-080,snap-044 \\
        --out-dir plots/hypothesis_p1_test

The script is tolerant: if a file is missing for a given variant/snap, that
variant is skipped for that test and a warning is printed. A single variant
missing will never kill the whole run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------- #
# Scan definitions: which 1P parameter axis each group varies, and the #
# CosmoAstroSeed column that holds the varied value. Only p1..p5 here  #
# because those are the ones relevant to the hypothesis.               #
# -------------------------------------------------------------------- #
_1P_MEMBERS = ['n2', 'n1', '0', '1', '2']   # fixed order; '0' is fiducial

SCANS = {
    'p1': {'column': 'Omega0',       'label': r'$\Omega_0$',       'direction_pred': 'down',
           'name_fmt': '1P_p1_{s}', 'members': _1P_MEMBERS},
    'p2': {'column': 'sigma8',       'label': r'$\sigma_8$',       'direction_pred': 'down',
           'name_fmt': '1P_p2_{s}', 'members': _1P_MEMBERS},
    'p7': {'column': 'OmegaBaryon',  'label': r'$\Omega_b$',       'direction_pred': 'up',
           'name_fmt': '1P_p7_{s}', 'members': _1P_MEMBERS},
    'p8': {'column': 'HubbleParam',  'label': r'$h$',              'direction_pred': 'mild',
           'name_fmt': '1P_p8_{s}', 'members': _1P_MEMBERS},
    'p9': {'column': 'n_s',          'label': r'$n_s$',            'direction_pred': 'down',
           'name_fmt': '1P_p9_{s}', 'members': _1P_MEMBERS},
    # The EX set: four sims at one cosmology (Omega_m=0.3, sigma_8=0.8, seed
    # 13560); only A_SN1/A_AGN1 vary, so there is no parameter axis to scan.
    # 'column': None makes param_value a 1-based ordinal, i.e. a categorical
    # x-axis. EX_0 is the fiducial and sits at index 0, same as the 1P '0'.
    'ex': {'column': None,           'label': 'feedback variant', 'direction_pred': 'mild',
           'name_fmt': 'EX_{s}',    'members': ['0', '1', '2', '3']},
}
# direction_pred: predicted direction of tau_eff ratio when the parameter is
# raised above fiducial, under the "structure-growth drives feedback" hypothesis.
# 'down' = tau_eff falls (less HI). 'up' = tau_eff rises (more HI). 'mild' = small.
VARIANT_SUFFIXES = _1P_MEMBERS   # 1P default, for callers that don't hold a scan
FIDUCIAL = '0'                   # fiducial member of every scan, EX included


def members(scan):
    return SCANS[scan]['members']


# =====================================================================
# CSV loaders
# =====================================================================

def _read_csv_or_none(path, **kwargs):
    """pd.read_csv that returns None for a missing or empty/header-only file
    instead of raising EmptyDataError."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return None


def _parse_headered_csv(path):
    """Read a CSV that begins with '# key = value' lines. Return (header_dict, dataframe)."""
    header = {}
    data_start = 0
    with open(path, 'r') as fh:
        for i, line in enumerate(fh):
            if not line.startswith('#'):
                data_start = i
                break
            s = line[1:].strip()
            if not s or '=' not in s:
                continue
            k, v = s.split('=', 1)
            k, v = k.strip(), v.strip().split(' ')[0]
            try:
                header[k] = float(v)
            except ValueError:
                header[k] = v
    df = _read_csv_or_none(path, skiprows=data_start)
    return header, df


def load_temp_density(path):
    """Return T0, gamma, their errors and n_pixels from the comment header."""
    out = {'T0': np.nan, 'gamma': np.nan, 'gamma_err': np.nan, 'T0_err': np.nan,
           'n_bins_fit': np.nan, 'n_pixels': np.nan}
    if not path.exists():
        return out
    with open(path, 'r') as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            s = line[1:].strip()
            for k in out:
                if s.startswith(k + ' '):
                    _, _, v = s.partition('=')
                    v = v.strip().split(' ')[0]
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out


def load_flux_stats(path):
    """Return a dict statistic -> value."""
    df = _read_csv_or_none(path)
    if df is None:
        return {}
    return dict(zip(df['statistic'], df['value']))


def load_cddf(path):
    """Return (header_dict, dataframe) for the CDDF file."""
    if not path.exists():
        return {}, None
    return _parse_headered_csv(path)


def load_power_spectrum(path):
    return _read_csv_or_none(path)


def load_line_widths(path):
    return _read_csv_or_none(path)


# =====================================================================
# Cosmology helpers
# =====================================================================

def hubble_ratio(z, Omega_m, Omega_L=None):
    """E(z) = H(z)/H_0 for flat wCDM with w=-1."""
    if Omega_L is None:
        Omega_L = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_L)


def dXdz(z, Omega_m):
    """Cosmological absorption distance path dX/dz = (1+z)^2 / E(z)."""
    return (1.0 + z) ** 2 / hubble_ratio(z, Omega_m)


# =====================================================================
# Variant discovery
# =====================================================================

def variant_dir(analysis_root, scan, suffix):
    return analysis_root / SCANS[scan]['name_fmt'].format(s=suffix)


def _param_label(rows):
    """Axis label for whichever parameter these rows scan."""
    return rows[0]['param_label']


def snap_dir(analysis_root, scan, suffix, snap):
    return variant_dir(analysis_root, scan, suffix) / snap


def load_cosmo_table(cosmo_csv):
    """Name-indexed parameter table, from either CosmoAstroSeed format.

    1P ships a comma-separated .csv; EX ships a whitespace-aligned .txt whose
    header is commented out and whose columns are named Omega_m / sigma_8. Both
    come back with the same index and the same Omega0 / sigma8 column names, so
    everything downstream is format-blind.
    """
    first = Path(cosmo_csv).read_text().split('\n', 1)[0]
    if ',' in first:
        df = pd.read_csv(cosmo_csv)
    else:
        df = pd.read_csv(cosmo_csv, sep=r'\s+')
        df.columns = [c.lstrip('#') for c in df.columns]
        df = df.rename(columns={'Omega_m': 'Omega0', 'sigma_8': 'sigma8'})
    return df.set_index('Name')


# =====================================================================
# Assemble per-variant data for one scan at one snap
# =====================================================================

def load_snap_row(d):
    """Every CSV under one snap-XXX directory, as the row dict the observable
    extractors expect. Knows nothing about which set or scan it came from, so
    both the 1P scans and the EX set load through the same code path.

    Missing files leave fields as np.nan rather than raising, so a hole shows up
    in the figures instead of killing the run.
    """
    row = {'snap_dir': d}

    td = load_temp_density(d / 'temp_density.csv')
    row.update(td)

    fs = load_flux_stats(d / 'flux_stats.csv')
    row['mean_flux']  = fs.get('mean_flux',     np.nan)
    row['tau_eff']    = fs.get('effective_tau', np.nan)
    row['median_flux'] = fs.get('median_flux',  np.nan)
    row['deep_frac']  = fs.get('deep_absorption_frac', np.nan)
    row['weak_frac']  = fs.get('weak_absorption_frac', np.nan)
    # _err is sigma/sqrt(N) on the ensemble value; _std is the per-sightline
    # spread. Neither contains cosmic variance: one box, one realisation.
    row['tau_eff_err']   = fs.get('tau_eff_err',   np.nan)
    row['tau_eff_std']   = fs.get('tau_eff_std',   np.nan)
    row['mean_flux_err'] = fs.get('mean_flux_err', np.nan)
    row['mean_flux_std'] = fs.get('mean_flux_std', np.nan)
    # NaN outside 1.7 < z < 4 (rescaling) or without the lya_h line.
    row['tau_scale_factor'] = fs.get('tau_scale_factor', np.nan)
    row['tau_eff_H_total']  = fs.get('tau_eff_H_total',  np.nan)
    row['hi_fraction_eff']  = fs.get('hi_fraction_eff',  np.nan)

    cddf_hdr, cddf_df = load_cddf(d / 'cddf.csv')
    row['redshift']    = cddf_hdr.get('redshift',    np.nan)
    row['dX_file']     = cddf_hdr.get('dX',          np.nan)
    row['n_absorbers'] = cddf_hdr.get('n_absorbers', np.nan)
    row['beta_fit']    = cddf_hdr.get('beta_fit',     np.nan)
    row['beta_fit_err'] = cddf_hdr.get('beta_fit_err', np.nan)
    row['cddf']        = cddf_df

    row['power_spectrum'] = load_power_spectrum(d / 'power_spectrum.csv')
    row['line_widths']    = load_line_widths(d / 'line_widths.csv')

    return row


def build_scan_frame(analysis_root, cosmo_table, scan, snap):
    """Return list of dicts, one per available variant, in the order n2..2.

    Each dict carries:
      label, suffix, param_value, redshift, T0, gamma, n_pixels,
      mean_flux, tau_eff, n_absorbers, dX_file, paths ...
    Missing files leave fields as np.nan but the row is still included so you
    can see holes.
    """
    col = SCANS[scan]['column']
    rows = []
    for i, suffix in enumerate(members(scan)):
        run_label = SCANS[scan]['name_fmt'].format(s=suffix)
        row = load_snap_row(snap_dir(analysis_root, scan, suffix, snap))
        row.update({
            'suffix': suffix,
            'label': run_label,
            'scan': scan,
            'param_label': SCANS[scan]['label'],
            # A scan with no varied parameter (EX) gets a 1-based ordinal, so
            # the categorical x still plots and dividing by the fiducial's
            # value is 1.0 rather than a zero-division.
            'param_value': (i + 1 if col is None else
                            (cosmo_table.loc[run_label, col]
                             if run_label in cosmo_table.index else np.nan)),
            # NOT param_value: E(z) and dX/dz need the real matter density, which
            # every scan but p1 holds at the fiducial.
            'omega0': (cosmo_table.loc[run_label, 'Omega0']
                       if run_label in cosmo_table.index else np.nan),
        })
        rows.append(row)
    return rows


# =====================================================================
# Plots
# =====================================================================

def _setup_style():
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def plot_thermal_trend(rows, out_path, snap_label):
    """Thermal trend: T0, gamma, n_pixels vs the scanned parameter."""
    plab = _param_label(rows)
    x   = np.array([r['param_value'] for r in rows], dtype=float)
    T0  = np.array([r['T0']          for r in rows], dtype=float)
    T0e = np.array([r.get('T0_err', np.nan) for r in rows], dtype=float)
    g   = np.array([r['gamma']       for r in rows], dtype=float)
    gerr= np.array([r['gamma_err']   for r in rows], dtype=float)
    npx = np.array([r['n_pixels']    for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.errorbar(x, T0 / 1e3, yerr=T0e / 1e3, fmt='o-', color='C0', lw=2, ms=7,
                 capsize=3)
    for xi, Ti, r in zip(x, T0, rows):
        if np.isfinite(Ti):
            ax0.annotate(r['suffix'], (xi, Ti / 1e3),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax0.set_xlabel(plab)
    ax0.set_ylabel(r'$T_0$  [$10^3$ K]')
    ax0.set_title(f'IGM temperature at mean density ({snap_label})')
    ax0.grid(alpha=0.3)

    ax0b = ax0.twinx()
    ax0b.plot(x, npx / 1e6, 's--', color='C3', lw=1.2, ms=5, alpha=0.7,
              label='n_pixels in TDR fit')
    ax0b.set_ylabel(r'n_pixels$_{\rm diffuse}$ [$10^6$]', color='C3')
    ax0b.tick_params(axis='y', labelcolor='C3')

    ax1.errorbar(x, g, yerr=gerr, fmt='o-', color='C2', lw=2, ms=7, capsize=3)
    ax1.set_xlabel(plab)
    ax1.set_ylabel(r'$\gamma$ (TDR slope)')
    ax1.set_title(f'TDR slope ({snap_label})')
    ax1.grid(alpha=0.3)

    fig.suptitle('Thermal state of the diffuse IGM')
    fig.tight_layout()
    _save(fig, out_path)


def plot_cddf_pathlength(rows, out_path, snap_label):
    """CDDF path-length control: apply the analytic dX(Omega_0)/dX(fid) correction to each CDDF
    and show the ordering is preserved. Off the p1 scan Omega_0 is fixed, so the
    correction is 1 and the right panel reproduces the left -- no artefact.
    """
    plab = _param_label(rows)
    fid = next(r for r in rows if r['suffix'] == FIDUCIAL)
    z_fid = fid['redshift']
    if not np.isfinite(z_fid):
        print('  [cddf-pathlength] fiducial redshift missing, skipping'); return

    dX_fid = dXdz(z_fid, fid['omega0'])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(rows)))

    for r, c in zip(rows, colors):
        cddf = r['cddf']
        if cddf is None or np.isnan(r['param_value']):
            continue
        # Visible bins only, so autoscale ignores the DLA tail beyond the x-window.
        mask = (cddf['f_N_HI'] > 0) & cddf['log10_N_HI'].between(*CDDF_XLIM)
        lbl = f"{r['suffix']} ({plab}={r['param_value']:.2f})"
        err = _cddf_poisson_err(cddf)

        dX_var = dXdz(r['redshift'], r['omega0'])
        corr   = dX_var / dX_fid

        for ax, denom in ((axL, 1.0), (axR, corr)):
            y = cddf['f_N_HI'][mask] / denom
            ax.plot(cddf['log10_N_HI'][mask], y, 'o-', color=c, lw=2, ms=4,
                    label=lbl, alpha=0.85)
            if err is not None:
                e = err[mask] / denom
                ax.fill_between(cddf['log10_N_HI'][mask],
                                np.clip(y - e, y * 1e-2, None), y + e,
                                color=c, alpha=0.18, lw=0)

    for ax, title in [(axL, 'As-published CDDF'),
                      (axR, r'after $\times$ dX(fid)/dX($\Omega_0$)')]:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log_{10}\, N_{\rm HI}\,[{\rm cm}^{-2}]$')
        ax.set_ylabel(r'$f(N_{\rm HI})$  [cm$^{2}$]')
        ax.set_xlim(*CDDF_XLIM)
        ax.grid(alpha=0.3, which='both')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'CDDF path-length control ({snap_label})')
    fig.tight_layout()
    _save(fig, out_path)


def plot_fgpa_residual(rows, out_path, snap_label):
    """FGPA residual: compare measured tau_eff ratio to the FGPA thermal-only prediction.

    FGPA: tau ~ Delta^(2-0.7(gamma-1)) * T0^-0.7 / H(z) * Gamma_HI^-1 * (Omega_b h^2)^2
    With Omega_b, h, Gamma_HI fixed (external UVB), the variant-to-variant ratio
    at fixed density Delta=1 reduces to:
        R_pred = (T0_fid/T0)^0.7 * (H_fid/H)
    The measured ratio is R_meas = tau_eff / tau_eff_fid.
    R_meas - R_pred is the part NOT accounted for by thermal+Hubble — i.e. the
    contribution of the absorber population itself (structure/feedback).
    """
    fid = next(r for r in rows if r['suffix'] == FIDUCIAL)
    if not np.isfinite(fid['T0']) or not np.isfinite(fid['tau_eff']):
        print('  [fgpa-residual] fiducial T0/tau_eff missing, skipping'); return

    plab = _param_label(rows)
    T0_fid = fid['T0']
    H_fid  = hubble_ratio(fid['redshift'], fid['omega0'])
    tau_fid= fid['tau_eff']

    tau_fid_err = fid.get('tau_eff_err', np.nan)

    x = np.array([r['param_value'] for r in rows], dtype=float)
    R_meas, R_pred, R_meas_err = [], [], []
    for r in rows:
        if not np.isfinite(r['T0']) or not np.isfinite(r['tau_eff']):
            R_meas.append(np.nan); R_pred.append(np.nan)
            R_meas_err.append(np.nan); continue
        H_v = hubble_ratio(r['redshift'], r['omega0'])
        R_pred.append((T0_fid / r['T0']) ** 0.7 * (H_fid / H_v))
        ratio = r['tau_eff'] / tau_fid
        R_meas.append(ratio)
        # In quadrature with the fiducial's own error, so the fiducial point does
        # not come out with a zero bar.
        rel_v = r.get('tau_eff_err', np.nan) / r['tau_eff']
        rel_f = tau_fid_err / tau_fid
        R_meas_err.append(ratio * np.sqrt(rel_v ** 2 + rel_f ** 2))
    R_meas = np.array(R_meas); R_pred = np.array(R_pred)
    R_meas_err = np.array(R_meas_err, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, R_meas, yerr=R_meas_err, fmt='o-', lw=2, ms=8, capsize=3,
                label=r'measured $\tau_{\rm eff}/\tau_{\rm eff, fid}$')
    ax.plot(x, R_pred, 's--', lw=2, ms=8, label=r'FGPA thermal-only prediction')
    ax.axhline(1.0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(plab)
    ax.set_ylabel('ratio to fiducial')
    ax.set_yscale('log')
    ax.set_title(f'FGPA thermal-only vs measured ({snap_label})')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)

    return {'param_value': x.tolist(),
            'R_measured': R_meas.tolist(),
            'R_fgpa_pred': R_pred.tolist(),
            'residual': (R_meas / R_pred).tolist()}


# =====================================================================
# Cross-parameter (cross-scan direction)
# =====================================================================

def plot_cross_scan_direction(analysis_root, cosmo_table, snap, out_path,
                              scans=None, out_dir_maker=None):
    """Overlay tau_eff(parameter) and T0(parameter) for every scan that has a
    real parameter axis, each normalised to its own fiducial. Same direction
    across scans corroborates the feedback/structure-growth story.

    Needs at least two such scans: the whole content is the comparison between
    them. A categorical set such as EX has no parameter axis at all, so a run
    restricted to it would otherwise write an empty figure and an empty summary
    that look like a null result rather than an inapplicable test.
    """
    selected = list(scans) if scans else list(SCANS)
    numeric = [s for s in selected if SCANS[s]['column'] is not None]
    if len(numeric) < 2:
        print(f'  [cross-scan] needs >= 2 selected scans with a numeric parameter '
              f'axis (have {numeric}) -- skipping')
        return {}
    if out_dir_maker is not None:
        out_dir_maker()

    fig, (axT, axTau) = plt.subplots(1, 2, figsize=(13, 5))

    summary = {}
    for scan in numeric:
        meta = SCANS[scan]
        rows = build_scan_frame(analysis_root, cosmo_table, scan, snap)
        fid  = next((r for r in rows if r['suffix'] == FIDUCIAL), None)
        if fid is None or not np.isfinite(fid['tau_eff']):
            print(f'  [cross-scan direction] skip {scan}: fiducial tau_eff missing'); continue

        x = np.array([r['param_value'] for r in rows], dtype=float)
        T = np.array([r['T0']          for r in rows], dtype=float)
        tau = np.array([r['tau_eff']    for r in rows], dtype=float)

        T_rel  = T   / fid['T0']   if np.isfinite(fid['T0']) else np.full_like(T, np.nan)
        tau_rel= tau / fid['tau_eff']

        axT  .plot(x / fid['param_value'], T_rel,  'o-', lw=1.8, ms=6, label=f'{scan} ({meta["label"]})')
        axTau.plot(x / fid['param_value'], tau_rel,'o-', lw=1.8, ms=6, label=f'{scan} ({meta["label"]})')

        summary[scan] = {
            'param_value':  x.tolist(),
            'T0_over_fid':  T_rel.tolist(),
            'tau_over_fid': tau_rel.tolist(),
        }

    for ax, ylabel in [(axT, r'$T_0 / T_{0,{\rm fid}}$'),
                       (axTau, r'$\tau_{\rm eff} / \tau_{\rm eff,fid}$')]:
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.axvline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('parameter value / fiducial')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=9)

    fig.suptitle(f'Direction of effect across parameter scans ({snap})')
    fig.tight_layout()
    _save(fig, out_path)
    return summary


# =====================================================================
# Across-snapshot grids: one figure per observable, one panel per snap,
# all five Omega_0 variants overlaid in each panel. Lets you read the
# redshift evolution of the p1 ordering at a glance.
# =====================================================================

def _variant_colors(scan='p1'):
    return plt.cm.viridis(np.linspace(0, 0.9, len(members(scan))))


def _grid_axes(n, ncols=3, panel=(4.4, 3.5)):
    """Return (fig, flat_axes_list) with unused trailing axes hidden."""
    # A small grid three-across buys nothing: it still leaves an orphan at n=4
    # and n=5, and the panels come out narrower. On the page the width is what
    # sets how readable a panel is, since the figure is scaled to the text width.
    ncols = 2 if n <= 6 else min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel[0] * ncols, panel[1] * nrows),
                             squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.axis('off')
    return fig, flat


def _panel_z(rows):
    fid = next((r for r in rows if r['suffix'] == FIDUCIAL), None)
    z = fid['redshift'] if fid is not None else np.nan
    return z


def _panel_title(rows, snap):
    z = _panel_z(rows)
    return f'{snap}  (z = {z:.2f})' if np.isfinite(z) else snap


def _frames_scan(frames):
    """(scan_name, axis_label) for a {snap: rows} mapping."""
    rows = next(iter(frames.values()))
    return rows[0]['scan'], rows[0]['param_label']


def _cddf_poisson_err(cddf):
    """Per-bin Poisson error on f(N). Falls back to f/sqrt(counts) for CSVs written
    before the f_N_HI_err column existed."""
    if cddf is None:
        return None
    if 'f_N_HI_err' in cddf.columns:
        return cddf['f_N_HI_err']
    if 'counts' in cddf.columns:
        counts = cddf['counts'].astype(float)
        return cddf['f_N_HI'] / np.sqrt(counts.where(counts > 0))
    return None


CDDF_XLIM = (13.0, 16.0)   # data starts at log N = 13.11; 12 wastes a third of the panel


def _grid_cddf(frames, snaps, out_path):
    scan, plab = _frames_scan(frames)
    colors = _variant_colors(scan)
    fig, flat = _grid_axes(len(snaps))
    for ax, snap in zip(flat, snaps):
        rows = frames[snap]
        shown = []          # f values inside CDDF_XLIM, for the y-limits below
        for r, c in zip(rows, colors):
            cddf = r['cddf']
            if cddf is None or not np.isfinite(r['param_value']):
                continue
            m = cddf['f_N_HI'] > 0
            ax.plot(cddf['log10_N_HI'][m], cddf['f_N_HI'][m], '-',
                    color=c, lw=1.5, label=f"{r['param_value']:.1f}")
            vis = m & cddf['log10_N_HI'].between(*CDDF_XLIM)
            shown.append(cddf['f_N_HI'][vis].values)
            # The high-N bins hold single-digit counts in a 25 Mpc/h box, so
            # f - err goes to zero there. Floor the band at a fixed fraction of
            # f: clipping to an absolute 1e-300 instead puts that value into the
            # axis data limits and autoscales the whole panel down 300 decades.
            err = _cddf_poisson_err(cddf)
            if err is not None:
                ax.fill_between(cddf['log10_N_HI'][m],
                                np.clip(cddf['f_N_HI'][m] - err[m],
                                        cddf['f_N_HI'][m] * 1e-2, None),
                                cddf['f_N_HI'][m] + err[m],
                                color=c, alpha=0.18, lw=0)
        ax.set_yscale('log')
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(r'$\log_{10}\, N_{\rm HI}$')
        ax.set_ylabel(r'$f(N_{\rm HI})$ [cm$^{2}$]')
        ax.set_xlim(*CDDF_XLIM)
        # f spans ~14 decades out to log N = 21.5 while only 12--16 is drawn,
        # and matplotlib autoscales over all data, not the x-window. Set the
        # limits from the visible bins alone.
        vals = np.concatenate(shown) if shown else np.array([])
        if vals.size:
            ax.set_ylim(0.1 * vals.min(), 10.0 * vals.max())
        ax.grid(alpha=0.3, which='both')
    flat[0].legend(title=plab, fontsize=7, loc='best')
    fig.suptitle(f'CDDF vs redshift ({scan} {plab} scan)')
    fig.tight_layout()
    _save(fig, out_path)


def _grid_power(frames, snaps, out_path):
    scan, plab = _frames_scan(frames)
    colors = _variant_colors(scan)
    fig, flat = _grid_axes(len(snaps))
    for ax, snap in zip(flat, snaps):
        rows = frames[snap]
        for r, c in zip(rows, colors):
            ps = r['power_spectrum']
            if ps is None or not np.isfinite(r['param_value']):
                continue
            k = ps['k_s_per_km'].values
            P = ps['P_k_mean_km_per_s'].values
            m = (k > 0) & (P > 0)
            ax.loglog(k[m], P[m], '-', color=c, lw=1.5,
                      label=f"{r['param_value']:.1f}")
            if 'P_k_err' in ps.columns:
                e = ps['P_k_err'].values
                ax.fill_between(k[m], np.clip(P[m] - e[m], P[m] * 1e-2, None), P[m] + e[m],
                                color=c, alpha=0.18, lw=0)
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(r'$k$ [s/km]')
        ax.set_ylabel(r'$P_F(k)$ [km/s]')
        ax.grid(alpha=0.3, which='both')
    flat[0].legend(title=plab, fontsize=7, loc='best')
    fig.suptitle(f'Flux power spectrum vs redshift ({scan} {plab} scan)')
    fig.tight_layout()
    _save(fig, out_path)


def _grid_scalar(frames, snaps, out_path, key, ylabel, title,
                 scale=1.0, logy=False, err_key=None, band_key=None):
    """One panel per snap: scalar quantity `key` vs the scanned parameter.

    err_key -> error bar (sigma/sqrt(N)); band_key -> shaded band (per-sightline
    sigma). Both internal to one box.
    """
    _, plab = _frames_scan(frames)
    fig, flat = _grid_axes(len(snaps))
    for ax, snap in zip(flat, snaps):
        rows = frames[snap]
        x = np.array([r['param_value'] for r in rows], dtype=float)
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float) * scale
        yerr = None
        if err_key is not None:
            yerr = np.array([r.get(err_key, np.nan) for r in rows], dtype=float) * scale
            if not np.any(np.isfinite(yerr)):
                yerr = None
        if band_key is not None:
            band = np.array([r.get(band_key, np.nan) for r in rows], dtype=float) * scale
            if np.any(np.isfinite(band)):
                ax.fill_between(x, y - band, y + band, color='C0', alpha=0.15,
                                lw=0, label=r'$\pm\sigma$ per sightline')
        ax.errorbar(x, y, yerr=yerr, fmt='o-', color='C0', lw=1.8, ms=6, capsize=3)
        if logy:
            ax.set_yscale('log')
        ax.set_title(_panel_title(rows, snap))
        ax.set_xlabel(plab)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which='both')
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path)


def _overlay_tau_eff(frames, snaps, out_path):
    """Single panel: tau_eff vs the scanned parameter, one line per snap."""
    _, plab = _frames_scan(frames)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(snaps)))
    for snap, c in zip(snaps, cmap):
        rows = frames[snap]
        x = np.array([r['param_value'] for r in rows], dtype=float)
        y = np.array([r['tau_eff']     for r in rows], dtype=float)
        e = np.array([r.get('tau_eff_err', np.nan) for r in rows], dtype=float)
        s = np.array([r.get('tau_eff_std', np.nan) for r in rows], dtype=float)
        z = _panel_z(rows)
        lbl = f'{snap} (z={z:.2f})' if np.isfinite(z) else snap
        if np.any(np.isfinite(s)):
            ax.fill_between(x, y - s, y + s, color=c, alpha=0.12, lw=0)
        ax.errorbar(x, y, yerr=e if np.any(np.isfinite(e)) else None,
                    fmt='o-', color=c, lw=2, ms=6, capsize=3, label=lbl)
    ax.set_yscale('log')
    ax.set_xlabel(plab)
    ax.set_ylabel(r'$\tau_{\rm eff}$')
    ax.set_title(rf'$\tau_{{\rm eff}}$({plab}) across redshift — ordering crossover'
                 '\n' r'bars: $\sigma/\sqrt{N}$   bands: per-sightline $\sigma$')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def _delta_tau_eff_vs_z(frames, snaps, out_path):
    """tau_eff(high parameter) - tau_eff(low parameter) as a function of z.

    Four panels: the raw difference, the same divided by the fiducial (so the
    z = 4 and z = 0.27 points are comparable), and both again after multiplying
    tau_eff by E(z). tau_eff per unit velocity carries a 1/H(z) factor, so a
    difference that vanishes in the bottom row is geometry, not gas.
    """
    scan, label = _frames_scan(frames)
    entries = []
    for snap in snaps:
        rows = frames[snap]
        z = _panel_z(rows)
        usable = [r for r in rows
                  if np.isfinite(r['param_value']) and np.isfinite(r['tau_eff'])]
        if not np.isfinite(z) or len(usable) < 2:
            continue
        usable.sort(key=lambda r: r['param_value'])
        entries.append({'z': z, 'lo': usable[0], 'hi': usable[-1], 'rows': usable,
                        'fid': next((r for r in usable if r['suffix'] == FIDUCIAL),
                                    None)})

    if len(entries) < 2:
        print('  [delta-tau_eff] need >= 2 snapshots with two variants - skipping')
        return

    entries.sort(key=lambda e: e['z'])
    z = np.array([e['z'] for e in entries], float)

    def _err(r):
        return r.get('tau_eff_err', np.nan)

    def _Ez(r, zz):
        return hubble_ratio(zz, r['omega0'])

    d_raw, d_raw_err, d_geo, d_geo_err = [], [], [], []
    fid_raw, fid_geo = [], []
    for e, zz in zip(entries, z):
        hi, lo = e['hi'], e['lo']
        d_raw.append(hi['tau_eff'] - lo['tau_eff'])
        d_raw_err.append(np.sqrt(_err(hi) ** 2 + _err(lo) ** 2))

        hi_g = hi['tau_eff'] * _Ez(hi, zz)
        lo_g = lo['tau_eff'] * _Ez(lo, zz)
        d_geo.append(hi_g - lo_g)
        d_geo_err.append(np.sqrt((_err(hi) * _Ez(hi, zz)) ** 2 +
                                 (_err(lo) * _Ez(lo, zz)) ** 2))

        f = e['fid']
        fid_raw.append(f['tau_eff'] if f is not None else np.nan)
        fid_geo.append(f['tau_eff'] * _Ez(f, zz) if f is not None else np.nan)

    d_raw = np.array(d_raw); d_raw_err = np.array(d_raw_err, float)
    d_geo = np.array(d_geo); d_geo_err = np.array(d_geo_err, float)
    fid_raw = np.array(fid_raw); fid_geo = np.array(fid_geo)

    p_lo = entries[0]['lo']['param_value']
    p_hi = entries[0]['hi']['param_value']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    def _bars(ax, y, yerr, color, lbl=None, ms=7):
        ax.errorbar(z, y, yerr=yerr if np.any(np.isfinite(yerr)) else None,
                    fmt='o-', color=color, lw=2, ms=ms, capsize=3, label=lbl)
        ax.axhline(0.0, color='gray', lw=0.8, ls=':')
        ax.grid(alpha=0.3)

    _bars(axes[0][0], d_raw, d_raw_err, 'C0',
          lbl=f'{label} = {p_hi:.2f} minus {p_lo:.2f}')
    axes[0][0].set_ylabel(r'$\Delta \tau_{\rm eff}$')
    axes[0][0].set_title(r'$\tau_{\rm eff}$ difference across the scan')
    axes[0][0].legend(fontsize=9)

    with np.errstate(divide='ignore', invalid='ignore'):
        _bars(axes[0][1], d_raw / fid_raw, d_raw_err / fid_raw, 'C0')
    axes[0][1].set_ylabel(r'$\Delta \tau_{\rm eff} / \tau_{\rm eff}^{\rm fid}$')
    axes[0][1].set_title('same, relative to the fiducial')

    _bars(axes[1][0], d_geo, d_geo_err, 'C1')
    axes[1][0].set_ylabel(r'$\Delta [\tau_{\rm eff} E(z)]$')
    axes[1][0].set_title(r'geometry removed ($\times E(z) = H(z)/H_0$)')

    with np.errstate(divide='ignore', invalid='ignore'):
        _bars(axes[1][1], d_geo / fid_geo, d_geo_err / fid_geo, 'C1')
    axes[1][1].set_ylabel(r'$\Delta [\tau_{\rm eff} E] / [\tau_{\rm eff} E]^{\rm fid}$')
    axes[1][1].set_title('same, relative to the fiducial')

    # Per-variant differences against the fiducial, so the ordering is visible
    # and not just the two extremes.
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(members(scan))))
    for suf, c in zip(members(scan), colors):
        if suf == FIDUCIAL:
            continue
        zv, dv, pv = [], [], np.nan
        for e, zz in zip(entries, z):
            r = next((x for x in e['rows'] if x['suffix'] == suf), None)
            if r is None or e['fid'] is None:
                continue
            zv.append(zz)
            dv.append(r['tau_eff'] - e['fid']['tau_eff'])
            pv = r['param_value']
        if len(zv) >= 2:
            axes[0][0].plot(zv, dv, '-', color=c, lw=1.0, alpha=0.7,
                            label=f'{pv:.2f} - fid')
    axes[0][0].legend(fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel('redshift z')

    fig.suptitle(rf'$\Delta \tau_{{\rm eff}}(z)$ across the {label} scan '
                 r'— bars: $\sigma/\sqrt{N}$, added in quadrature', fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)

    return {'z': z.tolist(),
            'param_low': float(p_lo), 'param_high': float(p_hi),
            'delta_tau_eff': d_raw.tolist(),
            'delta_tau_eff_err': d_raw_err.tolist(),
            'delta_tau_eff_Ez': d_geo.tolist(),
            'delta_tau_eff_Ez_err': d_geo_err.tolist()}


def make_snapshot_grids(analysis_root, cosmo_table, snaps, out_dir, scan='p1'):
    """Build the across-snapshot comparison figures (one panel per snap)."""
    plab = SCANS[scan]['label']
    print(f'\n=== across-snapshot grids, {scan} scan ===')
    frames = {snap: build_scan_frame(analysis_root, cosmo_table, scan, snap)
              for snap in snaps}
    # high-z first so panels read left->right as cosmic time advances backward
    snaps_sorted = sorted(
        snaps,
        key=lambda s: (_panel_z(frames[s]) if np.isfinite(_panel_z(frames[s]))
                       else -np.inf),
        reverse=True)

    grid_dir = out_dir / scan / 'across_snapshots'
    grid_dir.mkdir(parents=True, exist_ok=True)

    _grid_cddf  (frames, snaps_sorted, grid_dir / 'grid_CDDF.png')
    _grid_power (frames, snaps_sorted, grid_dir / 'grid_power_spectrum.png')
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_tau_eff.png',
                 key='tau_eff', ylabel=r'$\tau_{\rm eff}$',
                 title=f'Effective optical depth vs redshift ({scan} {plab} scan)',
                 logy=True, err_key='tau_eff_err', band_key='tau_eff_std')
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_mean_flux.png',
                 key='mean_flux', ylabel=r'$\langle F \rangle$',
                 title=f'Mean transmitted flux vs redshift ({scan} {plab} scan)',
                 err_key='mean_flux_err', band_key='mean_flux_std')
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_T0.png',
                 key='T0', ylabel=r'$T_0$ [$10^3$ K]', scale=1e-3,
                 title=f'IGM $T_0$ vs redshift ({scan} {plab} scan)',
                 err_key='T0_err')
    # NaN above z = 4.5, where the forest is saturated.
    _grid_scalar(frames, snaps_sorted, grid_dir / 'grid_beta.png',
                 key='beta_fit', ylabel=r'$\beta$ (CDDF slope)',
                 title=f'CDDF power-law slope vs redshift ({scan} {plab} scan)',
                 err_key='beta_fit_err')
    _overlay_tau_eff(frames, snaps_sorted, grid_dir / f'tau_eff_vs_{scan}_overlay.png')
    delta = _delta_tau_eff_vs_z(frames, snaps_sorted,
                                grid_dir / 'delta_tau_eff_vs_z.png')
    if delta is not None:
        with open(grid_dir / 'delta_tau_eff_vs_z.json', 'w') as fh:
            json.dump(delta, fh, indent=2)
        print(f'  saved {grid_dir / "delta_tau_eff_vs_z.json"}')


# =====================================================================
# Entry
# =====================================================================

def run_one_snap(analysis_root, cosmo_table, snap, out_dir, scan='p1'):
    print(f'\n=== {scan} scan, {snap} ===')
    rows = build_scan_frame(analysis_root, cosmo_table, scan, snap)
    snap_out = out_dir / scan / snap
    snap_out.mkdir(parents=True, exist_ok=True)

    summary = {'scan': scan, 'snap': snap,
               'param': SCANS[scan]['column'], 'variants': []}
    for r in rows:
        summary['variants'].append({
            'suffix': r['suffix'], 'label': r['label'],
            'param_value': r['param_value'],
            'omega0': r['omega0'],
            'redshift': r['redshift'],
            'T0_K': r['T0'], 'gamma': r['gamma'], 'gamma_err': r['gamma_err'],
            'n_pixels_diffuse': r['n_pixels'],
            'mean_flux': r['mean_flux'],
            'tau_eff': r['tau_eff'],
            'deep_absorption_frac': r['deep_frac'],
            'weak_absorption_frac': r['weak_frac'],
            'n_absorbers_cddf': r['n_absorbers'],
            'dX_from_file_Mpc': r['dX_file'],
            'dX_analytic_dzdX': dXdz(r['redshift'], r['omega0'])
                               if np.isfinite(r['redshift']) and np.isfinite(r['omega0']) else np.nan,
        })

    plot_thermal_trend      (rows, snap_out / 'thermal_trend_vs_param.png', snap)
    plot_cddf_pathlength    (rows, snap_out / 'cddf_pathlength_control.png', snap)
    fgpa = plot_fgpa_residual(rows, snap_out / 'fgpa_vs_measured.png', snap)

    summary['fgpa_residual'] = fgpa
    return summary, snap_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', required=True, type=Path,
                    help='directory containing 1P_pK_{suffix}/snap-XXX/ trees '
                         '(e.g. output/analysis/IllustrisTNG/1P)')
    ap.add_argument('--cosmo-csv', required=True, type=Path,
                    help='CosmoAstroSeed CSV (keyed by Name column)')
    ap.add_argument('--snaps', default='snap-080,snap-044',
                    help='comma-separated snap dirs to process')
    ap.add_argument('--scans', default='p1,p2',
                    help='comma-separated scans to process (any of '
                         + ','.join(SCANS) + '). p2 is what shows that sigma8 '
                         'leaves no 1/H(z) geometric footprint.')
    ap.add_argument('--out-dir', type=Path,
                    default=Path('plots/hypothesis_p1_test'))
    ap.add_argument('--skip-cross-param', action='store_true',
                    help='skip the cross-scan direction p1..p5 cross-parameter figure')
    ap.add_argument('--skip-grids', action='store_true',
                    help='skip the across-snapshot comparison grids')
    args = ap.parse_args()

    _setup_style()
    cosmo = load_cosmo_table(args.cosmo_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        ap.error(f'unknown scan(s) {unknown}; known: {list(SCANS)}')

    all_summary = {}
    for scan in scans:
        for snap in snaps:
            s, snap_out = run_one_snap(args.analysis_root, cosmo, snap,
                                       args.out_dir, scan=scan)
            all_summary[(scan, snap)] = s
            with open(snap_out / 'summary.json', 'w') as fh:
                json.dump(s, fh, indent=2, default=float)

    # cross-scan direction already overlays every scan in one figure, so it is per-snap, not
    # per-scan. Keep it at the top level rather than duplicating it under each.
    if not args.skip_cross_param:
        for snap in snaps:
            snap_out = args.out_dir / 'cross_parameter' / snap
            # Ask first, then make the directory -- a skipped test should leave
            # no trace at all, not an empty dir that reads as a failed run.
            xp = plot_cross_scan_direction(
                args.analysis_root, cosmo, snap,
                snap_out / 'cross_scan_direction.png', scans=scans,
                out_dir_maker=lambda: snap_out.mkdir(parents=True, exist_ok=True))
            if not xp:
                continue
            with open(snap_out / 'summary.json', 'w') as fh:
                json.dump({'snap': snap, 'cross_scan_direction': xp}, fh,
                          indent=2, default=float)

    if not args.skip_grids:
        for scan in scans:
            make_snapshot_grids(args.analysis_root, cosmo, snaps,
                                args.out_dir, scan=scan)

    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
