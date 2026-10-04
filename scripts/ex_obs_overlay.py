"""
Overlay the EX feedback variants and the observed spectrum.

EX_0..EX_3 share cosmology and IC seed and differ only in feedback (fiducial, extreme AGN,
extreme SN, none), so any separation between their curves is feedback alone. The observation
goes on the same axes.

This reads the CSVs that `analyze` and `obs_analyze` already wrote, so it needs no spectra
files and runs in seconds.

**The observation is not measured like the simulations here.** It carries COS resolution,
COS noise and a one-sided tau clip; the simulated curves are noiseless at 0.1 km/s pixels.
That mismatch dominates the flux PDF and everything in the power spectrum beyond the COS
resolution limit, which is drawn as a vertical line. Treat this as the orientation plot: it
shows whether the feedback variants separate at all, and roughly where the data sits. The
like-for-like version needs EX sightlines pushed through observations.forward_model first,
which needs the EX spectra regenerated.

    python scripts/ex_obs_overlay.py --snap snap-090
    python scripts/ex_obs_overlay.py --snap snap-080 --obs output/analysis/Obs/pg1048_all
    python scripts/ex_obs_overlay.py --self-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EX_SIMS = ['EX_0', 'EX_1', 'EX_2', 'EX_3']
EX_LABEL = {
    'EX_0': 'EX_0 fiducial',
    'EX_1': r'EX_1 extreme AGN ($A_{\rm AGN1}{=}100$)',
    'EX_2': r'EX_2 extreme SN ($A_{\rm SN1}{=}100$)',
    'EX_3': 'EX_3 no feedback',
}
EX_COLOR = {'EX_0': 'k', 'EX_1': 'C3', 'EX_2': 'C0', 'EX_3': 'C2'}
OBS_COLOR = 'C1'

# COS G130M resolution: the observed power spectrum is damped by the LSF above this k.
COS_FWHM_KMS = 17.0


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


def read_table(path):
    """Read one of the analysis CSVs, skipping the '#' header lines."""
    path = Path(path)
    if not path.exists():
        return None
    rows, header = [], None
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if header is None:
            header = [c.strip() for c in line.split(',')]
            continue
        parts = line.split(',')
        if len(parts) != len(header):
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    if header is None or not rows:
        return None
    data = np.array(rows, dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def read_flux_stats(path):
    """flux_stats.csv is a two-column statistic/value table, not a data table."""
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines()[1:]:
        if ',' not in line or line.startswith('#'):
            continue
        key, _, value = line.partition(',')
        try:
            out[key.strip()] = float(value)
        except ValueError:
            continue           # tau_eff_per_sightline is an array literal
    return out


def load_set(root, snap, sims=EX_SIMS):
    """Every CSV for each simulation at one snapshot."""
    out = {}
    for sim in sims:
        d = Path(root) / sim / snap
        out[sim] = {
            'power_spectrum': read_table(d / 'power_spectrum.csv'),
            'flux_pdf': read_table(d / 'flux_pdf.csv'),
            'tau_pdf': read_table(d / 'tau_pdf.csv'),
            'line_widths': read_table(d / 'line_widths.csv'),
            'flux_stats': read_flux_stats(d / 'flux_stats.csv'),
        }
    return out


def load_obs(obs_dir):
    d = Path(obs_dir)
    return {
        'power_spectrum': read_table(d / 'power_spectrum.csv'),
        'flux_pdf': read_table(d / 'flux_pdf.csv'),
        'tau_pdf': read_table(d / 'tau_pdf.csv'),
        'line_widths': read_table(d / 'line_widths.csv'),
        'flux_stats': read_flux_stats(d / 'flux_stats.csv'),
    }


# tau_eff was called effective_tau before the 2026-07 fix; both appear in the CSVs on disk.
STAT_ALIASES = {'tau_eff': ('tau_eff', 'effective_tau')}


def _stat(stats, key):
    for name in STAT_ALIASES.get(key, (key,)):
        if name in stats:
            return stats[name]
    return float('nan')


def _first(table, *names):
    """Pick whichever column name this CSV generation used."""
    if table is None:
        return None
    for name in names:
        if name in table:
            return table[name]
    return None


def plot_power(ex, obs, out_path, snap, ratio=True):
    """Dimensionless flux power k P(k) / pi, with the ratio to EX_0 below."""
    _setup_style()
    if ratio:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.07})
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        axr = None

    def curve(table):
        k = _first(table, 'k_s_per_km', 'k')
        p = _first(table, 'P_k_mean_km_per_s', 'P_k_mean')
        if k is None or p is None:
            return None, None
        ok = k > 0
        return k[ok], (k * p / np.pi)[ok]

    k0, p0 = curve(ex['EX_0']['power_spectrum'])
    for sim in EX_SIMS:
        k, p = curve(ex[sim]['power_spectrum'])
        if k is None:
            continue
        ax.plot(k, p, color=EX_COLOR[sim], lw=1.6, label=EX_LABEL[sim],
                ls='--' if sim == 'EX_0' else '-')
        if axr is not None and k0 is not None and sim != 'EX_0':
            axr.plot(k, p / np.interp(k, k0, p0), color=EX_COLOR[sim], lw=1.4)

    if obs is not None:
        k, p = curve(obs['power_spectrum'])
        if k is not None:
            ax.plot(k, p, color=OBS_COLOR, lw=2.0, marker='o', ms=3, label='observed (PG1048)')
            if axr is not None and k0 is not None:
                axr.plot(k, p / np.interp(k, k0, p0), color=OBS_COLOR, lw=1.6, marker='o', ms=3)

    k_res = 1.0 / COS_FWHM_KMS
    for a in filter(None, [ax, axr]):
        a.axvline(k_res, color='0.5', ls=':', lw=1)
        a.set_xscale('log')
        a.grid(alpha=0.25)
    ax.text(k_res * 1.1, ax.get_ylim()[1] * 0.5, 'COS resolution', rotation=90,
            fontsize=8, color='0.4', va='top')
    ax.set_yscale('log')
    ax.set_ylabel(r'$k P_F(k) / \pi$')
    ax.set_title(f'flux power spectrum, EX set at {snap}')
    ax.legend()
    if axr is not None:
        axr.axhline(1.0, color='k', lw=0.8)
        axr.set_ylabel('/ EX_0')
        axr.set_xlabel(r'$k$ [s/km]')
    else:
        ax.set_xlabel(r'$k$ [s/km]')
    _save(fig, out_path)


def plot_pdf(ex, obs, which, out_path, snap):
    """Flux PDF or log tau PDF, overlaid."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    xcol = ('flux_bin_center', 'bin_center') if which == 'flux_pdf' else \
           ('log_tau_bin_center', 'bin_center', 'tau_bin_center')

    for sim in EX_SIMS:
        table = ex[sim][which]
        x, y = _first(table, *xcol), _first(table, 'density')
        if x is None or y is None:
            continue
        ax.plot(x, y, color=EX_COLOR[sim], lw=1.6, label=EX_LABEL[sim],
                ls='--' if sim == 'EX_0' else '-')

    if obs is not None:
        table = obs[which]
        x, y = _first(table, *xcol), _first(table, 'density')
        err = _first(table, 'density_err')
        if x is not None and y is not None:
            ax.plot(x, y, color=OBS_COLOR, lw=2.0, marker='o', ms=3, label='observed (PG1048)')
            if err is not None:
                ax.fill_between(x, y - err, y + err, color=OBS_COLOR, alpha=0.25, lw=0)

    ax.set_yscale('log')
    ax.set_xlabel('transmitted flux' if which == 'flux_pdf' else r'$\log_{10}\tau$')
    ax.set_ylabel('probability density')
    ax.set_title(f'{"flux" if which == "flux_pdf" else "optical depth"} PDF, EX set at {snap}')
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, out_path)


B_BINS = np.linspace(0, 150, 46)


def _b_density(table):
    """line_widths.csv is a per-line catalogue (N_HI, b), so bin it here."""
    b = _first(table, 'b_param_km_s', 'b_param', 'b')
    if b is None or b.size == 0:
        return None, None
    counts, edges = np.histogram(b[np.isfinite(b)], bins=B_BINS)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, counts / max(counts.sum(), 1) / np.diff(edges)


def plot_line_widths(ex, obs, out_path, snap):
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for sim in EX_SIMS:
        x, y = _b_density(ex[sim]['line_widths'])
        if x is None:
            continue
        ax.plot(x, y, color=EX_COLOR[sim], lw=1.6, label=EX_LABEL[sim],
                ls='--' if sim == 'EX_0' else '-')
    if obs is not None:
        x, y = _b_density(obs['line_widths'])
        if x is not None:
            ax.plot(x, y, color=OBS_COLOR, lw=2.0, marker='o', ms=3, label='observed (PG1048)')
    ax.axvline(COS_FWHM_KMS, color='0.5', ls=':', lw=1)
    ax.text(COS_FWHM_KMS * 1.05, ax.get_ylim()[1] * 0.5, 'COS resolution',
            rotation=90, fontsize=8, color='0.4', va='top')
    ax.set_xlabel('b [km/s]')
    ax.set_ylabel('probability density')
    ax.set_title(f'line widths, EX set at {snap}')
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, out_path)


def plot_scalars(ex, obs, out_path, snap, keys=('mean_flux', 'tau_eff')):
    """Scalar observables as points, with the observation as a horizontal line."""
    _setup_style()
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.8))
    for ax, key in zip(np.atleast_1d(axes), keys):
        vals = [_stat(ex[sim]['flux_stats'], key) for sim in EX_SIMS]
        errs = [ex[sim]['flux_stats'].get(f'{key}_err', np.nan) for sim in EX_SIMS]
        x = np.arange(len(EX_SIMS))
        ax.errorbar(x, vals, yerr=errs, fmt='o', color='k', capsize=3)
        for i, sim in enumerate(EX_SIMS):
            ax.plot(x[i], vals[i], 'o', color=EX_COLOR[sim], ms=8)
        if obs is not None and np.isfinite(_stat(obs['flux_stats'], key)):
            o = _stat(obs['flux_stats'], key)
            oerr = obs['flux_stats'].get(f'{key}_err', 0.0)
            ax.axhline(o, color=OBS_COLOR, lw=1.8, label='observed')
            ax.axhspan(o - oerr, o + oerr, color=OBS_COLOR, alpha=0.2)
            ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('EX_', '') for s in EX_SIMS])
        ax.set_xlabel('EX variant')
        ax.set_ylabel(key)
        ax.grid(alpha=0.25)
    fig.suptitle(f'EX set at {snap} against PG1048')
    fig.tight_layout()
    _save(fig, out_path)


def self_test():
    rng = np.random.default_rng(0)
    k = np.logspace(-3, -1, 40)

    def fake(scale):
        return {
            'power_spectrum': {'k_s_per_km': k, 'P_k_mean_km_per_s': scale * k ** -1.5},
            'flux_pdf': {'flux_bin_center': np.linspace(0, 1, 50),
                         'density': scale * np.exp(np.linspace(0, 3, 50)),
                         'density_err': np.full(50, 0.01)},
            'tau_pdf': {'log_tau_bin_center': np.linspace(-3, 1, 40),
                        'density': scale * np.ones(40)},
            'line_widths': {'N_HI': rng.lognormal(31, 1, 400),
                            'b_param_km_s': rng.normal(40 * scale, 12, 400)},
            'flux_stats': {'mean_flux': 0.9 * scale, 'effective_tau': 0.1 / scale,
                           'tau_eff_err': 0.01},
        }

    ex = {sim: fake(1.0 + 0.1 * i) for i, sim in enumerate(EX_SIMS)}
    obs = fake(0.95)

    # The CSV reader must survive the '#' header block and a trailing array literal.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'flux_pdf.csv'
        p.write_text('# n_pixels = 10\n# grid = fixed\nflux_bin_center,density\n0.5,1.0\n0.7,2.0\n')
        table = read_table(p)
        assert table is not None and table['density'][1] == 2.0, table

        s = Path(tmp) / 'flux_stats.csv'
        s.write_text('statistic,value\nmean_flux,0.95\ntau_eff,0.046\n'
                     'tau_eff_per_sightline,"[0.1 0.2]"\n')
        stats = read_flux_stats(s)
        assert stats['mean_flux'] == 0.95 and 'tau_eff_per_sightline' not in stats

        assert read_table(Path(tmp) / 'missing.csv') is None

        plot_power(ex, obs, Path(tmp) / 'p.png', 'snap-090')
        plot_pdf(ex, obs, 'flux_pdf', Path(tmp) / 'f.png', 'snap-090')
        plot_pdf(ex, obs, 'tau_pdf', Path(tmp) / 't.png', 'snap-090')
        plot_line_widths(ex, obs, Path(tmp) / 'b.png', 'snap-090')
        plot_scalars(ex, obs, Path(tmp) / 's.png', 'snap-090')
        assert (Path(tmp) / 's.png').exists()

        # A missing simulation must not take the whole figure down.
        broken = {**ex, 'EX_2': {k2: None for k2 in ex['EX_2']}}
        broken['EX_2']['flux_stats'] = {}
        plot_power(broken, obs, Path(tmp) / 'p2.png', 'snap-090')

    print('ex_obs_overlay self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snap', default='snap-090',
                        help='snapshot directory name (default snap-090, z = 0)')
    parser.add_argument('--ex-root', default='output/analysis/IllustrisTNG/EX')
    parser.add_argument('--obs', default='output/analysis/Obs/pg1048_all',
                        help='obs_analyze output directory; "none" to omit')
    parser.add_argument('--out-dir', default=None, help='default plots/ex_obs/<snap>')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    ex = load_set(args.ex_root, args.snap)
    missing = [s for s in EX_SIMS if ex[s]['power_spectrum'] is None]
    if missing:
        print(f"warning: no power_spectrum.csv for {', '.join(missing)} at {args.snap}")
    obs = None if args.obs.lower() == 'none' else load_obs(args.obs)
    if obs is not None and obs['power_spectrum'] is None:
        print(f'warning: no observed CSVs in {args.obs}')

    out_dir = Path(args.out_dir or f'plots/ex_obs/{args.snap}')
    plot_power(ex, obs, out_dir / 'power_spectrum.png', args.snap)
    plot_pdf(ex, obs, 'flux_pdf', out_dir / 'flux_pdf.png', args.snap)
    plot_pdf(ex, obs, 'tau_pdf', out_dir / 'tau_pdf.png', args.snap)
    plot_line_widths(ex, obs, out_dir / 'line_widths.png', args.snap)
    plot_scalars(ex, obs, out_dir / 'flux_scalars.png', args.snap)

    print(f'\n{args.snap}: scalar comparison')
    print(f"  {'':<10}{'mean_flux':>12}{'tau_eff':>12}")
    for sim in EX_SIMS:
        s = ex[sim]['flux_stats']
        print(f"  {sim:<10}{_stat(s, 'mean_flux'):>12.4f}{_stat(s, 'tau_eff'):>12.4f}")
    if obs is not None:
        s = obs['flux_stats']
        print(f"  {'observed':<10}{_stat(s, 'mean_flux'):>12.4f}{_stat(s, 'tau_eff'):>12.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
