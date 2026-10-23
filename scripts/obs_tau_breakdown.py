"""
Account for the difference between the observed and simulated effective optical depth.

The observed tau_eff over the whole COS sample is 1.7 times the simulated one at z = 0.
This splits that factor into its parts and says how large each is:

  redshift          the observed absorbers run to z = 0.6 while the comparison used z = 0
  intervening metals  identified in the collaborator's own VPFIT fits
  Galactic lines    the Milky Way ISM, which her sightline windows do not fully exclude

What is left over after those three is the real disagreement, and it is not spread evenly
across the band -- which is the point of the wavelength figure.

Also measures the feedback discriminant, the fraction of simulated lines narrower than
30 km/s, at every available redshift, because a signal that only exists in one snapshot is
not a signal.

    python scripts/obs_tau_breakdown.py --analysis-root /path/to/cluster/output/analysis
    python scripts/obs_tau_breakdown.py --self-test
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observations as obs
import vpfit_catalog as vp

LYA = 1215.67
C_KMS = 299792.458

# Verified snapshot redshifts, not interpolated from the CAMELS documentation.
SNAP_Z = {'090': 0.0000, '088': 0.0485, '086': 0.1001, '084': 0.1541,
          '082': 0.2101, '080': 0.2696, '072': 0.5373}
SIMS = {'EX_0': 'fiducial', 'EX_1': 'extreme AGN', 'EX_2': 'extreme SN',
        'EX_3': 'no feedback'}

# Rest wavelengths for the ions the collaborator identifies, so each fitted metal
# component can be placed. A component is masked at every transition of its ion, since
# the fit constrains them jointly.
METAL_REST = {
    'CII': [1334.53], 'CIII': [977.02], 'CIV': [1548.20, 1550.78],
    'NII': [1083.99], 'NIII': [989.80], 'NV': [1238.82, 1242.80],
    'OI': [1302.17], 'OVI': [1031.93, 1037.62],
    'SiII': [1190.42, 1193.29, 1260.42, 1304.37, 1526.71],
    'SiIII': [1206.50], 'SiIV': [1393.76, 1402.77],
    'FeII': [1144.94, 1608.45], 'AlII': [1670.79],
    'SII': [1250.58, 1253.81], 'SIII': [1190.21], 'SVI': [933.38, 944.52],
    'NeVIII': [770.41, 780.32], 'MgII': [2796.35, 2803.53],
}
MASK_HALFWIDTH_KMS = 100.0
METAL_HALFWIDTH_MIN = 50.0

Z_BINS = [(0.0, 0.075), (0.075, 0.13), (0.13, 0.19), (0.19, 0.26)]
MASK_LEVELS = ['sightline windows only', '+ intervening metals', '+ Galactic lines']
B_NARROW = 30.0


def velocity_offset(wave, centre):
    return np.abs(wave - centre) / centre * C_KMS


def masks_for(path, cos_meta):
    """The three nested pixel masks for one observed sightline, widest first."""
    name = Path(path).stem.replace('_all', '')
    wave, flux, err = obs.load_spectrum(path)
    info = cos_meta.get(name)
    if info is None:
        return None
    if info['windows']:
        good = obs.mask_from_windows(wave, err, info['windows'], info['z_em'])
    else:
        good = obs.usable_mask(wave, err, info['z_em'])
    good &= np.isfinite(flux) & (flux > -0.5) & (flux < 2.0)

    try:
        rows = vp.catalogue([str(Path(path).with_name(name + '.all'))])
    except (OSError, ValueError):
        rows = []
    no_metal = good.copy()
    for r in rows:
        if r['ion_key'] == 'HI':
            continue
        lams = METAL_REST.get(r['ion_key'])
        if lams is None:
            # An unidentified line sits where it was fitted, so mask it there.
            lams = [LYA] if r.get('unidentified') else []
        half = max(3.0 * r['b'], METAL_HALFWIDTH_MIN)
        for lam in lams:
            no_metal &= velocity_offset(wave, lam * (1 + r['z'])) > half

    no_mw = no_metal.copy()
    for line in obs.MW_LINES:
        no_mw &= velocity_offset(wave, line) > MASK_HALFWIDTH_KMS
    lo, hi = obs.MW_LYA_TROUGH
    no_mw &= (wave < lo) | (wave > hi)

    return name, wave, flux, [good, no_metal, no_mw]


def observed_tau(template_glob, cos_list, bootstrap=200, seed=0):
    """tau_eff of the observed sample per redshift bin, at each masking level.

    tau_eff is -ln of the mean flux over pixels, so it is summed over the sample rather
    than averaged per sightline. Errors are bootstrapped over sightlines, which is the
    unit that carries a continuum and a reduction.
    """
    meta = obs.load_cos_list(cos_list) if cos_list else {}
    per_sightline = []
    for path in sorted(glob.glob(template_glob)):
        got = masks_for(path, meta)
        if got is None:
            continue
        name, wave, flux, levels = got
        z = wave / LYA - 1.0
        entry = {'name': name, 'bins': {}}
        for li, m0 in enumerate(levels):
            for b in Z_BINS:
                m = m0 & (z >= b[0]) & (z < b[1])
                entry['bins'][(li, b)] = (float(flux[m].sum()), int(m.sum()),
                                          float(z[m].sum()))
        per_sightline.append(entry)

    rng = np.random.default_rng(seed)
    out = {}
    n_s = len(per_sightline)
    for li in range(len(MASK_LEVELS)):
        for b in Z_BINS:
            cells = [e['bins'][(li, b)] for e in per_sightline]
            s = sum(c[0] for c in cells)
            n = sum(c[1] for c in cells)
            zs = sum(c[2] for c in cells)
            if n == 0:
                continue
            draws = []
            for _ in range(bootstrap):
                pick = rng.integers(0, n_s, size=n_s)
                bs = sum(cells[i][0] for i in pick)
                bn = sum(cells[i][1] for i in pick)
                if bn:
                    draws.append(-np.log(bs / bn))
            out[(li, b)] = {'tau': -np.log(s / n), 'n_pix': n, 'z_mean': zs / n,
                            'err': float(np.std(draws)) if draws else np.nan}
    return out, per_sightline


def observed_by_wavelength(template_glob, cos_list, edges, level=2, bootstrap=400, seed=2):
    """tau_eff in fixed wavelength bins at one masking level, to localise the residual.

    Errors are bootstrapped over sightlines, so a bin high because one spectrum is odd
    carries a large error rather than a small one.
    """
    meta = obs.load_cos_list(cos_list) if cos_list else {}
    nb = len(edges) - 1
    per = []
    for path in sorted(glob.glob(template_glob)):
        got = masks_for(path, meta)
        if got is None:
            continue
        _, wave, flux, levels = got
        good = levels[level]
        si = np.zeros(nb)
        ni = np.zeros(nb)
        idx = np.digitize(wave[good], edges) - 1
        ok = (idx >= 0) & (idx < nb)
        np.add.at(si, idx[ok], flux[good][ok])
        np.add.at(ni, idx[ok], 1)
        per.append((si, ni))

    s = np.sum([c[0] for c in per], axis=0)
    n = np.sum([c[1] for c in per], axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = -np.log(s / n)
    rng = np.random.default_rng(seed)
    draws = np.empty((bootstrap, nb))
    for k in range(bootstrap):
        pick = rng.integers(0, len(per), size=len(per))
        bs = np.sum([per[i][0] for i in pick], axis=0)
        bn = np.sum([per[i][1] for i in pick], axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            draws[k] = -np.log(bs / bn)
    err = np.nanstd(draws, axis=0)
    return tau, n, err


def sim_flux_stats(analysis_root, snaps, key='effective_tau'):
    """One statistic from flux_stats.csv for every variant and snapshot that exists."""
    out = {}
    for sim in SIMS:
        for snap in snaps:
            p = Path(analysis_root) / 'IllustrisTNG' / 'EX' / sim / f'snap-{snap}' / 'flux_stats.csv'
            if not p.exists():
                continue
            for row in csv.reader(p.open()):
                if row and row[0] == key and row[1]:
                    out[(sim, snap)] = float(row[1])
                    break
    return out


def sim_line_widths(analysis_root, snaps):
    """Column densities and b values per variant and snapshot, from line_widths.csv."""
    out = {}
    for sim in SIMS:
        for snap in snaps:
            p = Path(analysis_root) / 'IllustrisTNG' / 'EX' / sim / f'snap-{snap}' / 'line_widths.csv'
            if not p.exists():
                continue
            d = np.genfromtxt(p, delimiter=',', names=True)
            out[(sim, snap)] = (d['N_HI'], d['b_param_km_s'])
    return out


def path_weighted(sim_tau, sim_key, z_pixels, snaps):
    """Simulated tau_eff averaged over where the observed pixels actually are.

    The simulation is a set of fixed redshifts; the observation is a path. Interpolating
    the simulated curve onto every observed pixel and averaging is what makes the two
    comparable. Redshifts past the last snapshot are clipped, so the result is a lower
    bound whenever the sample reaches beyond it.
    """
    zs = np.array([SNAP_Z[s] for s in snaps])
    order = np.argsort(zs)
    zs = zs[order]
    ts = np.array([sim_tau[(sim_key, s)] for s in snaps])[order]
    return float(np.interp(np.clip(z_pixels, zs[0], zs[-1]), zs, ts).mean())


def bootstrap_band(template_glob, cos_list, lo, hi, level=2, bootstrap=400, seed=1):
    """tau_eff in one wavelength band, with an error bootstrapped over sightlines."""
    meta = obs.load_cos_list(cos_list) if cos_list else {}
    cells = []
    for path in sorted(glob.glob(template_glob)):
        got = masks_for(path, meta)
        if got is None:
            continue
        _, wave, flux, levels = got
        m = levels[level] & (wave >= lo) & (wave < hi)
        cells.append((float(flux[m].sum()), int(m.sum())))
    n = sum(c[1] for c in cells)
    if n == 0:
        return np.nan, np.nan, 0
    tau = -np.log(sum(c[0] for c in cells) / n)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        pick = rng.integers(0, len(cells), size=len(cells))
        bs = sum(cells[i][0] for i in pick)
        bn = sum(cells[i][1] for i in pick)
        if bn:
            draws.append(-np.log(bs / bn))
    return tau, float(np.std(draws)), n


def observed_b_fraction(template_glob, max_b_err=10.0, max_logN_err=0.5):
    paths = [str(Path(p).with_name(Path(p).stem.replace('_all', '') + '.all'))
             for p in sorted(glob.glob(template_glob))]
    rows = vp.hi_rows(vp.catalogue([p for p in paths if Path(p).exists()]),
                      max_b_err=max_b_err, max_logN_err=max_logN_err)
    b = np.array([r['b'] for r in rows])
    return b, float((b < B_NARROW).mean())


def self_test():
    wave = np.linspace(1250.0, 1260.0, 400)
    assert np.isclose(velocity_offset(wave, wave[200]).min(), 0.0, atol=1e-6)
    # A 100 km/s offset at 1255 A is 0.42 A, so the mask half-width lands where expected.
    assert abs(velocity_offset(np.array([1255.0 + 0.4184]), 1255.0)[0] - 100.0) < 0.5

    # path_weighted on a flat simulated curve returns that value whatever the path is.
    flat = {('EX_0', '090'): 0.03, ('EX_0', '082'): 0.03}
    assert np.isclose(path_weighted(flat, 'EX_0', np.array([0.0, 0.1, 0.2]), ['090', '082']), 0.03)
    # On a rising curve it sits between the ends and tracks the path, not the midpoint.
    rising = {('EX_0', '090'): 0.02, ('EX_0', '082'): 0.04}
    low = path_weighted(rising, 'EX_0', np.full(100, 0.02), ['090', '082'])
    high = path_weighted(rising, 'EX_0', np.full(100, 0.19), ['090', '082'])
    assert low < 0.025 < high, (low, high)
    # Clipping past the last snapshot cannot extrapolate upward.
    assert np.isclose(path_weighted(rising, 'EX_0', np.full(10, 0.9), ['090', '082']), 0.04)

    print('obs_tau_breakdown self-test passed')


COLOURS = {'EX_0': '#1f77b4', 'EX_1': '#d62728', 'EX_2': '#2ca02c', 'EX_3': '#7f3f9f'}
MARKERS = ['o', 's', 'D']


def figure_tau_z(obs_tau, sim_tau, snaps, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    zs = sorted(SNAP_Z[s] for s in snaps)
    for sim, label in SIMS.items():
        y = [sim_tau[(sim, s)] for s in sorted(snaps, key=lambda s: SNAP_Z[s])]
        ax.plot(zs, y, '-', color=COLOURS[sim], lw=1.8, label=f'{label}', zorder=2)
    for li, level in enumerate(MASK_LEVELS):
        pts = [(v['z_mean'], v['tau'], v['err'])
               for (l, _), v in sorted(obs_tau.items(), key=lambda kv: kv[1]['z_mean'])
               if l == li]
        if not pts:
            continue
        z, t, e = zip(*pts)
        ax.errorbar(z, t, yerr=e, fmt=MARKERS[li], ms=6, mfc='none' if li < 2 else 'k',
                    color='k', alpha=0.45 + 0.275 * li, lw=1.2, capsize=3,
                    label=f'observed, {level}', zorder=3)
    ax.set_xlabel('redshift')
    ax.set_ylabel(r'$\tau_{\rm eff}$')
    ax.set_xlim(-0.01, 0.26)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_b_fraction(widths, snaps, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.3))
    ordered = sorted(snaps, key=lambda s: SNAP_Z[s])
    zs = [SNAP_Z[s] for s in ordered]
    for sim, label in SIMS.items():
        f = [(widths[(sim, s)][1] < B_NARROW).mean() for s in ordered]
        ax.plot(zs, f, 'o-', color=COLOURS[sim], lw=1.8, ms=5, label=label)
    ax.set_xlabel('redshift')
    ax.set_ylabel(r'fraction of lines with $b < 30$ km/s')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # The same variants' full b distributions at the snapshot nearest the observed median.
    near = min(snaps, key=lambda s: abs(SNAP_Z[s] - 0.1215))
    bins = np.linspace(0, 150, 46)
    for sim, label in SIMS.items():
        ax2.hist(widths[(sim, near)][1], bins=bins, histtype='step', density=True,
                 color=COLOURS[sim], lw=1.6, label=label)
    ax2.axvline(B_NARROW, color='k', ls=':', lw=1.0)
    ax2.set_xlabel(r'$b$ [km/s], noiseless estimator')
    ax2.set_ylabel('probability density')
    ax2.set_title(f'snapshot {near}, $z = {SNAP_Z[near]:.3f}$', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def figure_wavelength(tau_w, n_w, err_w, edges, sim_tau, snaps, out_path,
                      min_pix=2000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    centres = 0.5 * (edges[:-1] + edges[1:])
    z = centres / LYA - 1.0
    zs = np.array(sorted(SNAP_Z[s] for s in snaps))
    ts = np.array([sim_tau[('EX_0', s)] for s in sorted(snaps, key=lambda s: SNAP_Z[s])])
    ref = np.interp(np.clip(z, zs[0], zs[-1]), zs, ts)
    keep = n_w >= min_pix
    ratio = tau_w / ref

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.axhspan(0.9, 1.1, color='0.85', zorder=0)
    ax.axhline(1.0, color='k', lw=1.0, zorder=1)
    ax.axvspan(1270, 1345, color='#d62728', alpha=0.10, zorder=0)
    ax.errorbar(centres[keep], ratio[keep], yerr=(err_w / ref)[keep], fmt='o-',
                color='#1f77b4', ms=5, lw=1.4, capsize=3, zorder=2)
    ax.text(1307, ax.get_ylim()[1] * 0.97, r'1270--1345 $\AA$', ha='center', va='top',
            fontsize=9, color='#d62728')
    ax.set_xlabel(r'wavelength [$\AA$]')
    ax.set_ylabel(r'observed $\tau_{\rm eff}$ / fiducial simulation')
    ax.set_title('intervening metals and Galactic lines masked', fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _tabular(rows, fmt):
    return ' \\\\\n'.join(fmt % r for r in rows) + ' \\\\'


def write_tables(path, sim_tau, widths, snaps, obs_tau, ladder, wave_rows, obs_frac,
                 band_rows):
    ordered = sorted(snaps, key=lambda s: SNAP_Z[s], reverse=True)
    sims = list(SIMS)

    tau_rows = []
    for s in ordered:
        v = [sim_tau[(x, s)] for x in sims]
        tau_rows.append((s, SNAP_Z[s], *v, (max(v) - min(v)) / min(v) * 100))
    frac_rows = []
    for s in ordered:
        v = [(widths[(x, s)][1] < B_NARROW).mean() for x in sims]
        frac_rows.append((s, SNAP_Z[s], *v, (v[3] - v[0]) / v[0] * 100))
    count_rows = []
    for s in ordered:
        v = [len(widths[(x, s)][1]) / 10000 for x in sims]
        count_rows.append((s, SNAP_Z[s], *v))

    bin_rows = []
    for b in Z_BINS:
        cells = [obs_tau[(li, b)] for li in range(len(MASK_LEVELS)) if (li, b) in obs_tau]
        if len(cells) < len(MASK_LEVELS):
            continue
        zbar = cells[0]['z_mean']
        zz = np.array(sorted(SNAP_Z[s] for s in snaps))
        tt = np.array([sim_tau[('EX_0', s)] for s in sorted(snaps, key=lambda s: SNAP_Z[s])])
        fid = float(np.interp(min(zbar, zz[-1]), zz, tt))
        bin_rows.append((b[0], b[1], zbar, fid,
                         cells[0]['tau'], cells[0]['tau'] / fid,
                         cells[1]['tau'], cells[1]['tau'] / fid,
                         cells[2]['tau'], cells[2]['tau'] / fid))

    with open(path, 'w') as fh:
        fh.write('% Generated by scripts/obs_tau_breakdown.py -- do not edit by hand.\n')
        fh.write('\\newcommand{\\tauGrid}{%\n')
        fh.write(_tabular(tau_rows, '%s & %.4f & %.4f & %.4f & %.4f & %.4f & %.1f\\%%'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\fracGrid}{%\n')
        fh.write(_tabular(frac_rows, '%s & %.4f & %.3f & %.3f & %.3f & %.3f & $%+.1f$\\%%'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\countGrid}{%\n')
        fh.write(_tabular(count_rows, '%s & %.4f & %.3f & %.3f & %.3f & %.3f'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\ladderRows}{%\n')
        fh.write(_tabular(ladder, '%s & %.4f & %.4f & %.2f'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\binRows}{%\n')
        fh.write(_tabular(bin_rows,
                          '%.3f--%.3f & %.3f & %.4f & %.4f & %.2f & %.4f & %.2f & %.4f & %.2f'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\waveRows}{%\n')
        fh.write(_tabular(wave_rows, '%.0f--%.0f & %.3f & %d & %.4f & %.4f & %.2f'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\bandRows}{%\n')
        fh.write(_tabular(band_rows,
                          '%.0f--%.0f & %d & $%.4f \\pm %.4f$ & %.4f & %.2f & %.1f'))
        fh.write('\n}\n\n')
        fh.write('\\newcommand{\\obsNarrowFrac}{%.3f}\n' % obs_frac)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analysis-root', default='output/analysis',
                   help='directory holding IllustrisTNG/EX/<sim>/snap-<nnn>/')
    p.add_argument('--templates', default='data/Obs/*_all.dat')
    p.add_argument('--cos-list', default='data/Obs/cos.list')
    p.add_argument('--snaps', default='090,088,086,084,082')
    p.add_argument('--bootstrap', type=int, default=200)
    p.add_argument('--out', default='plots/obs/tau_breakdown')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return 0

    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    sim_tau = sim_flux_stats(args.analysis_root, snaps)
    missing = [(s, n) for s in SIMS for n in snaps if (s, n) not in sim_tau]
    if missing:
        p.error(f'no flux_stats.csv for {missing[:4]}{" ..." if len(missing) > 4 else ""}')
    widths = sim_line_widths(args.analysis_root, snaps)

    obs_tau, per_sightline = observed_tau(args.templates, args.cos_list, args.bootstrap)
    obs_b, obs_frac = observed_b_fraction(args.templates)

    # The observed path, at the strictest masking, is what the simulation is weighted onto.
    meta = obs.load_cos_list(args.cos_list)
    z_pixels = []
    for path in sorted(glob.glob(args.templates)):
        got = masks_for(path, meta)
        if got is not None:
            _, wave, _, levels = got
            z_pixels.append((wave / LYA - 1.0)[levels[2]])
    z_pixels = np.concatenate(z_pixels)
    in_range = z_pixels < 0.26

    ladder = []
    for sim, label in SIMS.items():
        w = path_weighted(sim_tau, sim, z_pixels[in_range], snaps)
        masked = sum(obs_tau[(2, b)]['tau'] * obs_tau[(2, b)]['n_pix'] for b in Z_BINS)
        npix = sum(obs_tau[(2, b)]['n_pix'] for b in Z_BINS)
        ladder.append((label, sim_tau[(sim, '090')], w, (masked / npix) / w))

    edges = np.arange(1225.0, 1500.0, 15.0)
    tau_w, n_w, err_w = observed_by_wavelength(args.templates, args.cos_list, edges)
    zz = np.array(sorted(SNAP_Z[s] for s in snaps))
    tt = np.array([sim_tau[('EX_0', s)] for s in sorted(snaps, key=lambda s: SNAP_Z[s])])
    wave_rows = []
    for i in range(len(edges) - 1):
        if n_w[i] < 2000:
            continue
        lam = 0.5 * (edges[i] + edges[i + 1])
        z = lam / LYA - 1.0
        ref = float(np.interp(min(z, zz[-1]), zz, tt))
        wave_rows.append((edges[i], edges[i + 1], z, int(n_w[i]), tau_w[i], ref,
                          tau_w[i] / ref))

    # The residual sits in a contiguous band, so test the band as a whole rather than
    # relying on any single bin, where the bootstrap error is of order the excess itself.
    bands = [(1225.0, 1270.0), (1270.0, 1345.0), (1345.0, 1500.0)]
    band_rows = []
    for lo, hi in bands:
        t, e, n = bootstrap_band(args.templates, args.cos_list, lo, hi)
        zbar = 0.5 * (lo + hi) / LYA - 1.0
        ref = float(np.interp(min(zbar, zz[-1]), zz, tt))
        band_rows.append((lo, hi, n, t, e, ref, t / ref, (t - ref) / e))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figure_tau_z(obs_tau, sim_tau, snaps, out / 'obs_tau_eff_z.png')
    figure_b_fraction(widths, snaps, out / 'obs_b_fraction.png')
    figure_wavelength(tau_w, n_w, err_w, edges, sim_tau, snaps,
                      out / 'obs_ratio_wavelength.png')
    write_tables(out / 'tables.tex', sim_tau, widths, snaps, obs_tau, ladder,
                 wave_rows, obs_frac, band_rows)

    print(f'{len(per_sightline)} observed sightlines, {len(obs_b)} well-measured '
          f'H I components, {(obs_b < B_NARROW).mean():.3f} narrower than {B_NARROW:.0f} km/s')
    print('\nobserved tau_eff per redshift bin')
    for b in Z_BINS:
        cells = [obs_tau.get((li, b)) for li in range(len(MASK_LEVELS))]
        if any(c is None for c in cells):
            continue
        print(f"  z {b[0]:.3f}-{b[1]:.3f}  " + '  '.join(
            f'{lv}: {c["tau"]:.4f}+-{c["err"]:.4f} ({c["n_pix"]} px)'
            for lv, c in zip(MASK_LEVELS, cells)))
    print('\ntau_eff ladder, observed with everything masked against each variant')
    for label, z0, w, ratio in ladder:
        print(f'  {label:14} z=0 only {z0:.4f}   path-weighted {w:.4f}   obs/sim {ratio:.2f}')
    print('\nresidual by wavelength band, everything masked')
    for lo, hi, n, t, e, ref, ratio, sig in band_rows:
        print(f'  {lo:.0f}-{hi:.0f} A  {n:6d} px  tau {t:.4f}+-{e:.4f}  '
              f'fiducial {ref:.4f}  ratio {ratio:.2f}  {sig:+.1f} sigma')
    print(f'\nwritten: {out}/obs_tau_eff_z.png, obs_b_fraction.png, '
          f'obs_ratio_wavelength.png, tables.tex')
    return 0


if __name__ == '__main__':
    sys.exit(main())
