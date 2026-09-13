"""
Matter power spectrum P(k) per variant, straight from the snapshots.

sigma_8 changes P(k) by a k-independent factor, so P_var / P_fid is flat;
Omega_0 also changes its shape. The ratio's logarithmic slope
d ln(P_var/P_fid) / d ln k over 0.3-5 h/Mpc is reported per variant. The box
(k_min = 0.25 h/Mpc) is far below k_eq, so this measures nonlinear growth, not
the linear turnover.

Per variant and snapshot: CIC-deposit the particles on an ngrid^3 mesh, FFT,
divide out the CIC window, subtract shot noise, and average in log-k bins.
Fields: 'dm' (PartType1) or 'total' (gas, DM, stars, black holes; mass-weighted).
Coordinates are converted from ckpc/h to cMpc/h, so k is in h/Mpc.

Run:
    python scripts/matter_pk_test.py \\
        --data-root data/IllustrisTNG/1P \\
        --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv \\
        --snaps 080,024 --ngrid 256 --out-dir plots/matter_pk_test
    python scripts/matter_pk_test.py --self-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from hypothesis_test_p1 import SCANS, FIDUCIAL, members, load_cosmo_table

FIELD_PARTS = {'dm': (1,), 'total': (0, 1, 4, 5)}
FIELD_LABEL = {'dm': 'DM', 'total': 'total matter'}


def _S8(omega0, sigma8):
    return sigma8 * np.sqrt(omega0 / 0.3)


# =====================================================================
# Power spectrum
# =====================================================================

def cic_deposit(pos, ngrid, boxsize, weights=None, out=None):
    """Cloud-in-cell assignment onto `out` (or a new grid). weights=None deposits
    particle counts."""
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64) if out is None else out
    x = (pos / boxsize) * ngrid                  # cell coordinates in [0, ngrid)
    i = np.floor(x).astype(np.int64)
    d = x - i                                    # offset within the cell
    i0 = i % ngrid
    i1 = (i + 1) % ngrid
    wx = [1.0 - d[:, 0], d[:, 0]]
    wy = [1.0 - d[:, 1], d[:, 1]]
    wz = [1.0 - d[:, 2], d[:, 2]]
    ix = [i0[:, 0], i1[:, 0]]
    iy = [i0[:, 1], i1[:, 1]]
    iz = [i0[:, 2], i1[:, 2]]
    gflat = grid.reshape(-1)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = wx[a] * wy[b] * wz[c]
                if weights is not None:
                    w = w * weights
                flat = (ix[a] * ngrid + iy[b]) * ngrid + iz[c]
                gflat += np.bincount(flat, weights=w, minlength=gflat.size)
    return grid


def deposit_chunks(chunks, ngrid, boxsize_mpc):
    """Deposit a list of (pos, weights) onto one grid, one particle type at a time.

    Returns (grid, sum w, sum w^2); the sums give the mean density and shot noise.
    """
    if len(chunks) > 1 and any(w is None for _, w in chunks):
        raise ValueError('multi-type field: every chunk needs masses')
    grid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
    wsum = w2sum = 0.0
    for pos, w in chunks:
        cic_deposit(pos, ngrid, boxsize_mpc, weights=w, out=grid)
        if w is None:
            wsum += pos.shape[0]
            w2sum += pos.shape[0]
        else:
            wsum += float(w.sum())
            w2sum += float((w.astype(np.float64) ** 2).sum())
    return grid, wsum, w2sum


def power_spectrum(pos, ngrid, boxsize_mpc, nkbins=40, weights=None):
    grid, wsum, w2sum = deposit_chunks([(pos, weights)], ngrid, boxsize_mpc)
    return power_spectrum_from_grid(grid, wsum, w2sum, ngrid, boxsize_mpc, nkbins)


def power_spectrum_from_grid(grid, wsum, w2sum, ngrid, boxsize_mpc, nkbins=40):
    """(k [h/Mpc], P(k) [(Mpc/h)^3], modes per bin), log bins from k_f to k_Ny."""
    delta = grid / (wsum / ngrid ** 3) - 1.0
    vol = boxsize_mpc ** 3
    pk3d = np.abs(np.fft.rfftn(delta)) ** 2 * vol / ngrid ** 6

    kf = 2.0 * np.pi / boxsize_mpc
    kx = np.fft.fftfreq(ngrid, d=1.0 / ngrid) * kf
    kz = np.fft.rfftfreq(ngrid, d=1.0 / ngrid) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    # CIC window W = prod_i sinc^2(pi k_i / 2 k_Ny); the measured P carries W^2.
    kny = np.pi * ngrid / boxsize_mpc
    # np.sinc(x) = sin(pi x) / (pi x)
    W = np.sinc(KX / (2.0 * kny)) * np.sinc(KY / (2.0 * kny)) * np.sinc(KZ / (2.0 * kny))
    W[W == 0] = 1.0
    pk3d /= W ** 4

    # Shot noise V sum(w^2) / (sum w)^2; equals V/N for equal masses.
    pk3d -= vol * w2sum / wsum ** 2

    bins = np.logspace(np.log10(kf), np.log10(kny), nkbins + 1)
    kflat, pflat = kmag.ravel(), pk3d.ravel()
    good = kflat > 0
    kflat, pflat = kflat[good], pflat[good]
    which = np.digitize(kflat, bins)
    kc, Pk, nm = [], [], []
    for b in range(1, nkbins + 1):
        m = which == b
        if m.any():
            kc.append(kflat[m].mean())
            Pk.append(pflat[m].mean())
            nm.append(int(m.sum()))
    return np.array(kc), np.array(Pk), np.array(nm)


# =====================================================================
# Snapshots
# =====================================================================

def read_field(snapshot, parts):
    """Return ([(coords, masses) per particle type], box [Mpc/h], z, {ptype: n}).

    masses is None only for a single equal-mass type (DM), whose mass is in the
    header MassTable.
    """
    chunks, counts = [], {}
    with h5py.File(snapshot, 'r') as f:
        box = float(f['Header'].attrs['BoxSize']) * 1e-3
        z = float(f['Header'].attrs['Redshift'])
        mtable = np.asarray(f['Header'].attrs['MassTable'], dtype=np.float64)
        for i in parts:
            key = f'PartType{i}'
            if key not in f or 'Coordinates' not in f[key]:
                continue
            pos = f[key]['Coordinates'][:].astype(np.float64) * 1e-3
            if pos.shape[0] == 0:
                continue
            pos = np.mod(pos, box)
            if mtable[i] > 0:
                mass = None if len(parts) == 1 else np.full(pos.shape[0], mtable[i])
            else:
                mass = f[key]['Masses'][:].astype(np.float64)
            chunks.append((pos, mass))
            counts[i] = pos.shape[0]
    return chunks, box, z, counts


def compute_snapshot(data_root, cosmo, scan, snapnum, ngrid, nkbins, field='dm'):
    """suffix -> {k, P, nmodes, label, omega0, sigma8, S8, z, param, counts}."""
    out = {}
    for suf in members(scan):
        label = SCANS[scan]['name_fmt'].format(s=suf)
        snap = data_root / label / f'snap_{snapnum}.hdf5'
        if not snap.exists():
            print(f'  [skip] {snap} missing')
            continue
        chunks, box, z, counts = read_field(snap, FIELD_PARTS[field])
        if not chunks:
            print(f'  [skip] {snap}: no particles for field {field}')
            continue
        grid, wsum, w2sum = deposit_chunks(chunks, ngrid, box)
        del chunks
        k, P, nm = power_spectrum_from_grid(grid, wsum, w2sum, ngrid, box, nkbins=nkbins)
        del grid
        om = float(cosmo.loc[label, 'Omega0']) if label in cosmo.index else np.nan
        s8 = float(cosmo.loc[label, 'sigma8']) if label in cosmo.index else np.nan
        col = SCANS[scan]['column']
        pv = float(cosmo.loc[label, col]) if col is not None and label in cosmo.index else np.nan
        out[suf] = {'k': k, 'P': P, 'nmodes': nm, 'label': label,
                    'omega0': om, 'sigma8': s8, 'S8': _S8(om, s8), 'z': z,
                    'param': pv, 'counts': counts}
        npart = ', '.join(f'PartType{i}={n}' for i, n in sorted(counts.items()))
        print(f'  {label} snap_{snapnum} [{field}]: z={z:.3f}, nk={len(k)}, '
              f'Omega0={om}, sigma8={s8}, {npart}')
    return out


def log_slope(k, ratio, klo=0.3, khi=5.0):
    """d ln(ratio) / d ln k over [klo, khi] h/Mpc."""
    m = (k >= klo) & (k <= khi) & np.isfinite(ratio) & (ratio > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(k[m]), np.log(ratio[m]), 1)[0])


# =====================================================================
# Figures. `results` maps scan -> compute_snapshot() output.
# =====================================================================

def _variant_label(scan, suf, r):
    """Parameter value, or the sim name for EX (no single parameter)."""
    if SCANS[scan]['column'] is None:
        return r.get('label', suf)
    return f"{suf} ({r['param']:.2f})"


def _results_z(results):
    for d in results.values():
        for r in d.values():
            return r['z']
    return np.nan


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def _scan_axes(n):
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.2), squeeze=False)
    return fig, axes[0]


def plot_snapshot(results, snapnum, out_dir, field='dm'):
    """P(k) and P_var/P_fid figures; returns scan -> suffix -> log slope."""
    flabel = FIELD_LABEL[field]
    z = _results_z(results)

    fig, axes = _scan_axes(len(results))
    for ax, (scan, d) in zip(axes, results.items()):
        mem = members(scan)
        for suf, c in zip(mem, plt.cm.viridis(np.linspace(0, 0.9, len(mem)))):
            if suf not in d:
                continue
            r = d[suf]
            m = r['P'] > 0
            ax.loglog(r['k'][m], r['P'][m], '-', color=c, lw=1.6,
                      label=_variant_label(scan, suf, r))
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P(k)$ [(Mpc/$h$)$^3$]')
        ax.set_title(f'{flabel} $P(k)$ -- {scan} ({SCANS[scan]["label"]})')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    fig.suptitle(f'snap_{snapnum}  (z={z:.2f}) -- {flabel}')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_snap_{snapnum}.png')

    fig, axes = _scan_axes(len(results))
    slopes = {}
    for ax, (scan, d) in zip(axes, results.items()):
        if FIDUCIAL not in d:
            continue
        kf_, Pf = d[FIDUCIAL]['k'], d[FIDUCIAL]['P']
        mem = members(scan)
        for suf, c in zip(mem, plt.cm.viridis(np.linspace(0, 0.9, len(mem)))):
            if suf not in d:
                continue
            r = d[suf]
            ratio = r['P'] / np.interp(r['k'], kf_, Pf)
            ax.semilogx(r['k'], ratio, '-', color=c, lw=1.6, label=_variant_label(scan, suf, r))
            slopes.setdefault(scan, {})[suf] = log_slope(r['k'], ratio)
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P_{\rm var}(k)/P_{\rm fid}(k)$')
        sub = ', '.join(f'{s}={v:+.2f}' for s, v in slopes.get(scan, {}).items())
        ax.set_title(f'{scan} ({SCANS[scan]["label"]})\n'
                     rf'$d\ln(P_{{\rm var}}/P_{{\rm fid}})/d\ln k$: {sub}', fontsize=9)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    fig.suptitle(f'P(k) relative to fiducial, snap_{snapnum} (z={z:.2f}, {flabel})')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_ratio_snap_{snapnum}.png')
    return slopes


def plot_field_ratio(res_total, res_dm, snapnum, out_dir):
    """P_total(k) / P_DM(k) per variant; feedback lowers it at high k."""
    fig, axes = _scan_axes(len(res_total))
    out = {}
    for ax, scan in zip(axes, res_total):
        mem = members(scan)
        for suf, c in zip(mem, plt.cm.viridis(np.linspace(0, 0.9, len(mem)))):
            if suf not in res_total[scan] or suf not in res_dm.get(scan, {}):
                continue
            rt, rd = res_total[scan][suf], res_dm[scan][suf]
            ratio = rt['P'] / np.interp(rt['k'], rd['k'], rd['P'])
            ax.semilogx(rt['k'], ratio, '-', color=c, lw=1.6, label=_variant_label(scan, suf, rt))
            out.setdefault(scan, {})[suf] = log_slope(rt['k'], ratio)
        ax.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$k$ [$h$/Mpc]')
        ax.set_ylabel(r'$P_{\rm total}(k)/P_{\rm DM}(k)$')
        ax.set_title(f'effect of baryons -- {scan} ({SCANS[scan]["label"]})')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8, title='variant')
    fig.suptitle(f'snap_{snapnum}: total matter vs DM only')
    fig.tight_layout()
    _save(fig, out_dir / f'Pk_total_over_dm_snap_{snapnum}.png')
    return out


# =====================================================================

def self_test():
    rng = np.random.default_rng(0)
    ngrid, box, n = 32, 100.0, 5000
    pos = rng.random((n, 3)) * box

    # equal weights of any size give the same P(k) as unweighted counts
    k0, P0, nm0 = power_spectrum(pos, ngrid, box, nkbins=12)
    k1, P1, nm1 = power_spectrum(pos, ngrid, box, nkbins=12, weights=np.full(n, 3.7))
    assert np.allclose(k0, k1) and np.array_equal(nm0, nm1)
    assert np.allclose(P0, P1, rtol=1e-8, atol=1e-8 * np.abs(P0).max()), np.abs(P0 - P1).max()

    # weighted shot noise reduces to V/N for equal weights
    _, wsum, w2sum = deposit_chunks([(pos, np.full(n, 3.7))], ngrid, box)
    assert np.isclose(box ** 3 * w2sum / wsum ** 2, box ** 3 / n, rtol=1e-12)

    # splitting particles into chunks changes neither the grid nor the sums
    g1, ws, w2s = deposit_chunks([(pos, np.ones(n))], ngrid, box)
    g2, ws2, w2s2 = deposit_chunks([(pos[:1000], np.ones(1000)),
                                    (pos[1000:], np.ones(n - 1000))], ngrid, box)
    assert (ws, w2s) == (ws2, w2s2) == (float(n), float(n))
    assert np.allclose(g1, g2)

    # counts and masses must not be mixed in one field
    try:
        deposit_chunks([(pos, None), (pos, np.ones(n))], ngrid, box)
    except ValueError:
        pass
    else:
        raise AssertionError('mixed weighted/unweighted chunks must raise')

    assert abs(log_slope(np.array([0.5, 1.0, 2.0]), np.array([0.5, 1.0, 2.0]) ** 0.3) - 0.3) < 1e-12
    print('self-test OK')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path,
                    help='directory holding <sim>/snap_NNN.hdf5, e.g. data/IllustrisTNG/1P')
    ap.add_argument('--cosmo-csv', type=Path, nargs='+', help='CosmoAstroSeed table(s)')
    ap.add_argument('--snaps', default='080', help='comma-separated, e.g. 080,024')
    ap.add_argument('--ngrid', type=int, default=256)
    ap.add_argument('--nkbins', type=int, default=40)
    ap.add_argument('--out-dir', type=Path, default=Path('plots/matter_pk_test'))
    ap.add_argument('--scans', default='p1,p2')
    ap.add_argument('--field', default='dm', choices=['dm', 'total', 'both'])
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.data_root is None or args.cosmo_csv is None:
        ap.error('--data-root and --cosmo-csv are required unless --self-test')

    scans = [s.strip() for s in args.scans.split(',') if s.strip()]
    unknown = [s for s in scans if s not in SCANS]
    if unknown:
        ap.error(f'unknown scan(s) {unknown}; known: {list(SCANS)}')
    fields = ['dm', 'total'] if args.field == 'both' else [args.field]

    plt.rcParams['figure.dpi'] = 150
    cosmo = pd.concat([load_cosmo_table(c) for c in args.cosmo_csv])
    snaps = [s.strip() for s in args.snaps.split(',') if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for snapnum in snaps:
        per_field = {}
        for field in fields:
            print(f'\n=== matter P(k), snap_{snapnum}, field={field} ===')
            results = {scan: compute_snapshot(args.data_root, cosmo, scan, snapnum,
                                              args.ngrid, args.nkbins, field=field)
                       for scan in scans}
            if not any(results.values()):
                print('  no data for this snap, skipping plots')
                continue
            fdir = args.out_dir / field if len(fields) > 1 else args.out_dir
            per_field[field] = {'results': results,
                                'slopes': plot_snapshot(results, snapnum, fdir, field=field)}
        if not per_field:
            continue
        entry = {'z': _results_z(next(iter(per_field.values()))['results']),
                 'log_slopes': {f: v['slopes'] for f, v in per_field.items()}}
        if 'total' in per_field and 'dm' in per_field:
            entry['total_over_dm_log_slope'] = plot_field_ratio(
                per_field['total']['results'], per_field['dm']['results'], snapnum, args.out_dir)
        summary[snapnum] = entry

    with open(args.out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f'\nAll outputs under {args.out_dir.resolve()}')


if __name__ == '__main__':
    sys.exit(main())
