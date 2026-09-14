"""
Compare raw CAMELS snapshots field by field, to find why one gives unphysical forest statistics.

The spectra use the stored NeutralHydrogenAbundance, so the key number is the HI photoionisation
rate implied by diffuse gas. In ionisation equilibrium x_HI ∝ Δ T^-0.7 / Γ_HI, so
median(Δ T^-0.7 / x_HI) ∝ Γ_HI. The same offset in every index block points to the ionisation
state written into the snapshot; an offset in only some blocks points to corrupt data.
Stored bytes per particle, per dataset, show which fields make a file larger than a normal seed's.

Run (first file is the reference):
    python scripts/check_snapshot.py REF.hdf5 OTHER.hdf5 [...]
    python scripts/check_snapshot.py --self-test
"""

import argparse
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

X_H = 0.76
FIELDS = ('Density', 'InternalEnergy', 'ElectronAbundance', 'NeutralHydrogenAbundance', 'Masses')
N_BLOCKS = 16


def temperature(u, xe):
    """K, from InternalEnergy [(km/s)^2] and ElectronAbundance (TNG convention)."""
    mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe) * 1.67262e-24
    return (2.0 / 3.0) * u * 1e10 * mu / 1.380649e-16


def load(path):
    with h5py.File(path, 'r') as f:
        header = dict(f['Header'].attrs)
        keys = set(f['PartType0'])
        gas = {k: f['PartType0'][k][:].astype(np.float64) for k in FIELDS}
    gas['T'] = temperature(gas['InternalEnergy'], gas['ElectronAbundance'])
    gas['Delta'] = gas['Density'] * header['BoxSize'] ** 3 / gas['Masses'].sum()
    return header, keys, gas


def storage(path):
    """Stored (compressed) bytes per particle for every PartType dataset."""
    with h5py.File(path, 'r') as f:
        return {f'{pt}/{k}': d.id.get_storage_size() / d.shape[0]
                for pt in f if pt.startswith('PartType')
                for k, d in f[pt].items()
                if isinstance(d, h5py.Dataset) and d.shape and d.shape[0]}


def summarise(gas):
    x, T, delta = gas['NeutralHydrogenAbundance'], gas['T'], gas['Delta']
    idx = np.flatnonzero((delta > 0.5) & (delta < 2) & (T > 0) & (T < 1e5) & (x > 0))
    gamma = lambda i: float(np.median(delta[i] * T[i] ** -0.7 / x[i]))
    return {
        'n': x.size,
        'nonfinite': {k: int((~np.isfinite(v)).sum()) for k, v in gas.items()
                      if not np.all(np.isfinite(v))},
        'pct': {k: np.percentile(gas[k], (1, 50, 99))
                for k in ('Delta', 'T', 'ElectronAbundance', 'NeutralHydrogenAbundance')},
        'hi_frac': float((gas['Masses'] * x).sum() / gas['Masses'].sum()),
        'diffuse_frac': idx.size / x.size,
        'diffuse_T': float(np.median(T[idx])),
        'gamma': gamma(idx),
        'gamma_blocks': np.array([gamma(b) for b in np.array_split(idx, N_BLOCKS)]),
    }


def header_diff(a, b):
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b))
            if k not in a or k not in b or not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))}


def report(paths):
    ref = None
    for p in paths:
        header, keys, gas = load(p)
        s = summarise(gas)
        del gas
        sizes = storage(p)
        print(f"\n=== {p}  (z = {header['Redshift']:.4f}, {s['n']:,} gas particles)")
        if ref is None:
            ref = header, keys, s, sizes
            print('  reference')
        else:
            for k, (a, b) in header_diff(ref[0], header).items():
                print(f'  header {k}: ref {a}  this {b}')
            if keys != ref[1]:
                print(f'  PartType0 fields missing {sorted(ref[1] - keys)}, extra {sorted(keys - ref[1])}')
            dev = {k: v / ref[3][k] - 1 for k, v in sizes.items() if ref[3].get(k)}
            for k in sorted(dev, key=lambda k: -abs(dev[k])):
                if abs(dev[k]) > 0.02:
                    print(f'  stored bytes/particle {k}: {dev[k]:+.1%} vs ref')
        print(f"  non-finite: {s['nonfinite'] or 'none'}")
        for k, v in s['pct'].items():
            print(f'  {k:25s} p1 {v[0]:.3e}  p50 {v[1]:.3e}  p99 {v[2]:.3e}')
        r = ref[2]
        blocks = s['gamma_blocks'] / r['gamma']
        print(f"  HI / H mass              {s['hi_frac']:.3e}  ({s['hi_frac'] / r['hi_frac']:.3f} x ref)")
        print(f"  diffuse gas              {s['diffuse_frac']:.1%} of particles, median T {s['diffuse_T']:.0f} K")
        print(f"  Gamma_HI / ref           {s['gamma'] / r['gamma']:.3f}  "
              f"(index blocks {blocks.min():.3f} to {blocks.max():.3f})")


def self_test():
    assert abs(temperature(210.5, 1.158) / 1e4 - 1) < 0.01   # fully ionised, mu = 0.588

    rng = np.random.default_rng(1)
    n = 200_000
    delta = 10 ** rng.uniform(-0.5, 0.5, n)
    u = 10 ** rng.uniform(1.8, 2.8, n)
    xe = np.full(n, 1.158)
    x = 1e-5 * delta * temperature(u, xe) ** -0.7
    half = x.copy()
    half[: n // 2] /= 2

    with tempfile.TemporaryDirectory() as d:
        stats, sizes = [], []
        for i, xhi in enumerate((x, x / 2, half)):
            path = Path(d) / f'{i}.hdf5'
            with h5py.File(path, 'w') as f:
                f.create_group('Header').attrs.update({'BoxSize': 1.0, 'Redshift': 4.0})
                g = f.create_group('PartType0')
                for k, v in zip(FIELDS, (delta, u, xe, xhi, np.full(n, 1.0 / n))):
                    g[k] = v
            stats.append(summarise(load(path)[2]))
            sizes.append(storage(path))

    assert abs(stats[1]['gamma'] / stats[0]['gamma'] - 2) < 1e-9      # global offset
    b = stats[2]['gamma_blocks'] / stats[0]['gamma']
    assert b.min() < 1.2 and b.max() > 1.8                             # offset in some blocks only
    assert set(sizes[0]) == {f'PartType0/{k}' for k in FIELDS} and sizes[1] == sizes[0]
    print('self-test OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('snapshots', nargs='*', type=Path, help='first one is the reference')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if len(args.snapshots) < 2:
        ap.error('need a reference snapshot and at least one to compare')
    report(args.snapshots)
    return 0


if __name__ == '__main__':
    sys.exit(main())
