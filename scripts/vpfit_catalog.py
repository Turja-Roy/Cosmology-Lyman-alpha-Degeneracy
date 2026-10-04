"""
Read the collaborator's VPFIT fits (`.all`, fort.26 format) into one absorber catalogue.

Each `.all` file holds the Voigt decomposition of one sightline: every fitted component with
its ion, redshift, Doppler parameter and column density, grouped into fitting regions that
name their continuum file and their COS line-spread function. That is three things the
analysis could not produce for itself:

* **An H I b-N catalogue** measured the way observers measure it, to compare against
  simulated absorbers run through the same detection.
* **Identified metals**, so intervening absorption can be masked by line rather than by the
  generic Galactic list in observations.MW_LINES.
* **The LSF file per region** (`COS_g160m_L1611_LF1.dat`), naming grating and lifetime
  position, which the forward model needs for a tabulated kernel.

Row format is ion then seven numbers: z, z_err, b, b_err, log N, log N_err, flag. VPFIT
appends tie characters to fitted values (`0.283456SZ`), which are stripped and kept. Rows
whose "ion" is `>>` or `<<` are continuum and zero-level nuisance parameters, not absorbers,
and are dropped; `??` marks an unidentified line and is kept as such.

The observed files are never modified; everything is written under output/.

    python scripts/vpfit_catalog.py 'data/Obs/*.all'
    python scripts/vpfit_catalog.py --self-test
"""

import argparse
import csv
import glob
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# VPFIT writes a fitted value as number + tie/fix letters, e.g. "0.283456SZ" or "1.000SC".
VALUE_RE = re.compile(r'^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)([A-Za-z%]*)$')
NUISANCE = {'>>', '<<'}
UNIDENTIFIED = {'??'}
N_FIELDS = 7      # z, z_err, b, b_err, logN, logN_err, flag


def _value(token):
    """Numeric part of a VPFIT field, plus whatever tie characters follow it."""
    match = VALUE_RE.match(token)
    if not match:
        return None, ''
    return float(match.group(1)), match.group(2)


def parse_all(path):
    """One `.all` file into (regions, components).

    A region is a `%%` line: continuum file, fitted wavelength range and LSF file. A component
    is an absorber row. Each component carries the region that was open when it was read, so
    its LSF is known.
    """
    regions, components = [], []
    current = None

    for raw in Path(path).read_text(errors='replace').splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue

        if line.lstrip().startswith('%%'):
            parts = line.lstrip('% ').split()
            lsf = next((p.split('=', 1)[1] for p in parts if p.lower().startswith('wdlsf=')), '')
            waves = [v for v, _ in (_value(p) for p in parts[1:]) if v is not None]
            current = {
                'continuum': parts[0] if parts else '',
                'wave_lo': waves[1] if len(waves) > 2 else float('nan'),
                'wave_hi': waves[2] if len(waves) > 2 else float('nan'),
                'lsf': lsf,
            }
            regions.append(current)
            continue

        if line.lstrip().startswith('!'):
            continue                      # "! Stats:" lines

        tokens = line.split()
        if tokens and tokens[-1] == '!':
            tokens = tokens[:-1]
        if len(tokens) < N_FIELDS + 1:
            continue

        values, ties = [], []
        for token in tokens[-N_FIELDS:]:
            value, tie = _value(token)
            if value is None:
                break
            values.append(value)
            ties.append(tie)
        if len(values) != N_FIELDS:
            continue

        ion = ' '.join(tokens[:-N_FIELDS])
        if ion in NUISANCE:
            continue                      # continuum / zero-level parameters

        z, z_err, b, b_err, logN, logN_err, flag = values
        components.append({
            'ion': ion,
            'z': z, 'z_err': z_err,
            'b': b, 'b_err': b_err,
            'logN': logN, 'logN_err': logN_err,
            'flag': flag,
            'unidentified': ion in UNIDENTIFIED,
            'tied': any(ties),
            'region_lo': current['wave_lo'] if current else float('nan'),
            'region_hi': current['wave_hi'] if current else float('nan'),
            'lsf': current['lsf'] if current else '',
        })
    return regions, components


def normalise_ion(ion):
    """'SiIII' and 'Si III' are the same ion written two ways."""
    return re.sub(r'\s+', '', ion)


def catalogue(paths):
    """Every component from every sightline, tagged with its sightline name."""
    rows = []
    for path in paths:
        name = Path(path).stem
        _, components = parse_all(path)
        for c in components:
            rows.append({'sightline': name, **c, 'ion_key': normalise_ion(c['ion'])})
    return rows


def hi_rows(rows, max_b_err=None, max_logN_err=None):
    """H I components only, optionally cut on how well they are constrained.

    Many fitted components have b or log N errors larger than the value itself, which happens
    where lines are blended or saturated. Those are real entries in the fit but carry no
    measurement, so any b-N comparison has to say which cut it used.
    """
    out = [r for r in rows if r['ion_key'] == 'HI']
    if max_b_err is not None:
        out = [r for r in out if r['b_err'] <= max_b_err]
    if max_logN_err is not None:
        out = [r for r in out if r['logN_err'] <= max_logN_err]
    return out


def write_csv(rows, path):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = ['sightline', 'ion', 'ion_key', 'z', 'z_err', 'b', 'b_err', 'logN', 'logN_err',
              'flag', 'unidentified', 'tied', 'region_lo', 'region_hi', 'lsf']
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


SAMPLE = """%% pg1048_g160_cont.dat     1   1547.3015   1549.0624 wdLSF=COS_g160m_L1611_LF1.dat  !  0.740077    48 2025/06/04
! Stats:  25     1.2022150   93   87  0.096  0  AICc:     117.57
 C IV    -0.000078    0.000007    34.69      3.31  14.278    0.028  0 !
 >>       0.283456SZ  0.000000    -3.61      3.68   1.000SC  0.000  0 !
%% pg1048_g130_cont.dat     1   1221.3262   1224.3676 wdLSF=COS_g130m_L1291_LF1.dat  !  0.000000   102 2025/06/04
 H I      0.005284    0.000018    14.68     50.59  15.928   16.431  0 !
 H I      0.143167    0.000006    28.71      2.96  13.850    0.032  0 !
 SiIII    0.105976    0.000028     9.75      1.13  12.355    0.124  0 !
 ??       0.060000    0.000100     20.00     5.00  13.000    0.100  0 !
"""


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'sample.all'
        path.write_text(SAMPLE)
        regions, components = parse_all(path)

        assert len(regions) == 2, regions
        assert regions[0]['lsf'] == 'COS_g160m_L1611_LF1.dat'
        assert abs(regions[0]['wave_lo'] - 1547.3015) < 1e-6
        assert regions[1]['continuum'] == 'pg1048_g130_cont.dat'

        # The >> nuisance row is dropped; the unidentified line is kept and marked.
        assert [c['ion'] for c in components] == ['C IV', 'H I', 'H I', 'SiIII', '??'], \
            [c['ion'] for c in components]
        assert components[-1]['unidentified'] and not components[0]['unidentified']

        civ = components[0]
        assert abs(civ['z'] + 0.000078) < 1e-9 and abs(civ['b'] - 34.69) < 1e-9
        assert abs(civ['logN'] - 14.278) < 1e-9
        assert civ['lsf'] == 'COS_g160m_L1611_LF1.dat'

        # Components inherit the region that was open when they were read.
        hi = [c for c in components if c['ion'] == 'H I']
        assert all(c['lsf'] == 'COS_g130m_L1291_LF1.dat' for c in hi), hi

        rows = catalogue([path])
        assert all(r['sightline'] == 'sample' for r in rows)
        assert normalise_ion('Si III') == normalise_ion('SiIII') == 'SiIII'

        # The error cut keeps the well-measured line and drops the blended one.
        assert len(hi_rows(rows)) == 2
        kept = hi_rows(rows, max_b_err=10.0, max_logN_err=0.5)
        assert len(kept) == 1 and abs(kept[0]['b'] - 28.71) < 1e-9, kept

        out = Path(tmp) / 'cat.csv'
        write_csv(rows, out)
        assert out.read_text().count('\n') == len(rows) + 1

    print('vpfit_catalog self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help="VPFIT .all files or a glob")
    parser.add_argument('--out', default='output/analysis/Obs/vpfit_components.csv')
    parser.add_argument('--max-b-err', type=float, default=10.0,
                        help='b error cut for the H I summary (default 10 km/s)')
    parser.add_argument('--max-logN-err', type=float, default=0.5,
                        help='log N error cut for the H I summary (default 0.5 dex)')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    paths = sorted(p for pattern in args.files for p in glob.glob(pattern)) or args.files
    if not paths:
        parser.error('need at least one .all file (or --self-test)')

    rows = catalogue(paths)
    if not rows:
        print('no components parsed')
        return 1
    write_csv(rows, args.out)

    ions = Counter(r['ion_key'] for r in rows)
    lsfs = Counter(r['lsf'] for r in rows if r['lsf'])
    print(f"{len(paths)} sightlines, {len(rows)} components")
    print(f"written: {args.out}\n")

    print('most common ions:')
    for ion, n in ions.most_common(12):
        print(f'  {ion:<10}{n:>6}')

    print('\nLSF kernels referenced:')
    for lsf, n in lsfs.most_common():
        print(f'  {lsf:<34}{n:>6}')

    all_hi = hi_rows(rows)
    good = hi_rows(rows, args.max_b_err, args.max_logN_err)
    print(f'\nH I components: {len(all_hi)} total, {len(good)} with '
          f'b_err <= {args.max_b_err} km/s and logN_err <= {args.max_logN_err} dex')
    if good:
        b = np.array([r['b'] for r in good])
        logN = np.array([r['logN'] for r in good])
        z = np.array([r['z'] for r in good])
        print(f'  b     median {np.median(b):.1f} km/s, 16-84% '
              f'{np.percentile(b, 16):.1f}-{np.percentile(b, 84):.1f}')
        print(f'  log N median {np.median(logN):.2f}, 16-84% '
              f'{np.percentile(logN, 16):.2f}-{np.percentile(logN, 84):.2f}')
        print(f'  z     {z.min():.4f} to {z.max():.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
