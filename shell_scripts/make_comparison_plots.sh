#!/bin/bash
# Regenerate the cross-simulation figures from the per-snapshot CSVs.
# Run from the repo root after syncing /work:
#
#   bash shell_scripts/make_comparison_plots.sh
set -euo pipefail

ROOT=output/analysis/IllustrisTNG/1P
COSMO=data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv
SNAPS=snap-024,snap-028,snap-032,snap-038,snap-044,snap-050,snap-060,snap-072,snap-080,snap-090
PY=${PY:-.venv/bin/python}

# Stop if any p1/p2 variant is missing: the plots would silently use fewer points.
missing=0
for sim in 1P_p1_n2 1P_p1_n1 1P_p1_0 1P_p1_1 1P_p1_2 \
           1P_p2_n2 1P_p2_n1 1P_p2_0 1P_p2_1 1P_p2_2; do
  for snap in ${SNAPS//,/ }; do
    f="$ROOT/$sim/$snap/flux_stats.csv"
    [ -f "$f" ] || { echo "MISSING: $f"; missing=$((missing + 1)); }
  done
done
if [ "$missing" -ne 0 ]; then
  echo "$missing of 100 snapshot dirs missing. Finish the cluster jobs or sync /work first."
  exit 1
fi
echo "1P complete: 100/100"

# Old snap-014/018 figures would otherwise sit next to the new ones.
rm -rf plots/hypothesis_p1_test plots/degeneracy_test

echo; echo "=== [1/5] flux / tau PDF evolution ==="
$PY scripts/pdf_evolution.py --analysis-root "$ROOT" --snaps "$SNAPS" \
    --scans p1,p2 --out-dir plots/pdf_evolution

echo; echo "=== [2/5] hypothesis tests (p1, p2) ==="
$PY scripts/hypothesis_test_p1.py --analysis-root "$ROOT" --cosmo-csv "$COSMO" \
    --snaps "$SNAPS" --scans p1,p2 --out-dir plots/hypothesis_p1_test

echo; echo "=== [3/5] Omega_0 - sigma_8 degeneracy ==="
$PY scripts/degeneracy_test.py --analysis-root "$ROOT" --cosmo-csv "$COSMO" \
    --snaps "$SNAPS" --out-dir plots/degeneracy_test

echo; echo "=== [4/5] multi-line plots ==="
$PY scripts/replot.py --root "$ROOT" --out-dir plots/IllustrisTNG/1P \
    --pattern '1P_p[12]_*' --only multi_line_comparison --workers 8

# Last, so a missing CV sync does not block the plots above.
echo; echo "=== [5/5] cosmic variance (CV set) ==="
$PY scripts/cosmic_variance.py --p1-root "$ROOT" --cosmo-csv "$COSMO" \
    --cv-root output/analysis/IllustrisTNG/CV --ex-root output/analysis/IllustrisTNG/EX \
    --snaps "$SNAPS" --out-dir plots/cosmic_variance

echo; echo "Done. Start with:"
for s in p1 p2; do
  echo "  plots/hypothesis_p1_test/$s/across_snapshots/delta_tau_eff_vs_z.json"
done
echo "  plots/cosmic_variance/cosmic_variance.json"
