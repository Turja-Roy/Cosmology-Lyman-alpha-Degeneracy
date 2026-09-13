#!/bin/bash
#SBATCH -J cv_set                  # Job name
#SBATCH -o logs/cv_%A_%a.out       # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/cv_%A_%a.err       # Error file
#SBATCH -p normal                  # Partition/queue (Lonestar6)
#SBATCH -N 1                       # One whole node per task
#SBATCH -n 48                      # Cores
#SBATCH -t 04:00:00                # Time limit
#SBATCH --array=1-27               # 27 CV sims, one snapshot per submission
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=turja.roy@uta.edu

# CV set (CV_0..CV_26): for each snapshot, download all 27, generate + analyze,
# check the output, delete the snapshot. One snapshot at a time because
# 27 x 2.5 GB is what fits on /work next to the 1P and EX data.
#
# Login node (nohup keeps it running after logout):
#   nohup bash shell_scripts/cv_set.sh > logs/cv_set.log 2>&1 &
#   bash shell_scripts/cv_set.sh run 032 044          # only these snapshots
#   bash shell_scripts/cv_set.sh download 032
#   bash shell_scripts/cv_set.sh clean 032 [--dry-run]
#   bash shell_scripts/cv_set.sh status
#   bash shell_scripts/cv_set.sh selftest
#
# Re-running skips snapshots already analysed for all 27 sims.
# SLURM runs this same file as the array worker.

N_SIMS=27
N_SIGHTLINES=10000
SIGHTLINES_FILE="output/sightlines/analysis_sightlines_2025-01-28.hdf5"
LINES="lya,lya_h"
ALL_SNAPS=(024 028 032 038 044 050 060 072 080 090)

DATA_ROOT="data/IllustrisTNG/CV"
OUT_ROOT="output/analysis/IllustrisTNG/CV"
SPECTRA_ROOT="spectra/IllustrisTNG/CV"

pad()      { printf '%03d' "$((10#$1))"; }
snap_file(){ echo "${DATA_ROOT}/CV_$1/snap_$2.hdf5"; }      # <sim idx> <padded snap>
out_dir()  { echo "${OUT_ROOT}/CV_$1/snap-$2"; }            # <sim idx> <padded snap>

# Number of sims with non-empty analysis output for a snapshot.
analysed_count() {
    local snap=$1 n=0 i d
    for i in $(seq 0 $((N_SIMS - 1))); do
        d=$(out_dir "$i" "$snap")
        [ -n "$(ls -A "$d" 2>/dev/null)" ] && n=$((n + 1))
    done
    echo "$n"
}

# True if the snapshot is fully downloaded: file size >= the end-of-file address
# stored in the HDF5 superblock (bytes 40-47 for superblock version 0, which all
# CAMELS snapshots use; other versions are not checked).
staged() {                                                  # <sim idx> <padded snap>
    local f; f=$(snap_file "$1" "$2")
    [ -f "$f" ] || return 1
    [ "$(dd if="$f" bs=1 count=8 2>/dev/null | od -An -tx1 | tr -d ' \n')" \
        = "894844460d0a1a0a" ] || return 1
    [ "$(dd if="$f" bs=1 skip=8 count=1 2>/dev/null | od -An -tu1 | tr -d ' \n')" \
        = "0" ] || return 0
    local eof; eof=$(dd if="$f" bs=8 skip=5 count=1 2>/dev/null | od -An -tu8 | tr -d ' \n')
    [ -n "$eof" ] && [ "$(stat -c %s "$f")" -ge "$eof" ]
}

staged_count() {
    local snap=$1 n=0 i
    for i in $(seq 0 $((N_SIMS - 1))); do
        staged "$i" "$snap" && n=$((n + 1))
    done
    echo "$n"
}


# =====================================================================
# Array worker (compute node): generate + analyze one sim at one snapshot
# =====================================================================
worker() {
    if [ -z "${SNAP:-}" ]; then
        echo "ERROR: SNAP not set. Submit with: sbatch --export=ALL,SNAP=032 shell_scripts/cv_set.sh"
        exit 1
    fi
    local snap; snap=$(pad "$SNAP")
    local idx=$((SLURM_ARRAY_TASK_ID - 1))
    local sim="CV_${idx}"

    source shell_scripts/timing.sh
    timing_context "$sim" "$snap"
    echo "Task $SLURM_ARRAY_TASK_ID: $sim snap_$snap on $(hostname), $(date)"

    module load cmake
    module load gcc/13.2.0
    module load impi/19.0.9
    module load eigen/3.4.0
    module load fftw3/3.3.10
    source .venv/bin/activate || { echo "ERROR: venv activation failed"; exit 1; }

    local snapshot out spectra link rc
    snapshot=$(snap_file "$idx" "$snap")
    out=$(out_dir "$idx" "$snap")
    # Sim name in the /tmp file name: two tasks on one node must not collide.
    spectra="/tmp/camel_lya_lya_h_spectra_${sim}_snap_${snap}_n${N_SIGHTLINES}.hdf5"
    link="${SPECTRA_ROOT}/${sim}/camel_lya_lya_h_spectra_snap_${snap}_n${N_SIGHTLINES}.hdf5"

    [ -f "$snapshot" ] || { echo "ERROR: snapshot not found: $snapshot"; exit 1; }
    # Same sightlines as every 1P and EX run, so the results are comparable.
    [ -f "$SIGHTLINES_FILE" ] || { echo "ERROR: sightline file missing: $SIGHTLINES_FILE"; exit 1; }
    if [ -d "$out" ]; then
        echo "SKIPPED, already analysed: $out"
        exit 0
    fi

    rm -f "$spectra" "$link"

    echo "[1/2] generate: $snapshot  (lines: $LINES)"
    step_start generate
    OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py generate "$snapshot" \
        --sightlines-from "$SIGHTLINES_FILE" --line "$LINES" -n $N_SIGHTLINES --output "$spectra"
    rc=$?
    step_end $rc
    if [ $rc -ne 0 ]; then
        echo "ERROR: generate failed ($sim snap_$snap)"
        timing_total $rc; rm -f "$spectra"; exit 1
    fi

    # analyze reads set/sim/snap from the spectra path, so link it into spectra/.
    mkdir -p "$(dirname "$link")"
    ln -sf "$spectra" "$link"

    echo "[2/2] analyze: $link"
    step_start analyze
    OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-128} python analyze_spectra.py analyze "$link" --workers 1
    rc=$?
    step_end $rc
    rm -f "$spectra" "$link"
    if [ $rc -ne 0 ]; then
        echo "ERROR: analyze failed ($sim snap_$snap)"
        timing_total $rc; exit 1
    fi

    timing_total 0
    echo "$sim snap_$snap done: $(date)"
    exit 0
}


# =====================================================================
# Login-node commands
# =====================================================================

# Downloads all missing sims for one snapshot in parallel and waits for them.
do_download() {
    local snap; snap=$(pad "${1:?usage: cv_set.sh download <snapshot>}")
    source .venv/bin/activate || { echo "ERROR: venv activation failed"; return 1; }
    local i n=0
    for i in $(seq 0 $((N_SIMS - 1))); do
        staged "$i" "$snap" && continue
        # downloader.py skips existing files, so remove a partial one first.
        rm -f "$(snap_file "$i" "$snap")"
        python downloader.py --suite IllustrisTNG --set CV --sim "$i" --snapshot "$((10#$snap))" \
            > "logs/download_CV_${i}_snap_${snap}.log" 2>&1 &
        n=$((n + 1))
    done
    echo "snap_${snap}: downloading ${n} of ${N_SIMS}"
    wait
    local have; have=$(staged_count "$snap")
    echo "snap_${snap}: ${have}/${N_SIMS} complete"
    [ "$have" -eq "$N_SIMS" ]
}

# Deletes a snapshot's files, but only once all 27 sims have analysis output.
do_clean() {
    local snap; snap=$(pad "${1:?usage: cv_set.sh clean <snapshot> [--dry-run]}")
    local dry=${2:-}
    local n; n=$(analysed_count "$snap")
    if [ "$n" -ne "$N_SIMS" ]; then
        echo "NOT deleting snap_${snap}: only ${n}/${N_SIMS} sims analysed."
        return 1
    fi

    local i f freed=0
    for i in $(seq 0 $((N_SIMS - 1))); do
        f=$(snap_file "$i" "$snap")
        [ -f "$f" ] || continue
        freed=$((freed + $(du -m "$f" | cut -f1)))
        if [ "$dry" = "--dry-run" ]; then echo "would remove $f"; else rm -f "$f"; fi
    done
    echo "snap_${snap}: ${dry:+(dry run) }freed ~${freed} MB"
}

do_status() {
    local total=0 snap n
    for snap in "${ALL_SNAPS[@]}"; do
        n=$(analysed_count "$snap")
        total=$((total + n))
        printf '  snap_%s  %2d/%d\n' "$snap" "$n" "$N_SIMS"
    done
    printf '  total    %d/%d\n' "$total" "$((N_SIMS * ${#ALL_SNAPS[@]}))"
}

do_run() {
    local snaps=("$@")
    [ ${#snaps[@]} -eq 0 ] && snaps=("${ALL_SNAPS[@]}")
    local s snap n

    for s in "${snaps[@]}"; do
        snap=$(pad "$s")
        echo; echo "=== snap_${snap}  $(date) ==="

        n=$(analysed_count "$snap")
        if [ "$n" -eq "$N_SIMS" ]; then
            echo "already analysed"
            do_clean "$snap"
            continue
        fi

        do_download "$snap" || { echo "ERROR: download incomplete for snap_${snap}"; return 1; }

        # --wait blocks until the array finishes; non-zero if any task failed.
        # Re-running is cheap: finished tasks exit immediately.
        if ! sbatch --wait --export=ALL,SNAP="$snap" "${BASH_SOURCE[0]}"; then
            echo "ERROR: array job failed for snap_${snap}; see logs/cv_*.err, then re-run."
            return 1
        fi

        do_clean "$snap" || return 1
    done

    echo; echo "CV set finished: $(date)"
    do_status
}

# Checks the delete guard and the download check on a throwaway tree.
do_selftest() {
    local tmp; tmp=$(mktemp -d)
    ( cd "$tmp" || exit 1
      for i in $(seq 0 26); do
          mkdir -p "$DATA_ROOT/CV_$i"; echo x > "$DATA_ROOT/CV_$i/snap_032.hdf5"
      done
      for i in $(seq 0 20); do
          mkdir -p "$OUT_ROOT/CV_$i/snap-032"; touch "$OUT_ROOT/CV_$i/snap-032/cddf.csv"
      done
      count() { find "$DATA_ROOT" -name 'snap_032.hdf5' | wc -l; }

      do_clean 032 > /dev/null && { echo "FAIL: clean ran with 21/27 analysed"; exit 1; }
      [ "$(count)" -eq 27 ] || { echo "FAIL: refused clean deleted files"; exit 1; }

      for i in $(seq 21 26); do
          mkdir -p "$OUT_ROOT/CV_$i/snap-032"; touch "$OUT_ROOT/CV_$i/snap-032/cddf.csv"
      done
      do_clean 032 --dry-run > /dev/null || { echo "FAIL: dry run errored"; exit 1; }
      [ "$(count)" -eq 27 ] || { echo "FAIL: dry run deleted files"; exit 1; }
      do_clean 032 > /dev/null || { echo "FAIL: clean refused 27/27"; exit 1; }
      [ "$(count)" -eq 0 ] || { echo "FAIL: clean left files"; exit 1; }

      [ "$(snap_file 3 044)" = "data/IllustrisTNG/CV/CV_3/snap_044.hdf5" ] \
          || { echo "FAIL: snap_file path"; exit 1; }

      # HDF5 signature + 32 bytes + EOF address 64, i.e. a 48-byte file claiming 64 bytes.
      local f="$DATA_ROOT/CV_0/snap_099.hdf5"
      printf '\211HDF\r\n\032\n' > "$f"
      head -c 32 /dev/zero >> "$f"
      printf '\100\0\0\0\0\0\0\0' >> "$f"
      staged 0 099 && { echo "FAIL: truncated file accepted"; exit 1; }
      head -c 16 /dev/zero >> "$f"
      staged 0 099 || { echo "FAIL: complete file rejected"; exit 1; }
      echo x > "$DATA_ROOT/CV_1/snap_099.hdf5"
      staged 1 099 && { echo "FAIL: non-HDF5 file accepted"; exit 1; }

      echo "self-test OK" )
    local rc=$?
    rm -rf "$tmp"
    return $rc
}


if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    worker
fi

cmd=${1:-run}; shift 2>/dev/null
case "$cmd" in
    run)      do_run "$@" ;;
    download) do_download "$@" ;;
    clean)    do_clean "$@" ;;
    status)   do_status ;;
    selftest) do_selftest ;;
    *)        echo "usage: cv_set.sh [run [snaps...] | download <snap> | clean <snap> [--dry-run] | status | selftest]"
              exit 2 ;;
esac
