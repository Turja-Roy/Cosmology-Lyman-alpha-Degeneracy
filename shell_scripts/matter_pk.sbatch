#!/bin/bash
#SBATCH -J matter_pk_test          # Job name
#SBATCH -o logs/job_%A_%a.out      # Output file (%A=job ID, %a=array task ID)
#SBATCH -e logs/job_%A_%a.err      # Error file
#SBATCH -p normal                  # Partition/queue (normal for Lonestar6)
#SBATCH -N 1                       # Number of nodes
#SBATCH -n 12                      # Number of tasks (cores) - for MPI or workers
#SBATCH -t 4:00:00                 # Time limit (4 hours)
#SBATCH --mail-type=END,FAIL       # Email notifications (optional)
#SBATCH --mail-user=turja.roy@uta.edu

# Load modules
module load cmake
module load gcc/13.2.0
module load impi/19.0.9
module load eigen/3.4.0
module load fftw3/3.3.10

source .venv/bin/activate

source shell_scripts/timing.sh
timing_context matter_pk all

step_start matter_pk
python scripts/matter_pk_test.py --data-root data/IllustrisTNG/1P --cosmo-csv data/IllustrisTNG/1P/CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv --snaps 014,018,024,032,044,080 --ngrid 256 --out-dir plots/matter_pk_test
rc=$?
step_end $rc
timing_total $rc
exit $rc
