#!/bin/bash
# Simulator calibration and Monte Carlo gates.
#
# NEVER run any of this on a login node. Every phase here loads a 141k-cell
# object and/or generates hundreds of full-scale replicates; a previous attempt
# on the login node was OOM-killed (exit 137).
#
#   sbatch scripts/slurm_calibrate.sh targets
#   sbatch scripts/slurm_calibrate.sh gates 200
#   sbatch scripts/slurm_calibrate.sh ablate
#   sbatch scripts/slurm_calibrate.sh freeze
#
#SBATCH --job-name=sctrial_calib
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=/common/omarmlab/members/omar/projects/sctrial/logs/calib_%x_%j.out
#SBATCH --error=/common/omarmlab/members/omar/projects/sctrial/logs/calib_%x_%j.err

# Source the profile BEFORE enabling strict mode: the system /etc/bashrc reads an
# unset variable, so `set -u` aborts the job at line 1 with a message that looks
# like a cluster problem rather than a script bug.
source ~/.bashrc
set -eo pipefail
PROJECT=/common/omarmlab/members/omar/projects/sctrial
cd "$PROJECT"
mkdir -p logs

# The manuscript tree lives INSIDE the project root on the cluster and beside
# the repo locally. Stated explicitly so no path guess can resolve elsewhere.
export SCTRIAL_MANUSCRIPT_DIR="$PROJECT/manuscript"

PHASE="${1:-targets}"
N_MC="${2:-200}"

# Pin the BLAS thread count. Unpinned, every worker spawns its own thread pool,
# they oversubscribe the node, and the wall times this benchmark reports become
# a function of scheduler contention rather than of the methods.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# TNBC column names, verified against the processed object:
#   participant_id / visit / arm / cell_type, raw counts in layer "counts".
# Cell type is NOT optional: pooling cell types inflates the conditional
# dispersion 2.8x by loading between-population mean differences onto it.
COMMON="--dataset tnbc --participant-col participant_id --visit-col visit \
        --arm-col arm --celltype-col cell_type --layer counts"

echo "=== phase: $PHASE  ($(date)) ==="

case "$PHASE" in
  targets)
    micromamba run -n sctrial python scripts/calibrate_simulator.py $COMMON targets
    ;;
  gates)
    micromamba run -n sctrial python scripts/calibrate_simulator.py $COMMON \
        gates --n-mc "$N_MC" --n-jobs 16
    ;;
  ablate)
    micromamba run -n sctrial python scripts/calibrate_simulator.py $COMMON \
        ablate --n-rep 5
    ;;
  freeze)
    micromamba run -n sctrial python scripts/calibrate_simulator.py $COMMON freeze
    ;;
  diagnose)
    micromamba run -n sctrial python scripts/calibrate_simulator.py $COMMON diagnose
    ;;
  nebula-offset)
    micromamba run -n sctrial python scripts/verify_nebula_offset.py
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

echo "=== done ($(date)) ==="
