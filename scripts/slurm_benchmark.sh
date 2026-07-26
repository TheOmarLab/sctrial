#!/bin/bash
# The definitive benchmark run. Runs ONCE, after the gates pass and the
# configuration is frozen.
#
#   sbatch scripts/slurm_benchmark.sh core        200
#   sbatch scripts/slurm_benchmark.sh sensitivity 200
#
# NEVER run on a login node.
#
#SBATCH --job-name=sctrial_bench
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --output=/common/omarmlab/members/omar/projects/sctrial/logs/bench_%x_%j.out
#SBATCH --error=/common/omarmlab/members/omar/projects/sctrial/logs/bench_%x_%j.err

# Source the profile BEFORE enabling strict mode: the system /etc/bashrc reads an
# unset variable, so `set -u` aborts the job at line 1 with a message that looks
# like a cluster problem rather than a script bug.
source ~/.bashrc
set -eo pipefail
PROJECT=/common/omarmlab/members/omar/projects/sctrial
cd "$PROJECT"
mkdir -p logs
export SCTRIAL_MANUSCRIPT_DIR="$PROJECT/manuscript"

GRID="${1:-core}"
N_ITER="${2:-200}"

module load gcc/11.2.0 2>/dev/null || true
module load openblas/dynamic/0.3.18 2>/dev/null || true
module load gsl/2.7.1 2>/dev/null || true
module load R/4.4.2 2>/dev/null || true
export R_LIBS_USER="$HOME/R/library"

# Pin BLAS threads. Unpinned, 32 workers each spawn a thread pool, oversubscribe
# the node, and the runtimes this benchmark reports become a measure of
# scheduler contention rather than of the methods.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# R subprocesses are the bottleneck and are memory-hungry at 2000 genes; more
# than ~10 concurrent workers caused every 2000-gene iteration to return NaN.
N_JOBS=10

case "$GRID" in
  core)        PHASE=simulate ;;
  sensitivity) PHASE=sensitivity ;;
  *) echo "unknown grid: $GRID" >&2; exit 2 ;;
esac

echo "=== $GRID grid, $N_ITER iterations, $N_JOBS workers ($(date)) ==="
micromamba run -n sctrial python scripts/run_benchmark.py \
    --phase "$PHASE" --n-jobs "$N_JOBS" --n-iterations "$N_ITER"
echo "=== done ($(date)) ==="
