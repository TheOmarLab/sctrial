#!/bin/bash
# The definitive benchmark run. Runs ONCE, after the gates pass and the
# configuration is frozen.
#
#   sbatch scripts/slurm_benchmark.sh core        200 "" ""          32
#   sbatch scripts/slurm_benchmark.sh sensitivity 200 "" "50 200 500" 24
#   sbatch scripts/slurm_benchmark.sh sensitivity 200 "" "2000"       10
#
# The worker count is an ARGUMENT because its binding constraint is panel-size
# dependent, not global. Thirty concurrent R workers at 2,000 genes returned 100%
# NaN, but that limit does not apply to the 50-gene panels -- and applying it
# everywhere was costing roughly 3x on the core grid, which is 89% Python
# simulation and touches R only briefly.
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
DESIGN="${3:-}"   # optional: two_arm | single_arm; split for wall-clock
PANELS="${4:-}"   # optional: sensitivity panel sizes, e.g. "50 200 500"
N_JOBS="${5:-}"   # optional: worker count; defaults below by panel size

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

# Default worker count, chosen by what actually binds.
#
# MEASURED (job 5292134, one replicate, one worker):
#   cells_1000_n40  wall 426 s = 381 s simulation + 46 s fitting  -> 89% simulation
#   sens_null_g2000 wall 539 s = 194 s simulation + 345 s fitting
#   peak 5.04 GB per worker, at 160,000 cells
#
# So the 10-worker R limit binds only where R dominates, which is the 2,000-gene
# panel. Everywhere else the cost is Python simulation and the limit is memory:
# 32 x 5.04 GB = 161 GB against a 400 GB request.
if [ -z "$N_JOBS" ]; then
  case "$PANELS" in
    *2000*) N_JOBS=10 ;;   # R at 2000 genes: 30 workers gave 100% NaN
    "")     N_JOBS=$([ "$GRID" = "core" ] && echo 32 || echo 10) ;;
    *)      N_JOBS=24 ;;
  esac
fi

case "$GRID" in
  core)        PHASE=simulate ;;
  sensitivity) PHASE=sensitivity ;;
  *) echo "unknown grid: $GRID" >&2; exit 2 ;;
esac

DESIGN_ARG=()
if [ -n "$DESIGN" ]; then DESIGN_ARG=(--designs "$DESIGN"); fi
if [ -n "$PANELS" ]; then DESIGN_ARG+=(--panels $PANELS); fi

echo "=== $GRID grid, $N_ITER iterations, $N_JOBS workers, design=${DESIGN:-all} ($(date)) ==="
micromamba run -n sctrial python scripts/run_benchmark.py \
    --phase "$PHASE" --n-jobs "$N_JOBS" --n-iterations "$N_ITER" "${DESIGN_ARG[@]}"
echo "=== done ($(date)) ==="
