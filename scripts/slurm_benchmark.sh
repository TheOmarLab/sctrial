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

# Default worker count per RESOURCE CLASS, every value measured rather than
# inferred. One global limit was wrong: the classes bind on different things.
#
# Single-replicate cost (job 5292134/5292154, one worker):
#   cells_1000_n40  426 s = 381 s simulation + 46 s fitting   -> 89% simulation
#   sens_null_g2000 539 s = 194 s simulation + 345 s fitting
#   null_n60        300 s = 285 s simulation + 15 s fitting
#   peak 5.70 GB per worker at 160,000 cells (sampled across the process tree)
#
# Under CONCURRENCY (jobs 5292318/19/20), every method 100% finite and 100%
# converged in all three:
#   32 workers, 50-gene      152.3 GB peak (4.76/worker)  CLEAN
#   24 workers, 1000 cells   127.8 GB peak (5.32/worker)  CLEAN
#   16 workers, 2000-gene     87.6 GB peak (5.48/worker)  CLEAN
#
# THE 10-WORKER LIMIT WAS TOO CONSERVATIVE. It came from thirty concurrent R
# workers at 2,000 genes returning 100% NaN; nobody had measured in between, and
# 16 is clean. That is a 1.6x speedup on the slowest job in the grid.
#
# Nodes are ~1 TB with 48 CPUs, so memory is not the binding constraint at any of
# these widths (152 GB is 15% of a node). Parallel efficiency is 85-88%.
if [ -z "$N_JOBS" ]; then
  case "$PANELS" in
    *2000*) N_JOBS=16 ;;   # measured clean; 30 was not, 10 was needlessly slow
    "")     N_JOBS=$([ "$GRID" = "core" ] && echo 32 || echo 16) ;;
    *)      N_JOBS=24 ;;   # 200/500-gene panels, bracketed by clean measurements
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
