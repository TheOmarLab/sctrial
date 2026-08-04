#!/bin/bash
# Maximum-load timing and memory probe. Run ONCE before sizing the definitive
# benchmark allocation, after any change that affects scenario size.
#
#   sbatch scripts/slurm_timing_probe.sh
#
# NEVER run on a login node.
#
#SBATCH --job-name=sctrial_probe
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/common/omarmlab/members/omar/projects/sctrial/logs/probe_%j.out
#SBATCH --error=/common/omarmlab/members/omar/projects/sctrial/logs/probe_%j.err

# Source the profile BEFORE enabling strict mode: the system /etc/bashrc reads an
# unset variable, so `set -u` aborts the job at line 1 with a message that looks
# like a cluster problem rather than a script bug.
source ~/.bashrc
set -eo pipefail
PROJECT=/common/vasanthakup/projects/sctrial
cd "$PROJECT"
mkdir -p logs
export SCTRIAL_MANUSCRIPT_DIR="$PROJECT/manuscript"

module load gcc/11.2.0 2>/dev/null || true
module load openblas/dynamic/0.3.18 2>/dev/null || true
module load gsl/2.7.1 2>/dev/null || true
module load R/4.4.2 2>/dev/null || true
export R_LIBS_USER="$HOME/R/library"

# Pinned for the same reason as the benchmark: an unpinned BLAS turns a timing
# measurement into a measurement of scheduler contention. The probe must be
# measured under the SAME threading the definitive run will use, or its numbers
# do not transfer.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ONE worker: this measures the cost of a single replicate in isolation, which is
# the quantity the allocation is sized from. Measuring it under concurrency would
# fold contention into the per-replicate figure and then multiply by the worker
# count again.
echo "=== timing probe ($(date)) ==="
conda run -n sctrial_benchmark python scripts/timing_probe.py --n-jobs 1
echo "=== done ($(date)) ==="
