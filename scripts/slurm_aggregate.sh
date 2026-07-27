#!/bin/bash
# Combine the benchmark shards. Runs ONCE, after every producer succeeds.
#
#   sbatch --dependency=afterok:J1:J2 scripts/slurm_aggregate.sh core 200
#
#SBATCH --job-name=sctrial_aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/common/omarmlab/members/omar/projects/sctrial/logs/aggregate_%x_%j.out
#SBATCH --error=/common/omarmlab/members/omar/projects/sctrial/logs/aggregate_%x_%j.err

source ~/.bashrc
set -eo pipefail
PROJECT=/common/omarmlab/members/omar/projects/sctrial
cd "$PROJECT"
mkdir -p logs
export SCTRIAL_MANUSCRIPT_DIR="$PROJECT/manuscript"

GRID="${1:-core}"
N_ITER="${2:-200}"

echo "=== aggregate $GRID ($(date)) ==="
micromamba run -n sctrial python scripts/aggregate_benchmark.py "$GRID" --iterations "$N_ITER"
echo "=== done ($(date)) ==="
