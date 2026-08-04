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
PROJECT=/common/vasanthakup/projects/sctrial
cd "$PROJECT"
mkdir -p logs
export SCTRIAL_MANUSCRIPT_DIR="$PROJECT/manuscript"

GRID="${1:-core}"
MIN_ITER="${2:-200}"

# The manifest hash is READ FROM THE FROZEN CONFIG, never passed by hand. It
# names the directory the producers wrote to, and a mistyped or stale hash would
# either aggregate the wrong run or fail to find one -- both after the compute
# has already been spent. There is exactly one frozen configuration, so there is
# exactly one correct answer and no reason for an operator to supply it.
SHA=$(conda run -n sctrial_benchmark python -c "
import json, sys
p = '$PROJECT/manuscript/benchmark/validation/frozen_simulator_config.json'
m = (json.load(open(p)).get('manifest') or {})
sha = m.get('manifest_sha256') or m.get('config_sha256')
if not sha:
    sys.exit('frozen config carries no manifest hash; re-run calibrate_simulator.py freeze')
print(sha)
")

echo "=== aggregate $GRID under manifest ${SHA:0:12} ($(date)) ==="
conda run -n sctrial_benchmark python scripts/aggregate_benchmark.py \
    "$GRID" "$SHA" --min-iterations "$MIN_ITER"
echo "=== done ($(date)) ==="
