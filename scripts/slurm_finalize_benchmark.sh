#!/bin/bash
# The publication finalizer. Runs ONCE, after BOTH grid aggregators succeed.
#
#   5 producers -> 2 grid aggregators -> 1 publication finalizer
#
#   P=$(sbatch --parsable --dependency=afterok:$A1:$A2 scripts/slurm_finalize_benchmark.sh)
#
# Writes results/<manifest>/publication_complete.json, which the figure and
# manuscript entry points require. A grid-level marker is NOT sufficient: core can
# aggregate successfully while sensitivity never finishes, and a figure script
# would then regenerate manuscript outputs from half the benchmark without error.
#
#SBATCH --job-name=sctrial_finalize
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/common/omarmlab/members/omar/projects/sctrial/logs/finalize_%j.out
#SBATCH --error=/common/omarmlab/members/omar/projects/sctrial/logs/finalize_%j.err

source ~/.bashrc
set -eo pipefail
PROJECT=/common/omarmlab/members/omar/projects/sctrial
cd "$PROJECT"
mkdir -p logs
export SCTRIAL_MANUSCRIPT_DIR="$PROJECT/manuscript"

# The manifest is read from the frozen configuration, never passed by hand.
echo "=== finalize benchmark ($(date)) ==="
micromamba run -n sctrial python scripts/finalize_benchmark.py
echo "=== done ($(date)) ==="
