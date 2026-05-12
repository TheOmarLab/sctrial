#!/bin/bash
#SBATCH --job-name=sctrial_sens_2k
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --output=sensitivity_2k_%j.out
#SBATCH --error=sensitivity_2k_%j.err

# Re-run only the 5 × 2000-gene scenarios that failed in the first run
# (dreamlet/NEBULA hit timeouts/OOM with 30 parallel workers).
# Strategy:
#   - Delete the 5 broken 2000-gene CSVs so the orchestrator's resume
#     logic re-runs them (and skips the already-good 50/200/500-gene CSVs)
#   - Use only 10 parallel workers (1/3 of original) to reduce memory
#     pressure on R subprocesses
#   - Increased dreamlet timeout to 1800s and NEBULA to 2400s

source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate sctrial

cd /common/omarmlab/members/omar/projects/sctrial

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start: $(date)"
echo "============================================"

# Sanity check the new code is loaded
python -c "
import inspect
from sctrial.benchmark.runners.dreamlet_runner import run as dr
src = inspect.getsource(dr)
assert 'timeout=1800' in src, 'STALE: dreamlet timeout not 1800'
print('dreamlet timeout=1800: OK')

from sctrial.benchmark.runners.nebula_runner import run as nr
src2 = inspect.getsource(nr)
assert 'timeout=2400' in src2, 'STALE: NEBULA timeout not 2400'
print('NEBULA timeout=2400: OK')
"

# Delete only the 2000-gene CSVs (resume logic will re-run them, skip rest)
echo ""
echo "Deleting broken 2000-gene CSVs..."
SENS_DIR="/common/omarmlab/members/omar/projects/sctrial/manuscript/benchmark/sensitivity"
for f in "$SENS_DIR"/two_arm__sens_g2000_f1.csv \
         "$SENS_DIR"/two_arm__sens_g2000_f5.csv \
         "$SENS_DIR"/two_arm__sens_g2000_f10.csv \
         "$SENS_DIR"/two_arm__sens_g2000_f20.csv \
         "$SENS_DIR"/two_arm__sens_null_g2000.csv; do
  if [ -f "$f" ]; then
    rm "$f"
    echo "  rm $(basename $f)"
  fi
done

# Also delete the combined CSV so it gets regenerated
rm -f "$SENS_DIR/sensitivity_combined.csv"
echo "  rm sensitivity_combined.csv (will be regenerated)"

# Re-run with reduced parallelism
echo ""
echo "Running 2000-gene scenarios with 10 workers (resume mode)..."
python scripts/run_benchmark.py \
  --phase sensitivity \
  --n-jobs 10 \
  --n-iterations 200

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
