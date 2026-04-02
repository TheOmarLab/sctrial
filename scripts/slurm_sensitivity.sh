#!/bin/bash
#SBATCH --job-name=sctrial_sensitivity
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=sensitivity_%j.out
#SBATCH --error=sensitivity_%j.err

# Activate environment
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate sctrial

# Set working directory
cd /common/omarmlab/members/omar/projects/sctrial

# Create output directory
mkdir -p manuscript/benchmark/sensitivity

# Print environment info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start: $(date)"
echo "Python: $(which python)"
echo "R: $(which R)"
echo "============================================"

# Verify packages
python -c "from sctrial.benchmark.simulator import SimulationConfig; print('sctrial OK')"
python -c "from sctrial.benchmark.orchestrator import build_sensitivity_grid; print('Sensitivity grid:', len(build_sensitivity_grid('two_arm')), 'scenarios')"
python -c "import subprocess; subprocess.run(['Rscript', '-e', 'library(dreamlet); library(nebula); cat(\"Rscript OK\n\")'], check=True)"

# Print git commit and installed package path for reproducibility
python -c "import sctrial.benchmark.orchestrator as o; import inspect; print('Orchestrator:', inspect.getfile(o))"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"

# Run the sensitivity benchmark
# 20 scenarios × 200 iterations × 4 methods
# 2000-gene scenarios are ~40× slower than 50-gene → allow 72h
python scripts/run_benchmark.py \
  --phase sensitivity \
  --n-jobs 30 \
  --n-iterations 200

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
