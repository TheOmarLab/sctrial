#!/bin/bash
#SBATCH --job-name=sctrial_benchmark
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err

# Activate environment
source ~/.bashrc
eval "$(micromamba shell hook --shell bash)"
micromamba activate sctrial

# Set working directory
cd /common/omarmlab/members/omar/projects/sctrial

# Create output directory
mkdir -p manuscript/benchmark/simulation

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
python -c "import rpy2.robjects as ro; ro.r('library(edgeR)'); print('rpy2+R OK')"

# Run the benchmark
python scripts/run_benchmark.py \
  --phase simulate \
  --n-jobs 30 \
  --n-iterations 200

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
