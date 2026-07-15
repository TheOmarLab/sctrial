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

# Load R modules (adjust names to match your cluster)
module load gcc/11.2.0
module load openblas/dynamic/0.3.18
module load gsl/2.7.1
module load R/4.4.2
export R_LIBS_USER="$HOME/R/library"

# Activate Python environment
source ~/.bashrc
conda activate sctrial_benchmark

# Set working directory
cd /common/vasanthakup/projects/sctrial   # ← update to your HPC project path

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
python -c "import subprocess; subprocess.run(['Rscript', '-e', 'library(edgeR); library(limma); library(dreamlet); library(nebula); cat(\"Rscript OK\n\")'], check=True)"

# Print git commit and installed package path for reproducibility
python -c "import sctrial.benchmark.orchestrator as o; import inspect; print('Orchestrator:', inspect.getfile(o))"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"

# Run the benchmark
python scripts/run_benchmark.py \
  --phase simulate \
  --n-jobs 30 \
  --n-iterations 200

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
