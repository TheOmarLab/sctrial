#!/bin/bash
#SBATCH --job-name=sctrial_realdata
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=realdata_%j.out
#SBATCH --error=realdata_%j.err

# Load R modules (adjust names to match your cluster)
module load gcc/11.2.0
module load openblas/dynamic/0.3.18
module load gsl/2.7.1
module load R/4.4.2
export R_LIBS_USER="$HOME/R/library"

# Activate Python environment
source ~/.bashrc
conda activate sctrial_benchmark

# Force unbuffered Python output so progress lines appear immediately in SLURM log
export PYTHONUNBUFFERED=1

# Set working directory
cd /common/vasanthakup/projects/sctrial   # ← update to your HPC project path

# Create output directory
mkdir -p manuscript/benchmark/realdata

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
python -c "from sctrial.benchmark.permutation import run_permutation_test; print('permutation OK')"
python -c "from sctrial.benchmark.subsample import run_subsampling; print('subsampling OK')"
python -c "from sctrial.datasets import load_sade_feldman, load_tnbc_zhang; print('datasets OK')"
python -c "import subprocess; subprocess.run(['Rscript', '-e', 'library(dreamlet); library(nebula); cat(\"Rscript OK\n\")'], check=True)"

# Print git commit for reproducibility
python -c "import sctrial.benchmark.orchestrator as o; import inspect; print('Orchestrator:', inspect.getfile(o))"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"

# Run phase 3: real-data permutation + subsampling on TNBC
# n-jobs=8: each worker loads TNBC adata + runs NEBULA in R (~3-4 GB/worker).
# 30 workers × ~4 GB = ~120 GB peak, which OOMs the 256 GB node.
# 8 workers × ~4 GB = ~32 GB, leaving ample headroom.
python -u scripts/run_benchmark.py \
  --phase realdata \
  --n-jobs 8

echo "============================================"
echo "Finished: $(date)"
echo "============================================"
