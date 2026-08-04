#!/bin/bash
# Pre-flight check before the definitive benchmark run.
#
#   sbatch scripts/slurm_smoke.sh
#   sbatch scripts/slurm_smoke.sh --panel-probe
#
#SBATCH --job-name=sctrial_smoke
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/common/omarmlab/members/omar/projects/sctrial/logs/smoke_%j.out
#SBATCH --error=/common/omarmlab/members/omar/projects/sctrial/logs/smoke_%j.err

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

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== smoke ($(date)) ==="
conda run -n sctrial_benchmark python scripts/smoke_benchmark.py "$@"
echo "=== done ($(date)) ==="
