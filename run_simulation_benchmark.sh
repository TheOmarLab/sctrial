#!/bin/bash
#SBATCH --job-name=simulation_benchmark
#SBATCH --partition=defq
#SBATCH --cpus-per-task=25
#SBATCH --mem=356G
#SBATCH --time=120:00:00
#SBATCH --output=/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/simulation_benchmark_%j.log
#SBATCH --error=/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/simulation_benchmark_%j.err

echo "=========================================="
echo "Simulation Benchmark Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ── Activate environment ───────────────────────────────
source /home/valenciai/anaconda3/etc/profile.d/conda.sh
conda activate sctrial_bench
module load gcc/11.2.0
module load openblas/dynamic/0.3.18
module load gsl/2.7.1
module load R/4.4.2
export LD_LIBRARY_PATH="/apps/spack/opt/spack/linux-rhel8-haswell/gcc-11.2.0/libiconv-1.16-myo6dp2rszjfqkp7w456qrt4aqdtcnis/lib:$LD_LIBRARY_PATH"

cd /common/omarmlab/members/itzel/sctrial_bench/sctrial
pip install -e . --quiet

# ── Output directory ──────────────────────────
mkdir -p /common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/simulation

# ── Run benchmarks ────────────────────────────
python -u /common/omarmlab/members/itzel/sctrial_bench/sctrial/run_simulation_benchmark.py

echo "=========================================="
echo "ALL DONE"
echo "Job finished: $(date)"
echo "=========================================="