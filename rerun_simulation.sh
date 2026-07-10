#!/bin/bash
#SBATCH --job-name=rerun_g2000_f10
#SBATCH --partition=defq
#SBATCH --cpus-per-task=25
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/rerun_g2000_f10_%j.log
#SBATCH --error=/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp/rerun_g2000_f10_%j.err

echo "=========================================="
echo "Rerun Failed Scenario Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

source /home/valenciai/anaconda3/bin/activate sctrial_bench
module load R/4.4.2
module load openblas/dynamic/0.3.18
module load gsl/2.7.1
export LD_LIBRARY_PATH=/apps/gsl/2.7.1/lib:/apps/spack/opt/spack/linux-rhel8-haswell/gcc-11.2.0/libiconv-1.16-myo6dp2rszjfqkp7w456qrt4aqdtcnis/lib:$LD_LIBRARY_PATH
cd /common/omarmlab/members/itzel/sctrial_bench/sctrial

python -u /common/omarmlab/members/itzel/sctrial_bench/sctrial/rerun_simulation.py

echo "Job finished: $(date)"