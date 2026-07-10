#!/bin/bash
#SBATCH --job-name=r_install
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/valenciai/r_install_%j.log
#SBATCH --error=/home/valenciai/r_install_%j.err

echo "=========================================="
echo "R Package Install Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

module load openblas/dynamic/0.3.18
module load gsl/2.7.1
module load libxml2-2.9.13-gcc-11.2.0-xzgzoe3
module load R/4.4.2

# Add nlopt and libiconv paths
export LD_LIBRARY_PATH=/apps/fsl/3.18.0/lib:/apps/spack/opt/spack/linux-rhel8-haswell/gcc-11.2.0/libxml2-2.9.13-xzgzoe35wh2fv7n4glf3tecymem7zjoj/lib:/apps/spack/opt/spack/linux-rhel8-haswell/gcc-11.2.0/libiconv-1.16-myo6dp2rszjfqkp7w456qrt4aqdtcnis/lib:$LD_LIBRARY_PATH

export PKG_CONFIG_PATH=/apps/spack/opt/spack/linux-rhel8-haswell/gcc-11.2.0/libxml2-2.9.13-xzgzoe35wh2fv7n4glf3tecymem7zjoj/lib/pkgconfig:$PKG_CONFIG_PATH

# Point nloptr to nlopt
export INCLUDE_DIR=/apps/fsl/3.18.0/pkgs/nlopt-2.10.1-np2py312h0f77346_2/include
export LIB_DIR=/apps/fsl/3.18.0/lib

# Remove conda from PATH to avoid conflicts
export PATH=$(echo $PATH | tr ':' '\n' | grep -v anaconda | tr '\n' ':')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v anaconda | tr '\n' ':')

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

Rscript -e '
.libPaths("~/R/library")
options(repos=c(CRAN="https://cran.r-project.org"))

if (!requireNamespace("BiocManager", quietly=TRUE)) {
    install.packages("BiocManager", lib="~/R/library")
}

# Step 1: CRAN dependencies
cat("Step 1: Installing CRAN dependencies...\n")
install.packages(c("XML", "RCurl", "nloptr", "lme4", "httr", "fs"), 
                 lib="~/R/library")

# Step 2: Bioconductor base dependencies
cat("Step 2: Installing Bioconductor base dependencies...\n")
BiocManager::install(c(
    "AnnotationDbi", "org.Hs.eg.db", "GO.db", "KEGGgraph",
    "KEGGREST", "annotate", "GSEABase", "SummarizedExperiment",
    "SingleCellExperiment", "variancePartition", "mashr", "beachmat"
), lib="~/R/library", ask=FALSE)

# Step 3: Seurat
cat("Step 3: Installing Seurat...\n")
install.packages("Seurat", lib="~/R/library")

# Step 4: dreamlet and nebula
cat("Step 4: Installing dreamlet and nebula...\n")
BiocManager::install(c("dreamlet", "nebula"), lib="~/R/library", ask=FALSE)

# Verify
cat("Checking installs...\n")
library(dreamlet)
library(nebula)
cat("Both loaded OK!\n")
'

echo "Job finished: $(date)"