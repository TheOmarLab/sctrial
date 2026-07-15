#!/bin/bash
#SBATCH --job-name=sctrial_r_install
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=r_install_%j.out
#SBATCH --error=r_install_%j.err

# ============================================================
# Install R and Bioconductor packages for sctrial benchmark.
#
# Run once before submitting benchmark jobs:
#   sbatch scripts/install_r.sh
#
# Packages are installed to ~/R/library (user-local, outside conda).
# R is loaded from the HPC module system — adjust module names below
# to match your cluster (check available versions with: module avail R).
# ============================================================

echo "=========================================="
echo "R Package Install Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# ── Load HPC modules ─────────────────────────────────────────
# Adjust these module names to match your cluster.
# Required: R itself + BLAS/GSL for mixed-model packages (lme4, dreamlet).
module load gcc/11.2.0
module load openblas/dynamic/0.3.18
module load gsl/2.7.1
module load R/4.4.2

echo "R: $(which R)"
echo "R version: $(R --version | head -1)"

# ── Preserve cmake before stripping conda ────────────────────
# cmake is needed to compile nloptr and fs (which bundles libuv).
# It may live in the conda env; save the path before we strip conda.
CMAKE_BIN=$(which cmake 2>/dev/null || echo "")

# ── Strip conda from PATH to prevent library conflicts ───────
# conda's libstdc++ and libgomp can shadow the system libs that R was
# compiled against, producing "undefined symbol" errors at package load time.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | grep -v anaconda | tr '\n' ':')
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v conda | grep -v anaconda | tr '\n' ':')

# Restore cmake specifically — it doesn't link R packages so it's safe to keep.
if [ -n "$CMAKE_BIN" ] && ! command -v cmake &>/dev/null; then
    export PATH="$(dirname "$CMAKE_BIN"):$PATH"
fi

echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo 'not found')"

# ── nlopt paths for nloptr ───────────────────────────────────
# nloptr looks for INCLUDE_DIR / LIB_DIR to find the nlopt C library.
# These paths match the FSL-bundled nlopt on this cluster; adjust if needed
# (find with: find /apps -name "nlopt.h" 2>/dev/null).
export INCLUDE_DIR=/apps/fsl/3.18.0/pkgs/nlopt-2.10.1-np2py312h0f77346_2/include
export LIB_DIR=/apps/fsl/3.18.0/lib
export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"

# ── R user library ────────────────────────────────────────────
mkdir -p "$HOME/R/library"
export R_LIBS_USER="$HOME/R/library"

Rscript -e '
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
options(repos = c(CRAN = "https://cran.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", lib = Sys.getenv("R_LIBS_USER"))

# Pin to Bioc 3.20 (matches R 4.4)
BiocManager::install(version = "3.20", ask = FALSE, update = FALSE)

lib <- Sys.getenv("R_LIBS_USER")

# Step 1: CRAN dependencies needed before Bioc packages
cat("Step 1: CRAN dependencies...\n")
install.packages(
    c("XML", "RCurl", "nloptr", "lme4", "lmerTest", "pbkrtest", "httr", "fs",
      "Rcpp", "RcppArmadillo", "data.table", "reshape2", "png", "digest",
      "future", "future.apply"),
    lib = lib
)

# Step 2: Bioconductor core (dependency chain for dreamlet)
cat("Step 2: Bioconductor core packages...\n")
BiocManager::install(
    c("BiocGenerics", "S4Vectors", "IRanges", "XVector", "Biostrings",
      "GenomicRanges", "SparseArray", "DelayedArray", "SummarizedExperiment",
      "SingleCellExperiment", "BiocParallel", "DelayedMatrixStats", "beachmat",
      "edgeR", "limma", "variancePartition", "mashr"),
    lib = lib, ask = FALSE
)

# Step 3: Seurat (large install, separate step for easier debugging)
cat("Step 3: Seurat...\n")
install.packages("Seurat", lib = lib)

# Step 4: dreamlet and nebula (the DE methods used by the benchmark runners)
cat("Step 4: dreamlet and nebula...\n")
BiocManager::install(c("dreamlet", "nebula"), lib = lib, ask = FALSE)

# Verify all critical packages load
cat("\nVerifying:\n")
for (pkg in c("edgeR", "limma", "variancePartition", "dreamlet", "nebula")) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE, lib.loc = lib))
    cat("OK:", pkg, as.character(packageVersion(pkg)), "\n")
}
cat("All R packages installed and verified.\n")
'

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
