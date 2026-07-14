#!/bin/bash
# ============================================================
# HPC environment setup for sctrial benchmark
#
# Run once interactively on the HPC login node:
#   bash scripts/setup_hpc_env.sh
#
# What this does:
#   1. Creates the conda env "sctrial" from the reference yml
#   2. Installs sctrial from source (dev version)
#   3. Installs all required R packages
# ============================================================

set -euo pipefail

# ── 0. Configuration ─────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="$REPO_ROOT/src/sctrial/benchmark/sctrial_bench_environment.yml"
ENV_NAME="sctrial_benchmark"   # must match slurm scripts

echo "============================================"
echo "sctrial HPC environment setup"
echo "Repo:    $REPO_ROOT"
echo "Env:     $ENV_NAME"
echo "Start:   $(date)"
echo "============================================"

# ── 1. Python environment ─────────────────────────────────────
echo ""
echo ">>> Step 1: Creating conda environment '$ENV_NAME'"

# If the env already exists (e.g. a previous interrupted run), update it
# instead of recreating from scratch so already-downloaded packages are reused.
# Strip the sctrial line from the yml — we install it from source in Step 2,
# so installing the PyPI version here only causes an uninstall/reinstall cycle
# that can corrupt numpy.
_CLEAN_YML="$(mktemp /tmp/sctrial_env_XXXXXX.yml)"
grep -v "sctrial==" "$ENV_YML" > "$_CLEAN_YML"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "    Environment '$ENV_NAME' already exists — updating (resuming)."
    conda env update --name "$ENV_NAME" --file "$_CLEAN_YML" --prune
else
    conda env create --name "$ENV_NAME" --file "$_CLEAN_YML" --yes
fi
rm -f "$_CLEAN_YML"

# Activate — source activate works universally on HPC without extra shell hooks.
source activate "$ENV_NAME"

# Print the environment path so you know exactly where packages are installed.
echo "    Env path: $CONDA_PREFIX"

# conda may have installed its own numpy as a transitive dependency, leaving
# two conflicting numpy installations (conda + pip). Force pip to own numpy
# and pandas so they are in a single consistent state before anything imports them.
pip install --force-reinstall numpy pandas

# ── 2. Install sctrial from source ───────────────────────────
echo ""
echo ">>> Step 2: Installing sctrial from source (dev version)"
pip install -e "$REPO_ROOT[plots,gsea]" --no-deps

# Verify
python -c "import sctrial; print('sctrial version:', sctrial.__version__)"
python -c "
from sctrial.benchmark.orchestrator import build_sensitivity_grid
print('Sensitivity grid:', len(build_sensitivity_grid('two_arm')), 'scenarios')
"

# ── 3. Install R and system-level R packages via conda ────────
echo ""
echo ">>> Step 3: Installing R into the conda environment"
if command -v Rscript &>/dev/null; then
    echo "    R already installed: $(Rscript --version 2>&1 | head -1)"
else
    # Remove any R-installed (non-conda) copies of packages conda is about
    # to manage — conda refuses to overwrite files it didn't install itself.
    R_LIB="$CONDA_PREFIX/lib/R/library"
    echo "    Removing R-managed copies that would conflict with conda..."
    for pkg in Matrix Rcpp RcppArmadillo lme4 pbkrtest lmerTest nloptr XML curl data.table; do
        rm -rf "$R_LIB/$pkg"
    done

    # Install R and all packages that require compiled C/C++ code via conda
    # so their binaries match the conda R ABI exactly. Installing these from
    # CRAN source against a conda R causes undefined-symbol linker errors.
    conda install --name "$ENV_NAME" -c conda-forge \
        r-base \
        r-rcpp r-rcpparmadillo \
        r-lme4 r-pbkrtest r-lmertest r-nloptr \
        r-xml r-curl r-data.table \
        --yes
    Rscript --version
fi

# ── 4. R packages ─────────────────────────────────────────────
# Reference versions (sctrial_bench_R_packages_used.csv):
#   Matrix    1.7-5
#   edgeR     4.4.2
#   limma     3.62.2
#   dreamlet  1.4.1   (from Bioconductor)
#   nebula    1.5.6   (from CRAN)
#
# Installing from source takes ~2h — run in a screen/tmux session.

echo ""
echo ">>> Step 4: Installing R packages (this takes ~2 hours)"
echo "    Tip: if interrupted, re-run this script — already-installed"
echo "    packages are skipped."

R_SETUP_SCRIPT="$(mktemp /tmp/sctrial_r_setup_XXXXXX.R)"
cat > "$R_SETUP_SCRIPT" << 'REOF'
# CRAN repo only — BiocManager manages its own Bioconductor repos internally.
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Install BiocManager and pin to Bioc 3.20 (matches R 4.4).
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(version = "3.20", ask = FALSE)

# CRAN packages — lme4/pbkrtest/lmerTest/nloptr installed via conda in Step 3;
# listed here so any that are still missing get caught.
cran_pkgs <- c("Matrix", "lme4", "nloptr", "pbkrtest", "lmerTest", "nebula")
for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing (CRAN):", pkg, "\n"); flush.console()
    install.packages(pkg)
  } else {
    cat("Already installed:", pkg, "\n")
  }
}

# Bioconductor packages — only install what is missing.
# Omitting update=FALSE so BiocManager can install all transitive deps
# (SparseArray, DelayedArray, SummarizedExperiment, etc.).
bioc_pkgs <- c("edgeR", "limma", "variancePartition", "dreamlet")
bioc_missing <- bioc_pkgs[!sapply(bioc_pkgs, requireNamespace, quietly = TRUE)]
if (length(bioc_missing) > 0) {
  cat("Installing Bioconductor packages:", paste(bioc_missing, collapse = ", "), "\n")
  flush.console()
  BiocManager::install(bioc_missing, ask = FALSE)
} else {
  cat("All Bioconductor packages already installed.\n")
}

# Verify all are loadable
for (p in c("Matrix", "nebula", "edgeR", "limma", "dreamlet")) {
  suppressPackageStartupMessages(library(p, character.only = TRUE))
  cat("OK:", p, as.character(packageVersion(p)), "\n")
}
REOF

Rscript "$R_SETUP_SCRIPT"
rm -f "$R_SETUP_SCRIPT"

# ── 4. Sanity check ───────────────────────────────────────────
echo ""
echo ">>> Step 5: Final verification"

python -c "from sctrial.benchmark.simulator import SimulationConfig; print('Python benchmark: OK')"
Rscript -e 'library(dreamlet); library(nebula); cat("R packages: OK\n")'
python -c "
from sctrial.benchmark.runners import dreamlet_runner, nebula_runner
print('Runner imports: OK')
"

echo ""
echo "============================================"
echo "Setup complete: $(date)"
echo ""
echo "Next steps:"
echo "  Sensitivity benchmark (panels C-F):"
echo "    1. Edit scripts/slurm_sensitivity.sh — update the 'cd' path"
echo "    2. sbatch scripts/slurm_sensitivity.sh"
echo ""
echo "  Main simulation benchmark:"
echo "    1. Edit scripts/slurm_benchmark.sh — update the 'cd' path"
echo "    2. sbatch scripts/slurm_benchmark.sh"
echo "============================================"
