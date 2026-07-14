#!/bin/bash
# ============================================================
# HPC environment setup for sctrial benchmark
#
# Run once interactively on the HPC login node:
#   bash scripts/setup_hpc_env.sh
#
# What this does:
#   1. Creates the conda env "sctrial_benchmark" from the reference yml
#   2. Installs sctrial from source (dev version)
#   3. Installs R + all R/Bioconductor packages via conda (no compilation)
#   4. R-only fallback for any package conda could not provide
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

# Strip the sctrial line from the yml — we install it from source in Step 2,
# so installing the PyPI version here only causes an uninstall/reinstall cycle.
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
echo "    Env path: $CONDA_PREFIX"

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

# ── 3. R packages via conda (bioconda + conda-forge) ─────────
# ALL packages with compiled C/C++ code must come from conda so their binaries
# match the conda R ABI exactly. Compiling from CRAN / BiocManager against a
# conda R produces "undefined symbol" linker errors at load time. Affected
# packages: RSQLite, reshape2, XVector, Biostrings, png, RCurl, and their
# entire downstream dependency chain (GenomicRanges → SummarizedExperiment →
# SingleCellExperiment → dreamlet). We use bioconda for Bioconductor packages —
# pre-built binaries, no system headers needed.
echo ""
echo ">>> Step 3: Installing R and Bioconductor packages via conda"
echo "    Channels: bioconda + conda-forge (no compilation needed)"

R_LIB="$CONDA_PREFIX/lib/R/library"

# 3a. Remove any R-installed (non-conda) copies that would block conda.
#     conda refuses to install a package whose files already exist on disk
#     from another package manager. Safe to run on every re-run — conda
#     reinstalls its managed copies immediately after.
echo "    Clearing R-managed package copies that would cause ClobberError..."
for pkg in \
    Matrix Rcpp RcppArmadillo Rfast digest future future.apply \
    lme4 pbkrtest lmerTest nloptr XML curl data.table \
    RSQLite reshape2 RCurl png \
    BiocGenerics S4Vectors IRanges XVector Biostrings GenomicRanges \
    SparseArray DelayedArray SummarizedExperiment SingleCellExperiment \
    BiocParallel DelayedMatrixStats beachmat BiocFileCache \
    edgeR limma variancePartition dreamlet nebula; do
    rm -rf "$R_LIB/$pkg"
done

# 3b. Remove pip's numpy so conda can install its own without a ClobberError.
#     The conda solver pulls in numpy as a transitive dep even when installing
#     R packages, and collides with pip's untracked files. Reinstalled at end.
echo "    Temporarily removing pip numpy to avoid ClobberError during conda solve..."
pip uninstall -y numpy 2>/dev/null || true

# 3c. Clear potentially corrupt package cache to avoid SafetyError for r-base
#     (reported when a prior download was interrupted mid-write).
conda clean --packages --yes -q 2>/dev/null || true

# 3d. Core install: packages definitely available on bioconda / conda-forge.
echo "    Installing core R / Bioconductor packages..."
conda install --name "$ENV_NAME" \
    -c bioconda -c conda-forge \
    jq \
    r-base \
    r-rcpp r-rcpparmadillo \
    r-rfast r-digest r-future r-future.apply \
    r-lme4 r-pbkrtest r-lmertest r-nloptr \
    r-xml r-curl r-rcurl r-data.table \
    r-rsqlite r-reshape2 r-png \
    bioconductor-biocgenerics \
    bioconductor-s4vectors \
    bioconductor-iranges \
    bioconductor-xvector \
    bioconductor-genomicranges \
    bioconductor-sparsearray \
    bioconductor-delayedarray \
    bioconductor-summarizedexperiment \
    bioconductor-singlecellexperiment \
    bioconductor-biocparallel \
    bioconductor-delayedmatrixstats \
    bioconductor-beachmat \
    bioconductor-edger \
    bioconductor-limma \
    bioconductor-variancepartition \
    --yes

# 3e. Optional: dreamlet and nebula via conda. Non-fatal if the version is not
#     yet in the channel — Step 4 installs them via R as a fallback.
echo "    Attempting dreamlet and nebula via conda (non-fatal if unavailable)..."
if conda install --name "$ENV_NAME" \
        -c bioconda -c conda-forge \
        bioconductor-dreamlet r-nebula \
        --yes 2>&1; then
    echo "    dreamlet and nebula installed via conda."
else
    echo "    NOTE: not found in conda channels — Step 4 will use R fallback."
fi

echo "    R version: $(Rscript --version 2>&1 | head -1)"

# ── 4. R fallback — install any packages conda missed ─────────
# Typically a no-op when conda installed everything above.
# Only runs BiocManager / CRAN for packages still absent after Step 3.
echo ""
echo ">>> Step 4: R package verification and fallback install"

R_SETUP_SCRIPT="$(mktemp /tmp/sctrial_r_setup_XXXXXX.R)"
cat > "$R_SETUP_SCRIPT" << 'REOF'
pkgs_all <- c("nebula", "edgeR", "limma", "variancePartition", "dreamlet")
missing  <- pkgs_all[!sapply(pkgs_all, requireNamespace, quietly = TRUE)]

if (length(missing) == 0) {
  cat("All packages already installed via conda — nothing to do.\n")
} else {
  cat("Packages missing from conda install:",
      paste(missing, collapse = ", "), "\n")
  cat("Installing via BiocManager / CRAN fallback...\n")

  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org")

  # Sync to Bioc 3.20 (required for R 4.4). force=TRUE handles the case where
  # a CRAN-installed BiocManager defaulted to a newer Bioc version.
  BiocManager::install(version = "3.20", ask = FALSE, update = FALSE, force = TRUE)

  bioc_pkgs <- c("edgeR", "limma", "variancePartition", "dreamlet")
  cran_pkgs <- c("nebula")

  bioc_missing <- intersect(missing, bioc_pkgs)
  cran_missing <- intersect(missing, cran_pkgs)

  if (length(bioc_missing) > 0)
    BiocManager::install(bioc_missing, ask = FALSE, version = "3.20")
  for (pkg in cran_missing)
    install.packages(pkg, repos = "https://cloud.r-project.org")
}

# Verify all are loadable
cat("\nVerifying R packages:\n")
for (p in c("nebula", "edgeR", "limma", "dreamlet")) {
  suppressPackageStartupMessages(library(p, character.only = TRUE))
  cat("OK:", p, as.character(packageVersion(p)), "\n")
}
REOF

Rscript "$R_SETUP_SCRIPT"
rm -f "$R_SETUP_SCRIPT"

# ── 5. Restore pip numpy / final sanity check ─────────────────
echo ""
echo ">>> Step 5: Final verification"

# Reinstall the exact pip numpy/pandas — conda may have installed its own
# versions during Step 3, which can shadow the versions the Python stack
# was tested against.
pip install --force-reinstall "numpy==2.4.6" "pandas==3.0.3"

python -c "import numpy; print('numpy:', numpy.__version__)"
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
