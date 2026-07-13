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
if conda env list | grep -q "^$ENV_NAME "; then
    echo "    Environment '$ENV_NAME' already exists — updating (resuming)."
    conda env update --name "$ENV_NAME" --file "$ENV_YML" --prune
else
    conda env create --name "$ENV_NAME" --file "$ENV_YML" --yes
fi

# Activate — source activate works universally on HPC without extra shell hooks.
source activate "$ENV_NAME"

# Print the environment path so you know exactly where packages are installed.
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

# ── 3. Install R ─────────────────────────────────────────────
echo ""
echo ">>> Step 3: Installing R into the conda environment"
if command -v Rscript &>/dev/null; then
    echo "    R already installed: $(Rscript --version 2>&1 | head -1)"
else
    conda install --name "$ENV_NAME" -c conda-forge r-base --yes
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
options(repos = c(
  CRAN    = "https://cloud.r-project.org",
  BioCsoft = "https://bioconductor.org/packages/3.20/bioc",
  BioCann  = "https://bioconductor.org/packages/3.20/data/annotation"
))

install_if_missing <- function(pkg, bioc = FALSE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing:", pkg, "\n"); flush.console()
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
      BiocManager::install(pkg, ask = FALSE, update = FALSE)
    } else {
      install.packages(pkg)
    }
  } else {
    cat("Already installed:", pkg, "\n")
  }
}

# CRAN
install_if_missing("Matrix")
install_if_missing("lme4")
install_if_missing("nloptr")
install_if_missing("pbkrtest")
install_if_missing("lmerTest")
install_if_missing("nebula")

# Bioconductor
install_if_missing("BiocManager")
install_if_missing("edgeR",             bioc = TRUE)
install_if_missing("limma",             bioc = TRUE)
install_if_missing("variancePartition", bioc = TRUE)
install_if_missing("dreamlet",          bioc = TRUE)

# Verify all are loadable
pkgs <- c("Matrix", "nebula", "edgeR", "limma", "dreamlet")
for (p in pkgs) {
  suppressPackageStartupMessages(library(p, character.only = TRUE))
  cat("OK:", p, packageVersion(p), "\n")
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
