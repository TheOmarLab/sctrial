#!/bin/bash
# ============================================================
# HPC environment setup for sctrial benchmark — Python only
#
# Run once interactively on the HPC login node:
#   bash scripts/setup_hpc_env.sh
#
# What this does:
#   1. Creates the conda env "sctrial" (Python packages only)
#   2. Installs sctrial from source (dev version)
#
# R and Bioconductor packages are installed separately via:
#   sbatch scripts/install_r.sh
# ============================================================

set -euo pipefail

# ── 0. Configuration ─────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="$REPO_ROOT/src/sctrial/benchmark/sctrial_bench_environment.yml"
ENV_NAME="sctrial"   # must match the slurm scripts, which all run `micromamba run -n sctrial`

echo "============================================"
echo "sctrial HPC environment setup (Python)"
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
pip install -e "$REPO_ROOT[plots,gsea]"

python -c "import sctrial; print('sctrial version:', sctrial.__version__)"
python -c "
from sctrial.benchmark.orchestrator import build_sensitivity_grid
print('Sensitivity grid:', len(build_sensitivity_grid('two_arm')), 'scenarios')
"
python -c "from sctrial.benchmark import TranscriptomeSimConfig; print('Python benchmark: OK')"

echo ""
echo "============================================"
echo "Python setup complete: $(date)"
echo ""
echo "Next steps:"
echo "  1. Install R packages (run once, ~1-2h):"
echo "       sbatch scripts/install_r.sh"
echo ""
echo "  2. Sensitivity benchmark (panels C-F):"
echo "       sbatch scripts/slurm_calibrate.sh targets   # calibrate first"
echo "       sbatch scripts/slurm_smoke.sh               # then smoke test"
echo "       sbatch scripts/slurm_benchmark.sh core 200 two_arm"
echo ""
echo "  3. Main simulation benchmark:"
echo "       Edit scripts/slurm_benchmark.sh — update the 'cd' path"
echo "       sbatch scripts/slurm_benchmark.sh"
echo "============================================"
