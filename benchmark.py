import anndata as ad
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/common/omarmlab/members/itzel/sctrial_bench/sctrial/temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load TNBC data ────────────────────────────
print("Loading TNBC h5ad...")
adata = ad.read_h5ad(
    "/common/omarmlab/members/itzel/sctrial_bench/h5ad/datatnbc_processed_responces.h5ad"
)
print(f"  Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# ── Swap X to raw counts for NEBULA ──────────
adata.X = adata.layers["counts"]
print("  Swapped adata.X to raw counts for NEBULA")

# ── Select genes ──────────────────────────────
if "highly_variable" in adata.var.columns:
    gene_cols = adata.var_names[adata.var["highly_variable"]].tolist()
    print(f"  Using {len(gene_cols)} highly variable genes")
else:
    gene_cols = adata.var["total_counts"].nlargest(2000).index.tolist()
    print(f"  No highly_variable flag found — using top {len(gene_cols)} genes by counts")

# ── Column names from TrialDesign ─────────────
PARTICIPANT_COL = "participant_id"
VISIT_COL       = "visit"
ARM_COL         = "arm"
TREATED_LABEL   = "anti-PDL1+Chemo"
CONTROL_LABEL   = "Chemo"

# ── Sanity check obs columns ──────────────────
for col in [PARTICIPANT_COL, VISIT_COL, ARM_COL]:
    assert col in adata.obs.columns, f"Missing expected column: {col}"
print(f"  Participants: {adata.obs[PARTICIPANT_COL].nunique()}")
print(f"  Visits:       {adata.obs[VISIT_COL].unique().tolist()}")
print(f"  Arms:         {adata.obs[ARM_COL].unique().tolist()}")

# ── 1. Permutation test ───────────────────────
print("\n==========================================")
print("STEP 1: Permutation test")
print("==========================================")
from sctrial.benchmark.permutation import run_permutation_test

perm_df = run_permutation_test(
    adata,
    gene_cols=gene_cols,
    design_type="two_arm",
    n_permutations=1000,
    n_jobs=25,
    participant_col=PARTICIPANT_COL,
    arm_col=ARM_COL,
    visit_col=VISIT_COL,
    treated_label=TREATED_LABEL,
    control_label=CONTROL_LABEL,
    output_path=OUTPUT_DIR / "tnbc_permutation_results.csv",
    seed=42,
)
print(f"Permutation done: {len(perm_df):,} rows")

# ── 2. Subsampling ────────────────────────────
print("\n==========================================")
print("STEP 2: Subsampling reproducibility")
print("==========================================")
from sctrial.benchmark.subsampling import run_subsampling

sub_df = run_subsampling(
    adata,
    gene_cols=gene_cols,
    fractions=[0.5, 0.7, 0.9],
    n_resamples=100,
    participant_col=PARTICIPANT_COL,
    arm_col=ARM_COL,
    visit_col=VISIT_COL,
    treated_label=TREATED_LABEL,
    control_label=CONTROL_LABEL,
    output_path=OUTPUT_DIR / "tnbc_subsampling_results.csv",
    seed=42,
)
print(f"Subsampling done: {len(sub_df):,} rows")

print("\n==========================================")
print("ALL DONE")
print(f"  Permutation CSV → {OUTPUT_DIR}/tnbc_permutation_results.csv")
print(f"  Subsampling CSV → {OUTPUT_DIR}/tnbc_subsampling_results.csv")
print("==========================================")