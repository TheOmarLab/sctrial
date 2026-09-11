"""
Compare our cell type annotations vs Zhang et al. 2021 author annotations.

Produces:
  1. Side-by-side UMAPs (our vs author)
  2. Confusion table printed to stdout
  3. Cell type composition per patient × visit (Pre / Post)
  4. Cell type composition per response (R vs NR / PR+SD vs PD)
  5. Cell type composition per treatment arm
  — all as paired panels (our annotation left, author annotation right)
"""
import anndata as ad
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
from pathlib import Path

PROCESSED_H5AD = Path("datasets/tnbc_zhang/processed/tnbc_zhang_processed.h5ad")
MMC3_XLSX      = Path("datasets/tnbc_zhang/raw/mmc3.xlsx")
OUT_DIR        = Path("scripts/compare_tnbc_figs")
OUT_DIR.mkdir(exist_ok=True)

# ── Load processed AnnData ────────────────────────────────────────────────────
print("Loading processed AnnData...")
adata = ad.read_h5ad(PROCESSED_H5AD)
print(f"  {adata.n_obs:,} cells, UMAP present: {'X_umap' in adata.obsm}")

# ── Load author annotations ───────────────────────────────────────────────────
print("Loading author annotations from mmc3.xlsx...")
author = pd.read_excel(MMC3_XLSX, sheet_name="Single cell clustering", header=1)
author = author.rename(columns={
    "Cell barcode":   "barcode",
    "Major celltype": "author_major",
    "Cluster":        "author_cluster",
    "Group":          "author_visit",
    "Treatment":      "author_arm",
    "Efficacy":       "author_efficacy",
    "Patient":        "author_patient",
})
author = author.set_index("barcode")[[
    "author_major", "author_cluster",
    "author_visit", "author_arm", "author_efficacy", "author_patient",
]]
print(f"  {len(author):,} cells in mmc3.xlsx")

# ── Join by barcode ───────────────────────────────────────────────────────────
n_before = adata.n_obs
adata.obs = adata.obs.join(author, how="left")
matched = adata.obs["author_major"].notna().sum()
print(f"  Matched {matched:,} / {n_before:,} cells to author annotations")
print(f"  Unmatched: {n_before - matched:,}")

adata.obs["author_major"]   = adata.obs["author_major"].fillna("Unmatched")
adata.obs["author_cluster"] = adata.obs["author_cluster"].fillna("Unmatched")

# Map author visit labels to match ours (Pre / Post)
visit_map = {"Pre-treatment": "Pre", "Post-treatment": "Post", "Progression": "Progression"}
adata.obs["author_visit_mapped"] = adata.obs["author_visit"].map(visit_map).fillna("Unknown")


# ── Helper: paired stacked-bar composition plot ───────────────────────────────
def stacked_bar_pair(
    obs: pd.DataFrame,
    group_col: str,
    our_col: str,
    author_col: str,
    title: str,
    out_path: Path,
    figwidth: float = 14,
    author_matched_only: bool = True,
):
    """
    Two side-by-side stacked bars:
      left  — our annotation (our_col) grouped by group_col
      right — author annotation (author_col) grouped by group_col,
              restricted to cells that were matched (author_col != 'Unmatched')
    """
    obs_ours = obs.copy()
    obs_auth = obs[obs[author_col] != "Unmatched"].copy() if author_matched_only else obs.copy()

    def make_frac(df, ct_col):
        counts = (
            df.groupby([group_col, ct_col], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        return counts.div(counts.sum(axis=1), axis=0)

    frac_ours = make_frac(obs_ours, our_col)
    frac_auth = make_frac(obs_auth, author_col)

    n_groups = max(len(frac_ours), len(frac_auth))
    fig, axes = plt.subplots(1, 2, figsize=(figwidth, max(4, n_groups * 0.45 + 2)))

    for ax, frac, label in [
        (axes[0], frac_ours, "Our annotation"),
        (axes[1], frac_auth, "Author annotation (Zhang et al.)"),
    ]:
        frac.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="none", legend=True)
        ax.set_ylabel("Fraction of cells")
        ax.set_title(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {out_path.name}")


# ── 1. UMAP side-by-side ─────────────────────────────────────────────────────
print("\nPlotting UMAPs...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sc.pl.umap(adata, color="cell_type",    title="Our annotation",
           ax=axes[0], show=False, legend_loc="on data", legend_fontsize=8)
sc.pl.umap(adata, color="author_major", title="Author annotation (Zhang et al. 2021)",
           ax=axes[1], show=False, legend_loc="on data", legend_fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "01_umap_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → saved 01_umap_comparison.png")

# ── 2. Confusion table ────────────────────────────────────────────────────────
print("\nConfusion table (our cell_type vs author major celltype):")
confusion = pd.crosstab(
    adata.obs["cell_type"],
    adata.obs["author_major"],
    margins=True,
)
print(confusion.to_string())

print("\nAuthor cluster breakdown per our cell_type label:")
for ct in adata.obs["cell_type"].unique():
    sub = adata.obs[adata.obs["cell_type"] == ct]["author_cluster"].value_counts().head(5)
    print(f"\n  {ct}:")
    for cluster, n in sub.items():
        print(f"    {cluster}: {n:,}")

# ── 3. Composition per patient × visit ───────────────────────────────────────
print("\nPlotting composition per patient × visit...")
obs = adata.obs.copy()
# Cast categoricals to str so string operations work
for col in obs.select_dtypes("category").columns:
    obs[col] = obs[col].astype(str)

for visit in ["Pre", "Post"]:
    sub = obs[obs["visit"] == visit].copy()
    sub_auth = sub[sub["author_visit_mapped"] == visit].copy()

    # Use participant_id as the grouping axis (cleaner labels)
    frac_ours = (
        sub.groupby(["participant_id", "cell_type"], observed=True)
        .size().unstack(fill_value=0)
        .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
    )
    frac_auth = (
        sub_auth[sub_auth["author_major"] != "Unmatched"]
        .groupby(["participant_id", "author_major"], observed=True)
        .size().unstack(fill_value=0)
        .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
    )

    n_pat = max(len(frac_ours), len(frac_auth))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_pat * 0.5 + 2)))

    for ax, frac, label in [
        (axes[0], frac_ours, "Our annotation"),
        (axes[1], frac_auth, "Author annotation"),
    ]:
        frac.plot(kind="bar", stacked=True, ax=ax, colormap="tab20",
                  edgecolor="none", legend=True)
        ax.set_ylabel("Fraction of cells")
        ax.set_title(label)
        ax.set_xlabel("Patient")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(f"Cell type composition per patient — {visit}", fontsize=12, y=1.01)
    plt.tight_layout()
    fname = f"02_composition_per_patient_{visit}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {fname}")

# ── 4. Composition per response ───────────────────────────────────────────────
print("\nPlotting composition per response...")

# Our response: R / NR  — use participant-level (not cell-level) to avoid
# cell-count imbalance dominating; also show raw cell-level for transparency
obs["response_label"] = obs["response"].fillna("Unknown")

# Author efficacy: map to a binary label comparable to ours
# PR (partial response) + SD (stable disease) ≈ R; PD (progressive disease) ≈ NR
efficacy_map = {"PR": "PR/SD (≈R)", "SD": "PR/SD (≈R)", "PD": "PD (≈NR)", "Na": "Na"}
obs["author_efficacy_mapped"] = obs["author_efficacy"].map(efficacy_map).fillna("Unknown")

stacked_bar_pair(
    obs=obs,
    group_col="response_label",
    our_col="cell_type",
    author_col="author_major",
    title="Cell type composition per response (R vs NR)",
    out_path=OUT_DIR / "03_composition_per_response.png",
    figwidth=12,
)

# Also show author efficacy grouping
stacked_bar_pair(
    obs=obs[obs["author_efficacy_mapped"] != "Unknown"].copy(),
    group_col="author_efficacy_mapped",
    our_col="cell_type",
    author_col="author_major",
    title="Cell type composition per author efficacy (PR/SD vs PD)",
    out_path=OUT_DIR / "04_composition_per_efficacy.png",
    figwidth=12,
)

# ── 5. Composition per treatment arm ─────────────────────────────────────────
print("\nPlotting composition per treatment arm...")

obs["arm_label"] = obs["arm"].fillna("Unknown")
obs["author_arm_label"] = obs["author_arm"].fillna("Unknown")

stacked_bar_pair(
    obs=obs,
    group_col="arm_label",
    our_col="cell_type",
    author_col="author_major",
    title="Cell type composition per treatment arm",
    out_path=OUT_DIR / "05_composition_per_arm.png",
    figwidth=12,
)

# ── 6. Composition per arm × response (2×2 grid) ─────────────────────────────
print("\nPlotting composition per arm × response...")
obs["arm_response"] = obs["arm_label"] + " / " + obs["response_label"]

stacked_bar_pair(
    obs=obs[obs["response_label"] != "Unknown"].copy(),
    group_col="arm_response",
    our_col="cell_type",
    author_col="author_major",
    title="Cell type composition per arm × response",
    out_path=OUT_DIR / "06_composition_per_arm_response.png",
    figwidth=14,
)

print(f"\nAll figures saved to: {OUT_DIR.resolve()}/")
print("  01_umap_comparison.png                — UMAP: our vs author")
print("  02_composition_per_patient_Pre.png    — per patient, Pre visit")
print("  02_composition_per_patient_Post.png   — per patient, Post visit")
print("  03_composition_per_response.png       — R vs NR (our labels)")
print("  04_composition_per_efficacy.png       — PR/SD vs PD (author efficacy)")
print("  05_composition_per_arm.png            — treatment arm")
print("  06_composition_per_arm_response.png   — arm × response")
