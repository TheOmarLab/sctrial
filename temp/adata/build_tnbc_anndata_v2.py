"""
Step 1: Build AnnData from GSE169246 (TNBC) MTX files.

Processes EXACTLY like the existing 5 datasets in sctrial:
  - QC: min_genes=200, max_genes=6000, pct_mt<20, min_cells=10
  - Normalization: normalize_total(1e4) + log1p → log1p_norm layer
  - Embedding: HVG(2000, seurat) → PCA(50, HVG) → neighbors(15, 30 PCs) → UMAP
  - Clustering: Leiden(0.8)
  - Cell type annotation: _annotate_immune_celltypes (Wilcoxon markers + weighted scoring)

Output: try/GSE169246/tnbc_processed_responces.h5ad
"""
import os
import sys
import gzip
import logging
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.io import mmread
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# NOTE WILL NEED TO CHANGE ALL PATHS SO COMPATIBLE WITH SCTRIAL SCRIPTS FOR NOW OK
# Commented out belong to M scripts aka parent scripts
#DATA_DIR = Path(__file__).parent
#I
DATA_DIR = Path("/Users/valenciai/Documents/Research/projects/TNBC/data")
#OUT_H5AD = DATA_DIR / "tnbc_processed.h5ad"
OUT_H5AD = Path("/Users/valenciai/Documents/Research/projects/TNBC/outs/datatnbc_processed_responces.h5ad")

# ═══════════════════════════════════════════════════════════════════
# Clinical response data from Table S1 (Zhang et al. 2021)
# Relative change of target lesions (Post vs Pre).
# Negative = tumor shrank (Responder), >= 0 = stable/grew (Non-responder).
# Only the 12 paired patients that survive all filtering steps are listed.
# ═══════════════════════════════════════════════════════════════════
_CLINICAL = pd.DataFrame({
    "participant_id": [
        # PTX + ATZ arm
        "P019", "P012", "P017", "P002", "P005", "P016",
        # PTX arm
        "P022", "P020", "P013", "P025", "P018", "P023",
    ],
    "response": [
        # PTX + ATZ arm
        "R",  "R",  "R",  "NR", "NR", "NR",
        # PTX arm
        "R",  "R",  "R",  "R",  "R",  "NR",
    ],
    "tumor_size_change": [
        # PTX + ATZ arm (negative = shrinkage)
        -0.67, -0.46, -0.22,  0.00,  0.09,  0.17,
        # PTX arm
        -0.85, -0.55, -0.30, -0.23, -0.09,  0.03,
    ],
}).set_index("participant_id")

# ═══════════════════════════════════════════════════════════════════
# Canonical immune markers — IDENTICAL to sctrial/datasets.py
# ═══════════════════════════════════════════════════════════════════
_IMMUNE_MARKERS: dict[str, set[str]] = {
    "CD8 T cell": {
        "CD8A", "CD8B", "GZMK", "GZMB", "GZMA", "PRF1", "NKG7",
        "CD3D", "CD3E", "IFNG", "EOMES", "TBX21",
    },
    "CD4 T cell": {
        "CD4", "IL7R", "CCR7", "LEF1", "TCF7", "ICOS",
        "CD3D", "CD3E", "CD40LG",
    },
    "Treg": {
        "FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18",
        "CD4", "CD3D", "CD3E",
    },
    "B cell": {
        "MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "CD19",
        "PAX5", "BLK", "IGKC", "LTB", "LY9",
    },
    "Plasma cell": {
        "MZB1", "SDC1", "XBP1", "JCHAIN", "PRDM1", "IGKC", "IGHG1",
    },
    "NK cell": {
        "KLRD1", "KLRF1", "KLRB1", "GNLY", "PRF1", "NKG7",
        "GZMB", "NCAM1", "FCGR3A",
    },
    "Monocyte/Macrophage": {
        "CD14", "CD68", "LYZ", "CST3", "S100A8", "S100A9",
        "C1QA", "C1QB", "C1QC", "MRC1", "CSF1R", "FCGR3A",
    },
    "Dendritic cell": {
        "FCER1A", "CLEC10A", "CD1C", "ITGAX", "HLA-DRA", "HLA-DQA1",
    },
}

# Annotation parameters — IDENTICAL to sctrial/datasets.py
_ANNOT_TOP_N = 50
_ANNOT_MIN_LFC = 0.25
_ANNOT_MAX_FDR = 0.1
_ANNOT_SECOND_DELTA = 0.25
_ANNOT_MIN_ACCEPT = 0.3


def _weighted_marker_score(
    marker_df: pd.DataFrame,
    gene_set: set[str],
) -> tuple[float, list[str]]:
    """IDENTICAL to sctrial/datasets.py _weighted_marker_score."""
    hits = marker_df[marker_df["names"].isin(gene_set)]
    if hits.empty:
        return 0.0, []
    lfc = hits["logfoldchanges"].clip(lower=0).values.astype(float)
    ranks = hits["rank"].values.astype(float)
    weights = (1.0 / ranks) * np.log1p(np.exp(lfc))
    top_genes = hits["names"].values[np.argsort(-weights)].tolist()
    return float(weights.sum()), top_genes


def _annotate_immune_celltypes(adata: ad.AnnData) -> pd.Series:
    """IDENTICAL to sctrial/datasets.py _annotate_immune_celltypes.

    Assigns cell types via cluster-level Wilcoxon marker scoring.
    Expects adata.obs["leiden"] to already exist (reuses pre-computed clusters).
    """
    aw = adata.copy()

    # Normalise for clustering if raw counts/TPM
    if aw.X.max() > 50:
        aw.X = np.log1p(aw.X)

    if "leiden" in adata.obs.columns:
        aw.obs["leiden"] = adata.obs["leiden"].values
        logger.info("  Using pre-computed Leiden clusters for annotation...")
    else:
        logger.info("  Computing PCA for cell-type annotation...")
        sc.pp.highly_variable_genes(aw, n_top_genes=2000, flavor="seurat")
        sc.tl.pca(aw, n_comps=30)
        sc.pp.neighbors(aw, n_neighbors=15, n_pcs=20)
        sc.tl.leiden(aw, resolution=1.0)

    # Wilcoxon marker finding
    logger.info("  Finding cluster markers (Wilcoxon)...")
    sc.tl.rank_genes_groups(aw, groupby="leiden", method="wilcoxon", n_genes=_ANNOT_TOP_N)

    clusters = sorted(aw.obs["leiden"].unique(), key=int)
    cluster_labels: dict[str, str] = {}

    for cl in clusters:
        result = aw.uns["rank_genes_groups"]
        idx = list(result["names"].dtype.names).index(cl)
        names = [result["names"][i][idx] for i in range(len(result["names"]))]
        lfcs = [result["logfoldchanges"][i][idx] for i in range(len(result["logfoldchanges"]))]
        padjs = [result["pvals_adj"][i][idx] for i in range(len(result["pvals_adj"]))]

        df_markers = pd.DataFrame({
            "names": names,
            "logfoldchanges": lfcs,
            "pvals_adj": padjs,
            "rank": np.arange(1, len(names) + 1),
        })

        # Filter by LFC and FDR
        df_filt = df_markers[
            (df_markers["logfoldchanges"] >= _ANNOT_MIN_LFC)
            & (df_markers["pvals_adj"] <= _ANNOT_MAX_FDR)
        ]

        # If strict filtering yields no markers, use unfiltered
        if df_filt.empty:
            df_filt = df_markers.copy()

        # Score against each cell type
        label_scores: dict[str, float] = {}
        for ct, gene_set in _IMMUNE_MARKERS.items():
            score, _ = _weighted_marker_score(df_filt, gene_set)
            if score > 0:
                label_scores[ct] = score

        if not label_scores:
            # Fallback: unfiltered markers with LFC clipped to 0
            df_unfiltered = pd.DataFrame({
                "names": names,
                "logfoldchanges": [max(0, v) for v in lfcs],
                "pvals_adj": padjs,
                "rank": np.arange(1, len(names) + 1),
            })
            for ct, gene_set in _IMMUNE_MARKERS.items():
                score, _ = _weighted_marker_score(df_unfiltered, gene_set)
                if score > 0:
                    label_scores[ct] = score

        if not label_scores:
            cluster_labels[cl] = "Unassigned"
            logger.warning(f"  Cluster {cl}: no marker overlap — 'Unassigned'")
            continue

        sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])
        best_label, best_score = sorted_labels[0]
        cluster_labels[cl] = best_label

        if len(sorted_labels) > 1:
            second_label, second_score = sorted_labels[1]
            if best_score - second_score < _ANNOT_SECOND_DELTA:
                logger.info(
                    f"  Cluster {cl}: {best_label} ({best_score:.2f}) "
                    f"vs {second_label} ({second_score:.2f}) [ambiguous]"
                )

    labels = aw.obs["leiden"].map(cluster_labels)
    labels.index = adata.obs.index
    labels.name = "cell_type"

    vc = labels.value_counts()
    for ct, n in vc.items():
        logger.info(f"  {ct}: {n:,} cells")

    del aw
    gc.collect()
    return labels


# ═══════════════════════════════════════════════════════════════════
# Main processing pipeline
# ═══════════════════════════════════════════════════════════════════
def main():
    # ── 1. Load sparse matrix ─────────────────────────────────────
    logger.info("Loading MTX matrix...")
    mat = mmread(str(DATA_DIR / "GSE169246_TNBC_RNA.counts.mtx.gz")).T.tocsc()
    logger.info(f"  Raw matrix: {mat.shape[0]} cells × {mat.shape[1]} genes")

    # ── 2. Load barcodes and features ─────────────────────────────
    with gzip.open(str(DATA_DIR / "GSE169246_TNBC_RNA.barcode.tsv.gz"), "rt") as f:
        barcodes = [l.strip() for l in f]
    with gzip.open(str(DATA_DIR / "GSE169246_TNBC_RNA.feature.tsv.gz"), "rt") as f:
        features_raw = [l.strip().split("\t") for l in f]
    gene_ids = [f[0] for f in features_raw]
    gene_names = [f[1] if len(f) > 1 else f[0] for f in features_raw]

    # ── 3. Build AnnData ──────────────────────────────────────────
    adata = ad.AnnData(X=mat, obs=pd.DataFrame(index=barcodes),
                       var=pd.DataFrame(index=gene_names))
    adata.var["gene_ids"] = gene_ids
    adata.var_names_make_unique()

    # ── 4. Parse sample metadata from barcode names ───────────────
    obs = adata.obs.copy()
    obs["barcode_full"] = obs.index
    obs["sample_id"] = obs["barcode_full"].str.split(".").str[1]
    obs["timepoint_raw"] = obs["sample_id"].str.split("_").str[0]
    obs["patient_id"] = obs["sample_id"].str.extract(r"(P\d+)")
    obs["tissue_type"] = obs["sample_id"].str.split("_").str[-1]

    # Treatment arm from GEO metadata
    geo_meta = pd.read_csv(DATA_DIR / "geo_metadata.csv")
    geo_meta["treatment"] = geo_meta["treatment"].replace({"Anti-PD-L1+Chemo": "anti-PDL1+Chemo"})
    geo_meta["patient_id"] = geo_meta["title"].str.extract(r"_?(P\d+)_")
    patient_arm = geo_meta.drop_duplicates("patient_id").set_index("patient_id")["treatment"]
    obs["arm"] = obs["patient_id"].map(patient_arm)

    # ── 4b. Add clinical response data from Table S1 ──────────────
    # Maps each cell to its patient's response (R/NR) and continuous
    # tumor size change. NaN for any patient not in _CLINICAL
    # (those cells are removed by the pairing filter in step 5 anyway).
    obs["response"] = obs["patient_id"].map(_CLINICAL["response"])
    obs["tumor_size_change"] = obs["patient_id"].map(_CLINICAL["tumor_size_change"])

    adata.obs = obs
    logger.info(f"  All cells: {adata.n_obs:,}")

    # ── 5. Filter to tumor biopsies, Pre/Post only ────────────────
    adata = adata[adata.obs["tissue_type"] == "t"].copy()
    logger.info(f"  Tumor biopsies: {adata.n_obs:,}")

    adata = adata[adata.obs["timepoint_raw"].isin(["Pre", "Post"])].copy()
    logger.info(f"  Pre+Post only: {adata.n_obs:,}")

    adata.obs["visit"] = adata.obs["timepoint_raw"].astype(str)
    adata.obs["participant_id"] = adata.obs["patient_id"].astype(str)

    # Remove patients with <50 cells in any visit
    remove_pids = set()
    for pid in adata.obs["participant_id"].unique():
        for visit in ["Pre", "Post"]:
            n = ((adata.obs["participant_id"] == pid) & (adata.obs["visit"] == visit)).sum()
            if 0 < n < 50:
                remove_pids.add(pid)
                logger.warning(f"  {pid} {visit}: only {n} cells — removing")
    if remove_pids:
        adata = adata[~adata.obs["participant_id"].isin(remove_pids)].copy()

    # Keep only paired patients
    paired = set()
    for pid in adata.obs["participant_id"].unique():
        visits = set(adata.obs.loc[adata.obs["participant_id"] == pid, "visit"].unique())
        if {"Pre", "Post"}.issubset(visits):
            paired.add(pid)
    adata = adata[adata.obs["participant_id"].isin(paired)].copy()
    logger.info(f"  Paired patients: {len(paired)}")
    logger.info(f"  Final cells: {adata.n_obs:,}")

    # ── 6. QC — IDENTICAL to CAR-T/AML pipeline ──────────────────
    logger.info("QC filtering...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info(f"  QC: {n_before:,} → {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    # ── 7. Normalization — IDENTICAL to CAR-T/AML pipeline ────────
    logger.info("Normalizing (normalize_total 1e4 + log1p)...")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()

    # ── 8. Embedding — IDENTICAL to CAR-T pipeline ────────────────
    logger.info("Computing embeddings (HVG→PCA→neighbors→UMAP→Leiden)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=False)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    # ── 9. Clustering — IDENTICAL to CAR-T pipeline ───────────────
    logger.info("Leiden clustering (resolution=0.8)...")
    sc.tl.leiden(adata, resolution=0.8)

    # ── 10. Cell type annotation — IDENTICAL pipeline ─────────────
    logger.info("Annotating cell types (Wilcoxon + weighted scoring)...")
    # Use log1p_norm for annotation (same as the stored .X)
    adata.obs["cell_type"] = _annotate_immune_celltypes(adata)

    # ── 11. Summary ───────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL DATASET SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Cells: {adata.n_obs:,}")
    logger.info(f"  Genes: {adata.n_vars:,}")
    logger.info(f"  Patients: {adata.obs['participant_id'].nunique()}")
    logger.info(f"  Arms: {dict(adata.obs['arm'].value_counts())}")
    logger.info(f"  Visits: {dict(adata.obs['visit'].value_counts())}")
    logger.info(f"  Layers: {list(adata.layers.keys())}")
    logger.info(f"  Cell types:")
    for ct, n in adata.obs["cell_type"].value_counts().items():
        logger.info(f"    {ct}: {n:,} cells")

    # NEW: Response summary
    logger.info(f"\n  Response by arm:")
    for arm in sorted(adata.obs["arm"].dropna().unique()):
        arm_pids = adata.obs[adata.obs["arm"] == arm]["participant_id"].unique()
        for pid in sorted(arm_pids):
            pid_obs = adata.obs[adata.obs["participant_id"] == pid]
            r  = pid_obs["response"].iloc[0]
            tc = pid_obs["tumor_size_change"].iloc[0]
            logger.info(f"    {pid} ({arm}): {r}  tumor_size_change={tc:+.0%}")

    logger.info(f"\n  Cells per patient-visit:")
    for pid in sorted(adata.obs["participant_id"].unique()):
        arm = adata.obs.loc[adata.obs["participant_id"] == pid, "arm"].iloc[0]
        r   = adata.obs.loc[adata.obs["participant_id"] == pid, "response"].iloc[0]
        for v in ["Pre", "Post"]:
            n = ((adata.obs["participant_id"] == pid) & (adata.obs["visit"] == v)).sum()
            logger.info(f"    {pid} ({arm}, {r}) {v}: {n:,}")

    # ── 12. Verify no NaN response in final paired cohort ─────────
    n_missing = adata.obs["response"].isna().sum()
    if n_missing > 0:
        missing_pids = adata.obs[adata.obs["response"].isna()]["participant_id"].unique()
        logger.warning(f"\n  WARNING: {n_missing} cells have no response label.")
        logger.warning(f"  Missing patient IDs: {sorted(missing_pids)}")
        logger.warning(f"  Add these patients to _CLINICAL at the top of this script.")
    else:
        logger.info(f"\n  All {adata.n_obs:,} cells have a valid response label. OK")

    # ── 13. Save ──────────────────────────────────────────────────
    adata.write(OUT_H5AD)
    logger.info(f"\nSaved to {OUT_H5AD}")


if __name__ == "__main__":
    main()
