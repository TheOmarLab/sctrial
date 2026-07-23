"""Built-in dataset loaders for clinical trial scRNA-seq cohorts."""

from __future__ import annotations

import gc
import gzip
import logging
import re
import tarfile
import urllib.error
import urllib.request
import warnings
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from statsmodels.stats.multitest import multipletests

from .preprocessing import drop_artifact_genes
from .utils import get_counts_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset root: resolved relative to the *repository* (two levels up from
# src/sctrial/), so loaders work regardless of the caller's cwd.
# When installed as a proper package (no repo checkout), falls back to cwd.
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent  # src/sctrial/
_REPO_ROOT = _PACKAGE_DIR.parent.parent  # sc_trial_inference/
_DATASETS_ROOT = _REPO_ROOT / "datasets"
if not _DATASETS_ROOT.is_dir():
    # Installed package without repo structure — fall back to cwd
    _DATASETS_ROOT = Path.cwd() / "datasets"


def _default_data_dir(name: str) -> str:
    """Return the absolute default data_dir for a given dataset name."""
    return str(_DATASETS_ROOT / name)


# ---------------------------------------------------------------------------
# Marker-based cell-type annotation for immune cells
# Approach adapted from Diab_duod project: Leiden clustering -> Wilcoxon
# marker finding -> weighted scoring against canonical marker gene sets
# ---------------------------------------------------------------------------

# Immune cell markers from Sade-Feldman et al. (Cell 2018, Fig 1, Table S1).
# The paper identifies 11 clusters (G1-G11) among CD45+ sorted cells:
#   G1: B cells, G2: Plasma cells, G3: Monocytes/Macrophages,
#   G4: Dendritic cells, G5-G11: T/NK/NKT subtypes.
# CD8 T cells are further split into memory-like (CD8_G) and
# exhausted (CD8_B) states.
_IMMUNE_MARKERS: dict[str, set[str]] = {
    "CD8 T cell": {
        "CD8A",
        "CD8B",
        "GZMK",
        "GZMB",
        "GZMA",
        "PRF1",
        "NKG7",
        "CD3D",
        "CD3E",
        "IFNG",
        "EOMES",
        "TBX21",
    },
    "CD4 T cell": {
        "CD4",
        "IL7R",
        "CCR7",
        "LEF1",
        "TCF7",
        "ICOS",
        "CD3D",
        "CD3E",
        "CD40LG",
    },
    "Treg": {
        "FOXP3",
        "IL2RA",
        "CTLA4",
        "IKZF2",
        "TNFRSF18",
        "CD4",
        "CD3D",
        "CD3E",
    },
    "B cell": {
        # G1 markers from publication: IGKC, LTB, LY9, SELL, TCF7, CCR7
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "CD74",
        "CD19",
        "PAX5",
        "BLK",
        "IGKC",
        "LTB",
        "LY9",
    },
    "Plasma cell": {
        # G2 cluster
        "MZB1",
        "SDC1",
        "XBP1",
        "JCHAIN",
        "PRDM1",
        "IGKC",
        "IGHG1",
    },
    "NK cell": {
        "KLRD1",
        "KLRF1",
        "KLRB1",
        "GNLY",
        "PRF1",
        "NKG7",
        "GZMB",
        "NCAM1",
        "FCGR3A",
    },
    "Monocyte/Macrophage": {
        # G3 cluster
        "CD14",
        "CD68",
        "LYZ",
        "CST3",
        "S100A8",
        "S100A9",
        "C1QA",
        "C1QB",
        "C1QC",
        "MRC1",
        "CSF1R",
        "FCGR3A",
    },
    "Dendritic cell": {
        # G4 cluster
        "FCER1A",
        "CLEC10A",
        "CD1C",
        "ITGAX",
        "HLA-DRA",
        "HLA-DQA1",
    },
}

# Annotation parameters (following Diab_duod conventions)
_ANNOT_TOP_N = 50  # top markers per cluster from Wilcoxon
_ANNOT_MIN_LFC = 0.25  # minimum log fold-change
_ANNOT_MAX_FDR = 0.1  # maximum adjusted p-value
_ANNOT_SECOND_DELTA = 0.25  # delta to flag ambiguous clusters
_ANNOT_MIN_ACCEPT = 0.3  # minimum weighted score to accept a label

_TNBC_EFFICACY_MAP = {"CR": "R", "PR": "R", "PD": "NR", "SD": "NR"}


def _load_tnbc_clinical(raw_dir: Path) -> pd.Series:
    """Read per-patient response from mmc3.xlsx (Zhang et al. 2021, Table S2).

    Returns a Series indexed by patient ID (e.g. 'P001') with values 'R' or 'NR'.
    CR/PR → R (responder), PD/SD → NR (non-responder), Na → NaN.
    """
    mmc3_path = raw_dir / "mmc3.xlsx"
    if not mmc3_path.exists():
        raise FileNotFoundError(
            f"mmc3.xlsx not found at {mmc3_path}. "
            "Download it from the Zhang et al. 2021 paper supplementary materials "
            "and place it in the raw data directory."
        )
    df = pd.read_excel(mmc3_path, sheet_name="Single cell clustering", header=1)
    pt_eff = (
        df[["Patient", "Efficacy"]]
        .drop_duplicates("Patient")
        .set_index("Patient")["Efficacy"]
    )
    return pt_eff.map(_TNBC_EFFICACY_MAP)


def _weighted_marker_score(
    marker_df: pd.DataFrame,
    gene_set: set[str],
) -> tuple[float, list[str]]:
    """Compute weighted score of a gene set against ranked cluster markers.

    Weight per gene = (1 / rank) * log1p(exp(clipped_logFC)).
    Mirrors the ``weighted_score`` function from the Diab_duod project.
    """
    hits = marker_df[marker_df["names"].isin(gene_set)]
    if hits.empty:
        return 0.0, []
    lfc = hits["logfoldchanges"].clip(lower=0).values.astype(float)
    ranks = hits["rank"].values.astype(float)
    weights = (1.0 / ranks) * np.log1p(np.exp(lfc))
    top_genes = hits["names"].values[np.argsort(-weights)].tolist()
    return float(weights.sum()), top_genes


def _annotate_immune_celltypes(adata: ad.AnnData) -> pd.Series:
    """Assign cell types to immune cells via cluster-level marker scoring.

    Pipeline (adapted from Diab_duod project):
    1. Use pre-computed Leiden clusters (from caller), or compute them here.
    2. Find differentially expressed markers per cluster (Wilcoxon).
    3. Score each cluster against canonical immune marker sets using
       a rank-weighted scoring function.
    4. Assign the best-scoring cell type to each cluster.

    Parameters
    ----------
    adata : AnnData
        Must contain expression values (TPM) in ``adata.X`` and gene names
        in ``adata.var_names``.  If ``adata.obs["leiden"]`` already exists
        (with PCA/neighbors pre-computed), those clusters are reused so that
        annotation and UMAP share the same embedding.

    Returns
    -------
    pd.Series
        Cell-type labels indexed like ``adata.obs``.
    """
    import scanpy as sc  # local import to avoid top-level dependency

    # Work on a copy so we don't modify the caller's object
    aw = adata.copy()

    # Normalise for clustering if raw TPM
    if aw.X.max() > 50:
        aw.X = np.log1p(aw.X)

    if "leiden" in adata.obs.columns:
        # Reuse pre-computed Leiden clusters (same embedding used for UMAP)
        aw.obs["leiden"] = adata.obs["leiden"].values
        logger.info("    Using pre-computed Leiden clusters for annotation...")
    else:
        # Fallback: compute PCA -> neighbors -> Leiden internally
        logger.info("    Computing PCA for cell-type annotation...")
        sc.pp.highly_variable_genes(aw, n_top_genes=2000, flavor="seurat")
        sc.tl.pca(aw, n_comps=30)
        sc.pp.neighbors(aw, n_neighbors=15, n_pcs=20)
        sc.tl.leiden(aw, resolution=1.0)

    # Wilcoxon marker finding per cluster
    logger.info("    Finding cluster markers (Wilcoxon)...")
    sc.tl.rank_genes_groups(aw, groupby="leiden", method="wilcoxon", n_genes=_ANNOT_TOP_N)

    clusters = sorted(aw.obs["leiden"].unique(), key=int)
    cluster_labels: dict[str, str] = {}

    for cl in clusters:
        # Extract marker table for this cluster
        result = aw.uns["rank_genes_groups"]
        idx = list(result["names"].dtype.names).index(cl)
        names = [result["names"][i][idx] for i in range(len(result["names"]))]
        lfcs = [result["logfoldchanges"][i][idx] for i in range(len(result["logfoldchanges"]))]
        padjs = [result["pvals_adj"][i][idx] for i in range(len(result["pvals_adj"]))]

        df_markers = pd.DataFrame(
            {
                "names": names,
                "logfoldchanges": lfcs,
                "pvals_adj": padjs,
                "rank": np.arange(1, len(names) + 1),
            }
        )

        # Filter by LFC and FDR
        df_markers = df_markers[
            (df_markers["logfoldchanges"] >= _ANNOT_MIN_LFC)
            & (df_markers["pvals_adj"] <= _ANNOT_MAX_FDR)
        ]

        # If strict filtering yields no markers, use unfiltered markers
        if df_markers.empty:
            df_markers = pd.DataFrame(
                {
                    "names": names,
                    "logfoldchanges": lfcs,
                    "pvals_adj": padjs,
                    "rank": np.arange(1, len(names) + 1),
                }
            )

        # Score against each cell type
        label_scores: dict[str, float] = {}
        for ct, gene_set in _IMMUNE_MARKERS.items():
            score, _ = _weighted_marker_score(df_markers, gene_set)
            if score > 0:
                label_scores[ct] = score

        if not label_scores:
            # Fallback: score with unfiltered markers (no LFC/FDR filter)
            df_unfiltered = pd.DataFrame(
                {
                    "names": names,
                    "logfoldchanges": [max(0, v) for v in lfcs],
                    "pvals_adj": padjs,
                    "rank": np.arange(1, len(names) + 1),
                }
            )
            for ct, gene_set in _IMMUNE_MARKERS.items():
                score, _ = _weighted_marker_score(df_unfiltered, gene_set)
                if score > 0:
                    label_scores[ct] = score

        if not label_scores:
            cluster_labels[cl] = "Unassigned"
            logger.warning(
                f"    Cluster {cl}: no marker overlap with any canonical "
                f"gene set — labelled 'Unassigned'"
            )
            continue

        sorted_labels = sorted(label_scores.items(), key=lambda x: -x[1])
        best_label, best_score = sorted_labels[0]

        # Always assign the best-scoring type (no "Unknown immune")
        cluster_labels[cl] = best_label

        # Log ambiguous clusters
        if len(sorted_labels) > 1:
            second_label, second_score = sorted_labels[1]
            if best_score - second_score < _ANNOT_SECOND_DELTA:
                logger.debug(
                    f"    Cluster {cl}: {best_label} ({best_score:.2f}) "
                    f"vs {second_label} ({second_score:.2f}) [ambiguous]"
                )

    # Map cluster -> cell type onto every cell
    labels = aw.obs["leiden"].map(cluster_labels)
    labels.index = adata.obs.index
    labels.name = "cell_type"

    # Log summary
    vc = labels.value_counts()
    for ct, n in vc.items():
        logger.info(f"    {ct}: {n:,} cells")

    del aw
    import gc

    gc.collect()

    return labels


__all__ = [
    "load_sade_feldman",
    "load_stephenson_data",
    "load_vaccine_gse171964",
    "load_aml",
    "load_cart",
    "load_tnbc_zhang",
    "harmonize_response",
    "count_paired",
    "verify_paired_participants",
    "categorize_celltype",
    "ensure_fdr",
]


def _resolve_dir_with_files(p: str, required_files: Sequence[str]) -> Path:
    """Resolve a directory path with required files."""
    path = Path(p)
    if path.is_absolute():
        if all((path / f).exists() for f in required_files):
            return path
    for base in [Path.cwd(), *Path.cwd().parents]:
        cand = base / path
        if all((cand / f).exists() for f in required_files):
            return cand
    return path


def _resolve_file(p: str) -> Path:
    path = Path(p)
    if path.is_absolute() and path.exists():
        return path
    for base in [Path.cwd(), *Path.cwd().parents]:
        cand = base / path
        if cand.exists():
            return cand
    return path


def _params_match(prev: dict, current: dict) -> bool:
    """Robustly compare processing parameters, handling None/NaN/list differences.

    Returns True if every key in *current* has a matching value in *prev*.
    Extra keys in *prev* (e.g. metadata added later) are tolerated.
    """
    if not set(current.keys()).issubset(set(prev.keys())):
        return False
    for key in current:
        v1, v2 = prev.get(key), current.get(key)
        # Handle None comparisons (h5ad may store None differently)
        if v1 is None or (isinstance(v1, float) and np.isnan(v1)):
            v1 = None
        if v2 is None or (isinstance(v2, float) and np.isnan(v2)):
            v2 = None
        # Handle string "None"
        if isinstance(v1, str) and v1 in ("None", "null"):
            v1 = None
        if isinstance(v2, str) and v2 in ("None", "null"):
            v2 = None
        # Handle list/array comparisons (h5ad may convert lists to numpy arrays)
        if isinstance(v1, (list, tuple, np.ndarray)) and isinstance(v2, (list, tuple, np.ndarray)):
            if list(v1) != list(v2):
                return False
            continue
        if v1 != v2:
            return False
    return True


def _looks_log1p(X, sample: int = 10000, seed: int = 0) -> bool:
    """Check if a matrix looks like log-transformed counts."""
    if X is None:
        return False
    if sp.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()
    if data.size == 0:
        return False
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    rng = np.random.default_rng(seed)
    if data.size > sample:
        data = rng.choice(data, size=sample, replace=False)
    return (
        (data.min() >= 0)
        and (data.max() < 50)
        and (not np.allclose(data, np.round(data), atol=1e-3))
    )


def _download_file(url: str, dest: Path, label: str = "file") -> None:
    """Download a single file with error handling and partial-file cleanup.

    Parameters
    ----------
    url : str
        URL to download from.
    dest : Path
        Local destination path.
    label : str
        Human-readable label for log messages (e.g. "TPM file").
    """
    logger.info(f"Downloading {label} from {url}...")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(
            f"Failed to download {label} from {url}: {e}. "
            f"Please download manually and place it in {dest.parent}"
        ) from e
    logger.info(f"Successfully downloaded {label}: {dest}")


def _get_counts_matrix(adata: ad.AnnData) -> tuple[np.ndarray | None, str | None]:
    """Get the counts matrix from the AnnData object."""
    return get_counts_matrix(adata)


def load_sade_feldman(
    data_dir: str | None = None,
    processed_name: str = "sade_feldman_processed_v6.h5ad",
    max_cells_per_participant_visit: int | None = None,
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load and preprocess Sade-Feldman melanoma immunotherapy dataset (GSE120575).

    Parameters
    ----------
    data_dir : str
        Directory containing (or to store) the raw data files.
    processed_name : str
        Filename for the cached processed h5ad file.
    max_cells_per_participant_visit : int or None
        Maximum number of cells to retain per participant-visit pair.
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download missing files from GEO automatically.
    force_reprocess : bool
        If True, reprocess even when a cached file exists.

    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    processing_params = {
        "version": "v7",
        "max_cells_per_participant_visit": max_cells_per_participant_visit,
        "seed": seed,
        "assay": "TPM",
    }

    data_dir = data_dir or _default_data_dir("sade_feldman")
    data_dir_path = Path(data_dir)
    processed_path = data_dir_path / "processed" / processed_name

    if not force_reprocess and processed_path.exists():
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if prev:
            if _params_match(prev, processing_params):
                logger.info(
                    f"Loaded processed Sade-Feldman dataset: {adata.n_obs:,} cells, {adata.n_vars:,} genes"
                )
                return adata
            logger.info("Processed file parameters differ; reprocessing.")
            logger.debug(f"  Stored: {prev}")
            logger.debug(f"  Current: {processing_params}")
        else:
            warnings.warn(
                "Cached file lacks processing_params metadata; cannot verify it matches "
                "current settings. Consider reprocessing with force_reprocess=True.",
                UserWarning,
                stacklevel=2,
            )
            logger.info(
                f"Loaded processed Sade-Feldman dataset: {adata.n_obs:,} cells, {adata.n_vars:,} genes"
            )
            return adata

    raw_dir = data_dir_path / "raw"
    raw_dir_resolved = _resolve_dir_with_files(
        str(raw_dir),
        [
            "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
            "GSE120575_patient_ID_single_cells.txt.gz",
        ],
    )

    # Check scanpy availability BEFORE downloading to avoid wasted bandwidth.
    # scanpy is required for cell-type annotation (Leiden + Wilcoxon scoring).
    try:
        import scanpy as sc  # noqa: F401
    except ImportError:
        raise ImportError(
            "scanpy is required for Sade-Feldman cell type annotation. "
            "Install with: pip install sctrial[plots]  or  pip install scanpy"
        ) from None

    tpm_path = raw_dir_resolved / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
    meta_path = raw_dir_resolved / "GSE120575_patient_ID_single_cells.txt.gz"

    _GEO_BASE = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120575&format=file&file="
    _sade_feldman_files = [
        (
            tpm_path,
            _GEO_BASE
            + "GSE120575%5FSade%5FFeldman%5Fmelanoma%5Fsingle%5Fcells%5FTPM%5FGEO%2Etxt%2Egz",
            "TPM file",
        ),
        (
            meta_path,
            _GEO_BASE + "GSE120575%5Fpatient%5FID%5Fsingle%5Fcells%2Etxt%2Egz",
            "metadata file",
        ),
    ]
    missing = [(p, url, label) for p, url, label in _sade_feldman_files if not p.exists()]
    if missing:
        if not allow_download:
            names = ", ".join(str(p) for p, _, _ in missing)
            raise FileNotFoundError(
                f"Missing file(s): {names}. Download from GEO: "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575"
            )
        raw_dir_resolved.mkdir(parents=True, exist_ok=True)
        for dest, url, label in missing:
            _download_file(url, dest, label)
    logger.info("Processing raw data (this may take a minute)...")
    with gzip.open(tpm_path, "rt") as f:
        header1 = f.readline().strip().split("\t")
        header2 = f.readline().strip().split("\t")
        if len(header1) != len(header2):
            raise ValueError("TPM file header rows have inconsistent lengths.")
        sample_ids = header1
        time_labels = header2
        data = f.read()

    df = pd.read_csv(StringIO(data), sep="\t", header=None)
    if df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]

    genes = df.iloc[:, 0].astype(str).values
    mat = df.iloc[:, 1:]
    if mat.shape[1] != len(sample_ids):
        raise ValueError(f"TPM matrix columns ({mat.shape[1]}) != sample IDs ({len(sample_ids)}).")
    mat.columns = sample_ids

    meta = pd.read_csv(meta_path, sep="\t", skiprows=19, encoding="latin1")
    meta = meta.rename(
        columns={
            "title": "sample_id",
            "characteristics: patinet ID (Pre=baseline; Post= on treatment)": "patient_raw",
            "characteristics: response": "response",
        }
    )
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta = meta.dropna(subset=["sample_id"]).copy()

    meta = meta[meta["sample_id"].str.match(r"^[A-Z]\d+_P\d+_M\d+")].copy()
    meta = meta[meta["response"].isin(["Responder", "Non-responder"])].copy()

    meta["visit"] = meta["patient_raw"].astype(str).str.split("_").str[0]
    meta["participant_id"] = meta["patient_raw"].astype(str).str.extract(r"(P\d+)")[0]

    time_map = dict(zip(sample_ids, time_labels))
    meta["time_label"] = meta["sample_id"].map(time_map)

    meta = meta.set_index("sample_id")
    meta = meta.loc[[s for s in sample_ids if s in meta.index]].copy()

    adata = ad.AnnData(X=mat.T.loc[meta.index].values.astype(np.float32))
    adata.obs = meta.copy()
    adata.var_names = genes

    if max_cells_per_participant_visit is not None:
        rng = np.random.default_rng(seed)
        keep_indices: list = []
        for (pid, visit), group in adata.obs.groupby(["participant_id", "visit"], observed=True):
            n_cells = len(group)
            if n_cells > max_cells_per_participant_visit:
                keep = rng.choice(group.index, size=max_cells_per_participant_visit, replace=False)
            else:
                keep = group.index.values
            keep_indices.extend(keep)
        adata = adata[keep_indices].copy()
        logger.info(
            f"Stratified sampling: {adata.n_obs:,} cells (max {max_cells_per_participant_visit} per participant-visit)"
        )
    else:
        logger.info(f"Using full dataset: {adata.n_obs:,} cells (no subsampling)")

    adata.layers["tpm"] = adata.X.copy()
    adata.layers["log1p_tpm"] = adata.X.copy() if _looks_log1p(adata.X) else np.log1p(adata.X)
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)

    # ── PCA → neighbors → UMAP → Leiden ────────────────────────────────
    # Compute BEFORE annotation so that cell-type labels are assigned to
    # the SAME Leiden clusters that the UMAP is built from.
    logger.info("Computing PCA / neighbors / UMAP / Leiden...")
    adata_work = adata.copy()
    adata_work.X = adata_work.layers["log1p_tpm"]
    sc.pp.highly_variable_genes(adata_work, n_top_genes=3000, flavor="seurat")
    adata_hvg = adata_work[:, adata_work.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    del adata_work, adata_hvg

    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=1.0)

    # ── Cell type annotation ────────────────────────────────────────────
    # Uses the Leiden clusters computed above (same embedding as UMAP).
    logger.info("Annotating cell types from marker genes...")
    adata.obs["cell_type"] = _annotate_immune_celltypes(adata)

    adata.uns["processing_params"] = processing_params
    adata.uns["data_source"] = "GSE120575"
    adata.uns["paper"] = "Sade-Feldman et al., Cell 2018"

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_path)
    logger.info(f"Saved processed file: {processed_path}")
    logger.info(f"Loaded Sade-Feldman dataset: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata


def load_stephenson_data(
    data_dir: str | None = None,
    processed_name: str = "stephenson_covid19_v3.h5ad",
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
    *,
    data_path: str | None = None,
) -> ad.AnnData:
    """Load and preprocess Stephenson COVID-19 dataset (E-MTAB-10026).

    Parameters
    ----------
    data_dir : str
        Directory containing (or to store) the raw data files.
    processed_name : str
        Filename for the cached processed h5ad file.
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download the data file automatically when missing.
    force_reprocess : bool
        If True, reprocess even when a cached file exists.
    data_path : str or None
        .. deprecated:: 0.2.2
            Use *data_dir* instead.  When supplied, *data_dir* is ignored
            and the parent directory of *data_path* is used.

    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    data_dir = data_dir or _default_data_dir("stephenson")

    # Backward compat: if someone passes an .h5ad file path positionally
    # as data_dir (old API had data_path as first param), treat it as data_path.
    if data_path is None and str(data_dir).endswith(".h5ad"):
        data_path = data_dir
        data_dir = _default_data_dir("stephenson")  # reset to default

    if data_path is not None:
        warnings.warn(
            "load_stephenson_data(data_path=...) is deprecated. Use data_dir=... instead.",
            FutureWarning,
            stacklevel=2,
        )
        raw_file = _resolve_file(data_path)
        data_dir_path = (
            raw_file.parent.parent if raw_file.exists() else Path(data_path).parent.parent
        )
    else:
        data_dir_path = Path(data_dir)
        raw_file = data_dir_path / "raw" / "covid_portal_210320_with_raw.h5ad"

    processed_path = data_dir_path / "processed" / processed_name

    if not force_reprocess and processed_path.exists():
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if not prev:
            warnings.warn(
                "Cached file lacks processing_params metadata; cannot verify it matches "
                "current settings. Consider reprocessing with force_reprocess=True.",
                UserWarning,
                stacklevel=2,
            )
        logger.info(f"Loaded cached file: {processed_path}")
        logger.info(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        return adata

    if not raw_file.exists():
        # Also check old location (data_dir directly, not raw subdir)
        legacy_raw = data_dir_path / "covid_portal_210320_with_raw.h5ad"
        if legacy_raw.exists():
            raw_file = legacy_raw
        elif not allow_download:
            raise FileNotFoundError(
                f"Data not found at {raw_file}. Download from: "
                "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/"
            )
        else:
            raw_file.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/"
                "covid_portal_210320_with_raw.h5ad"
            )
            _download_file(url, raw_file, "Stephenson COVID-19 data")
    logger.info("Processing raw data...")
    adata = ad.read_h5ad(raw_file)

    X_counts, source = _get_counts_matrix(adata)
    if X_counts is None:
        raise ValueError("No raw counts found in dataset.")
    adata.layers["counts"] = X_counts
    logger.info(f"  Counts source: {source}")

    obs = adata.obs.copy()
    obs["severity"] = obs["Status_on_day_collection_summary"].astype(str)
    obs = obs[obs["severity"].isin(["Mild", "Severe"])].copy()
    logger.info(f"  After severity filter: {len(obs):,} cells")

    obs["dfo"] = pd.to_numeric(obs["Days_from_onset"], errors="coerce")
    obs["dfo_bin"] = pd.cut(
        obs["dfo"],
        bins=[-np.inf, 7, 14, np.inf],
        labels=["DFO_0-7", "DFO_8-14", "DFO_15+"],
    ).astype(str)

    valid_dfo = obs["dfo_bin"].isin(["DFO_0-7", "DFO_8-14", "DFO_15+"])
    obs = obs[valid_dfo].copy()
    logger.info(f"  After DFO filter: {len(obs):,} cells")

    if "Collection_Day" in obs.columns:
        obs["collection_day"] = obs["Collection_Day"].astype(str)

    obs["participant_id"] = obs["patient_id"].astype(str)
    obs["celltype"] = obs["full_clustering"].astype(str)

    adata = adata[obs.index].copy()
    adata.obs = obs
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)
    adata.uns["processing_params"] = {"version": "v4", "qc_artifact_flag": True}

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_path)
    logger.info(f"  Saved: {processed_path}")
    logger.info(f"  Final: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata


def load_vaccine_gse171964(
    data_dir: str | None = None,
    processed_name: str = "vaccine_gse171964.h5ad",
    max_participants: int | None = None,
    max_cells_per_group: int | None = None,
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load and preprocess GSE171964 PBMC vaccine time course data (Day 0 vs Day 7).

    Parameters
    ----------
    data_dir : str
        Directory containing (or to store) the raw data files.
    processed_name : str
        Filename for the cached processed h5ad file.
    max_participants : int or None
        Maximum number of participants to retain.
    max_cells_per_group : int or None
        Maximum number of cells per participant-day-celltype group.
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download missing files from GEO automatically.
    force_reprocess : bool
        If True, reprocess even when a cached file exists.

    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    processing_params = {
        "version": "v3",
        "max_participants": max_participants,
        "max_cells_per_group": max_cells_per_group,
        "seed": seed,
        "days": [0, 7],
    }

    data_dir = data_dir or _default_data_dir("vaccine_gse171964")
    data_dir_path = Path(data_dir)
    processed_path = data_dir_path / "processed" / processed_name

    if not force_reprocess and processed_path.exists():
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if prev:
            if _params_match(prev, processing_params):
                logger.info(
                    f"Loaded processed vaccine dataset (GSE171964): {adata.n_obs} cells, {adata.n_vars} genes"
                )
                return adata
            logger.info("Processed file parameters differ; reprocessing.")
            logger.debug(f"  Stored: {prev}")
            logger.debug(f"  Current: {processing_params}")
        else:
            warnings.warn(
                "Cached file lacks processing_params metadata; cannot verify it matches "
                "current settings. Consider reprocessing with force_reprocess=True.",
                UserWarning,
                stacklevel=2,
            )
            logger.info(
                f"Loaded processed vaccine dataset (GSE171964): {adata.n_obs} cells, {adata.n_vars} genes"
            )
            return adata

    raw_dir = data_dir_path / "raw"
    raw_dir_resolved = _resolve_dir_with_files(
        str(raw_dir),
        [
            "GSE171964_barcodes_v2.tsv.gz",
            "GSE171964_feats_v2.tsv.gz",
            "GSE171964_geo_pheno_v2.csv.gz",
            "GSE171964_countsmatrix_v2.mtx.gz",
        ],
    )

    barcodes_path = raw_dir_resolved / "GSE171964_barcodes_v2.tsv.gz"
    feats_path = raw_dir_resolved / "GSE171964_feats_v2.tsv.gz"
    pheno_path = raw_dir_resolved / "GSE171964_geo_pheno_v2.csv.gz"
    mtx_path = raw_dir_resolved / "GSE171964_countsmatrix_v2.mtx.gz"

    _GEO_BASE_V = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE171964&format=file&file="
    _vaccine_files = [
        (barcodes_path, _GEO_BASE_V + "GSE171964%5Fbarcodes%5Fv2%2Etsv%2Egz", "barcodes file"),
        (feats_path, _GEO_BASE_V + "GSE171964%5Ffeats%5Fv2%2Etsv%2Egz", "features file"),
        (pheno_path, _GEO_BASE_V + "GSE171964%5Fgeo%5Fpheno%5Fv2%2Ecsv%2Egz", "pheno file"),
        (mtx_path, _GEO_BASE_V + "GSE171964%5Fcountsmatrix%5Fv2%2Emtx%2Egz", "counts matrix"),
    ]
    missing = [(p, url, label) for p, url, label in _vaccine_files if not p.exists()]
    if missing:
        if not allow_download:
            names = ", ".join(str(p) for p, _, _ in missing)
            raise FileNotFoundError(
                f"Missing file(s): {names}. Download from GEO: "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171964 "
                "or set allow_download=True to fetch automatically."
            )
        raw_dir_resolved.mkdir(parents=True, exist_ok=True)
        for dest, url, label in missing:
            _download_file(url, dest, label)
    barcodes = (
        pd.read_csv(barcodes_path, sep="\\s+", header=None, engine="python", skiprows=1)[1]
        .astype(str)
        .str.strip('"')
        .tolist()
    )
    features = (
        pd.read_csv(feats_path, sep="\\s+", header=None, engine="python", skiprows=1)[1]
        .astype(str)
        .str.strip('"')
        .tolist()
    )

    with gzip.open(mtx_path, "rb") as f:
        X = mmread(f).tocsr()

    if X.shape[0] == len(features) and X.shape[1] == len(barcodes):
        X = X.T
    elif X.shape[0] == len(barcodes) and X.shape[1] == len(features):
        pass
    else:
        raise ValueError("Matrix dimensions do not match barcodes/features.")

    adata = ad.AnnData(X=X)
    adata.obs_names = barcodes
    adata.var_names = features

    pheno = pd.read_csv(pheno_path)
    pheno["barcode"] = pheno["barcode"].astype(str)
    pheno = pheno.set_index("barcode")
    pheno = pheno.loc[adata.obs_names]
    adata.obs = pheno

    adata = adata[adata.obs["day"].isin([0, 7])].copy()

    import scanpy as sc  # local import (optional dependency), like the other loaders

    # ── QC (10x UMI): mito %, gene/cell filters -- matches the other 10x loaders ──
    # Run BEFORE the pairing filter so pairing is computed on QC-passing cells.
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info(f"QC: {n_before:,} → {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    paired = adata.obs.groupby("pt_id")["day"].nunique()
    keep_ids = paired[paired >= 2].index
    adata = adata[adata.obs["pt_id"].isin(keep_ids)].copy()

    rng = np.random.default_rng(seed)
    uniq_ids = adata.obs["pt_id"].unique()
    # If max_participants is None, use all participants (no subsampling)
    if max_participants is not None:
        n = min(len(uniq_ids), max_participants)
        sel = rng.choice(uniq_ids, size=n, replace=False)
        adata = adata[adata.obs["pt_id"].isin(sel)].copy()
    # else: use all participants

    if max_cells_per_group is not None:
        grp = ["pt_id", "day", "clustnm"]
        sampled = adata.obs.groupby(grp, observed=True, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_cells_per_group), random_state=seed)
        )
        adata = adata[sampled.index].copy()

    adata.layers["counts"] = adata.X.copy()
    # Normalize (GSE171964 matrix is raw integer UMI counts): CP10K + log1p, so
    # .X is log-normalized like the other 10x datasets (counts kept in the layer).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)
    adata.uns["processing_params"] = processing_params

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_path)
    logger.info(f"Saved processed file: {processed_path}")
    logger.info(f"Loaded vaccine dataset (GSE171964): {adata.n_obs} cells, {adata.n_vars} genes")
    logger.info(f"Days: {adata.obs['day'].unique()}")
    logger.info(f"Participants: {adata.obs['pt_id'].nunique()}")
    logger.info(f"Cell types: {adata.obs['clustnm'].nunique()}")
    return adata


def count_paired(
    obs: pd.DataFrame,
    visit_col: str,
    visits: Sequence[str],
    participant_col: str = "participant_id",
) -> int:
    """Count participants with data at both visits.

    Parameters
    ----------
    obs
        DataFrame containing the participant-visit data.
    visit_col
        Column name in `obs` to use for visit labels.
    visits
        Sequence of visit labels to check (e.g. ["baseline", "followup"]).
    participant_col
        Column name in `obs` to use for participant IDs.

    Returns
    -------
    int
        Number of participants with data at both visits.

    Raises
    ------
    ValueError
        If visits does not contain at least 2 labels (baseline and followup).
    """
    if len(visits) < 2:
        raise ValueError(
            f"visits must contain at least 2 labels, got {len(visits)}: {list(visits)}"
        )
    wide = obs.groupby([participant_col, visit_col], observed=True).size().unstack(fill_value=0)
    if visits[0] not in wide.columns or visits[1] not in wide.columns:
        return 0
    has_both = (wide[visits[0]] > 0) & (wide[visits[1]] > 0)
    return int(has_both.sum())


def verify_paired_participants(
    obs: pd.DataFrame,
    visit_col: str,
    visits: Sequence[str],
    features: Sequence[str] | None = None,
    participant_col: str = "participant_id",
) -> dict:
    """Validate paired participants by visit presence and optional feature completeness.

    Parameters
    ----------
    obs
        DataFrame containing the participant-visit data.
    visit_col
        Column name in `obs` to use for visit labels.
    visits
        Sequence of visit labels to check (e.g. ["baseline", "followup"]).
    features
        Sequence of feature names to check.
    participant_col
        Column name in `obs` to use for participant IDs.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - paired_ids: set of participant IDs with both visits (and non-NaN features if provided)
        - dropped_ids: list of participant IDs dropped by validation
        - n_paired: count of paired_ids
        - n_total: total unique participants
    """
    if len(visits) < 2:
        raise ValueError(
            f"visits must contain at least 2 labels, got {len(visits)}: {list(visits)}"
        )
    wide = obs.groupby([participant_col, visit_col], observed=True).size().unstack(fill_value=0)
    if visits[0] not in wide.columns or visits[1] not in wide.columns:
        paired_ids = set()
    else:
        paired_ids = set(wide[(wide[visits[0]] > 0) & (wide[visits[1]] > 0)].index)

    if features:
        grouped = obs.groupby([participant_col, visit_col], observed=True)[list(features)]
        # Use .first() instead of .mean() so categorical/string features
        # don't raise TypeError.  For numeric columns the NaN-presence check
        # below is still correct because .first() returns NaN for empty groups.
        df_pv = grouped.first().reset_index()
        valid_ids: set | None = None
        for feat in features:
            wide_feat = df_pv.pivot(index=participant_col, columns=visit_col, values=feat)
            if visits[0] not in wide_feat.columns or visits[1] not in wide_feat.columns:
                feat_valid = set()
            else:
                mask = wide_feat[visits[0]].notna() & wide_feat[visits[1]].notna()
                feat_valid = set(wide_feat[mask].index)
            valid_ids = feat_valid if valid_ids is None else (valid_ids & feat_valid)
        if valid_ids is not None:
            paired_ids = paired_ids & valid_ids

    all_ids = set(obs[participant_col].unique())
    return {
        "paired_ids": paired_ids,
        "dropped_ids": sorted(all_ids - paired_ids),
        "n_paired": len(paired_ids),
        "n_total": len(all_ids),
    }


def categorize_celltype(ct: str) -> str:
    """Map fine-grained cell types to coarse lineages (COVID-19 example).

    Parameters
    ----------
    ct
        Cell type string.

    Returns
    -------
    str
        Coarse lineage string.
    """
    ct_lower = str(ct).lower()
    if "cd4" in ct_lower or "th1" in ct_lower or "th2" in ct_lower or "treg" in ct_lower:
        return "CD4_T"
    if "cd8" in ct_lower or "cytotoxic" in ct_lower:
        return "CD8_T"
    if "nk" in ct_lower or "natural killer" in ct_lower:
        return "NK"
    # DC check must precede B cell check so "plasmacytoid dendritic cell"
    # is not captured by the "plasma" substring in the B cell rule.
    if "dc" in ct_lower or "dendritic" in ct_lower:
        return "DCs"
    if "b cell" in ct_lower or "plasma" in ct_lower or "b_cell" in ct_lower:
        return "B_cells"
    if "mono" in ct_lower or "cd14" in ct_lower or "cd16" in ct_lower:
        return "Monocytes"
    return "Other"


def _extract_aml_sample_name(filename: str) -> str | None:
    """Extract sample name from AML filename (ignoring GSM number)."""
    m = re.search(r"GSM\d+_(.+)\.(dem|anno)\.txt\.gz", filename)
    return m.group(1) if m else None


def _parse_aml_sample_info(sample_name: str) -> tuple[str, str, int]:
    """Parse patient ID and day from AML sample name."""
    if "-D" in sample_name:
        parts = sample_name.rsplit("-D", 1)
        patient = parts[0]
        try:
            day = int(parts[1])
        except ValueError:
            day = 0
    else:
        patient = sample_name
        day = 0
    return sample_name, patient, day


def _process_aml_raw(
    raw_dir: Path,
    max_cells_per_sample: int | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Process raw GSE116256 AML files into an AnnData object.

    Reads per-sample expression (.dem.txt.gz) and annotation (.anno.txt.gz)
    files, combines them, applies QC, normalisation, and computes embeddings.
    Cell-type labels come from the original van Galen et al. annotations.
    """
    import scanpy as sc

    # Build file mapping: match dem ↔ anno by sample name
    dem_files: dict[str, Path] = {}
    anno_files: dict[str, Path] = {}
    for fp in raw_dir.glob("GSM*_*.dem.txt.gz"):
        name = _extract_aml_sample_name(fp.name)
        if name:
            dem_files[name] = fp
    for fp in raw_dir.glob("GSM*_*.anno.txt.gz"):
        name = _extract_aml_sample_name(fp.name)
        if name:
            anno_files[name] = fp

    matched = sorted(set(dem_files) & set(anno_files))
    if not matched:
        raise ValueError(
            f"No matched sample pairs found in {raw_dir}. "
            f"Found {len(dem_files)} expression and {len(anno_files)} annotation files."
        )
    logger.info(f"Found {len(matched)} matched AML sample pairs")

    # Pass 1: collect all gene names
    all_genes: set[str] = set()
    for sname in matched:
        with gzip.open(dem_files[sname], "rt") as f:
            df = pd.read_csv(f, sep="\t", usecols=[0])
            all_genes.update(df.iloc[:, 0].tolist())
    all_genes_sorted = sorted(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes_sorted)}
    logger.info(f"Total unique genes across samples: {len(all_genes_sorted)}")

    # Pass 2: load expression + annotations
    rng = np.random.default_rng(seed)
    all_X: list[sp.csr_matrix] = []
    all_obs: list[pd.DataFrame] = []
    n_genes_total = len(all_genes_sorted)

    for sname in matched:
        with gzip.open(dem_files[sname], "rt") as f:
            expr_df = pd.read_csv(f, sep="\t", index_col=0)
        with gzip.open(anno_files[sname], "rt") as f:
            anno_df = pd.read_csv(f, sep="\t", index_col=0)

        common_cells = sorted(set(expr_df.columns) & set(anno_df.index))
        if not common_cells:
            logger.warning(f"No common cells for {sname}, skipping")
            continue

        if max_cells_per_sample and len(common_cells) > max_cells_per_sample:
            common_cells = list(rng.choice(common_cells, max_cells_per_sample, replace=False))

        expr_df = expr_df[common_cells]
        anno_df = anno_df.loc[common_cells]

        X_raw = sp.csr_matrix(expr_df.T.values.astype(np.float32))
        genes = list(expr_df.index)

        # Re-index genes to common set
        X_re = sp.lil_matrix((X_raw.shape[0], n_genes_total), dtype=np.float32)
        for j, gene in enumerate(genes):
            if gene in gene_to_idx:
                X_re[:, gene_to_idx[gene]] = X_raw[:, j].toarray()

        _, patient, day = _parse_aml_sample_info(sname)
        unique_cells = [f"{sname}_{c}" for c in common_cells]
        obs_df = anno_df.copy()
        obs_df.index = unique_cells
        obs_df["sample_id"] = sname
        obs_df["patient_id"] = patient
        obs_df["day"] = day

        all_X.append(sp.csr_matrix(X_re))
        all_obs.append(obs_df)
        del X_raw, X_re
        gc.collect()

    if not all_X:
        raise ValueError("No samples could be loaded from raw AML data.")

    X_combined = sp.vstack(all_X)
    obs_combined = pd.concat(all_obs, axis=0)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, var=pd.DataFrame(index=all_genes_sorted))
    del all_X
    gc.collect()

    # ── QC and normalisation ──────────────────────────────────────────
    adata.layers["counts"] = adata.X.copy()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info(f"QC: {n_before:,} → {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)

    # ── Standardised sctrial obs columns ──────────────────────────────
    obs = adata.obs
    obs["participant_id"] = obs["patient_id"].astype(str)
    obs["visit"] = obs["day"].apply(lambda d: "Pre" if d == 0 else "Post")
    obs["sample_type"] = obs["patient_id"].apply(
        lambda pid: "AML" if str(pid).startswith("AML") else "Healthy"
    )
    if "CellType" in obs.columns:
        obs["cell_type"] = obs["CellType"]
    elif "PredictionRefined" in obs.columns:
        obs["cell_type"] = obs["PredictionRefined"]
    else:
        obs["cell_type"] = "Unknown"
    if "PredictionRefined" in obs.columns:
        obs["is_malignant"] = obs["PredictionRefined"].apply(
            lambda x: "malignant" in str(x).lower() if pd.notna(x) else False
        )
    else:
        obs["is_malignant"] = False
    obs["response"] = (
        obs["sample_type"].map({"AML": "Treatment", "Healthy": "Control"}).fillna("Unknown")
    )
    adata.obs = obs

    # ── Embeddings (HVG → PCA → neighbours → UMAP) ───────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=False)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    # Store paired-patient list
    aml_obs = adata.obs[adata.obs["sample_type"] == "AML"]
    paired = []
    for pid in aml_obs["participant_id"].unique():
        days = set(aml_obs.loc[aml_obs["participant_id"] == pid, "day"].unique())
        if 0 in days and any(d > 0 for d in days):
            paired.append(pid)
    adata.uns["paired_aml_patients"] = paired

    adata.uns["dataset"] = "GSE116256"
    adata.uns["paper"] = "van Galen et al., Cell 2019"
    adata.uns["description"] = "AML chemotherapy longitudinal scRNA-seq"
    return adata


def load_aml(
    data_dir: str | None = None,
    processed_name: str = "gse116256_aml_processed.h5ad",
    max_cells_per_sample: int | None = None,
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load the van Galen AML chemotherapy dataset (GSE116256).

    This dataset contains pre/post-chemotherapy bone marrow samples from AML
    patients with cell-type annotations and treatment-response metadata.

    Parameters
    ----------
    data_dir : str
        Directory containing (or to store) the data files.
        Raw files go in ``<data_dir>/raw/`` and the processed cache in
        ``<data_dir>/processed/``.
    processed_name : str
        Filename for the cached processed h5ad file.
    max_cells_per_sample : int
        Maximum cells to keep per sample after subsampling.
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download raw data from GEO when not found locally.
    force_reprocess : bool
        If True, reprocess even when a cached file exists.

    Returns
    -------
    AnnData
        The processed AnnData object.

    Notes
    -----
    The raw data is automatically downloaded from GEO when
    ``allow_download=True``:
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256

    Reference: van Galen et al., Cell 2019.

    Examples
    --------
    >>> adata = sctrial.load_aml(allow_download=True)
    """
    data_dir = data_dir or _default_data_dir("aml")
    data_dir_path = Path(data_dir)

    processing_params = {
        "version": "v2",
        "max_cells_per_sample": max_cells_per_sample,
        "seed": seed,
    }

    # ── Try to load cached processed file ─────────────────────────────
    processed_path = data_dir_path / "processed" / processed_name

    if not force_reprocess and processed_path.exists():
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if prev:
            if _params_match(prev, processing_params):
                logger.info(
                    f"Loaded AML dataset (GSE116256): {adata.n_obs:,} cells, {adata.n_vars:,} genes"
                )
                return adata
            logger.info("Processed file parameters differ; reprocessing.")
        else:
            warnings.warn(
                "Cached file lacks processing_params metadata; cannot verify it matches "
                "current settings. Consider reprocessing with force_reprocess=True.",
                UserWarning,
                stacklevel=2,
            )
            logger.info(
                f"Loaded AML dataset (GSE116256): {adata.n_obs:,} cells, {adata.n_vars:,} genes"
            )
            return adata

    # ── Locate or download raw files ──────────────────────────────────
    try:
        import scanpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "scanpy is required for AML dataset processing. "
            "Install with: pip install sctrial[plots]  or  pip install scanpy"
        ) from None

    raw_dir = data_dir_path / "raw"
    found_raw = raw_dir if raw_dir.is_dir() and list(raw_dir.glob("GSM*_*.dem.txt.gz")) else None

    if found_raw is None:
        if not allow_download:
            raise FileNotFoundError(
                f"AML dataset not found. Searched for raw files and "
                f"'{processed_name}' in several locations including {data_dir_path}. "
                "Download from GEO: "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116256 "
                "or set allow_download=True to fetch automatically."
            )
        # Download tar from GEO and extract
        raw_dir.mkdir(parents=True, exist_ok=True)
        tar_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE116256&format=file"
        tar_dest = raw_dir / "GSE116256_RAW.tar"
        _download_file(tar_url, tar_dest, "GSE116256 supplementary tar")
        logger.info("Extracting raw files...")
        with tarfile.open(tar_dest, "r") as tf:
            tf.extractall(path=raw_dir)
        # Clean up tar to save disk space
        tar_dest.unlink(missing_ok=True)
        found_raw = raw_dir

    # ── Process raw data ──────────────────────────────────────────────
    logger.info("Processing raw AML data (this may take several minutes)...")
    adata = _process_aml_raw(found_raw, max_cells_per_sample=max_cells_per_sample, seed=seed)
    adata.uns["processing_params"] = processing_params

    # Save cache
    processed_dir = data_dir_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / processed_name
    adata.write_h5ad(out_path)
    logger.info(f"Saved processed AML file: {out_path}")
    logger.info(f"Loaded AML dataset (GSE116256): {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata


def _parse_cart_sample_info(filename: str) -> tuple[str | None, str | None, int]:
    """Parse patient and timepoint from CAR-T filename."""
    m = re.search(r"GSM\d+_(P\d+)_(.+)_rna\.csv\.gz", filename)
    if not m:
        return None, None, -1
    patient = m.group(1)
    tp_raw = m.group(2)
    if "Leukapheresis" in tp_raw:
        return patient, "Leukapheresis", 0
    if "4wk" in tp_raw:
        return patient, "4wk_post", 28
    if "6mo" in tp_raw:
        return patient, "6mo_post", 180
    if "12mo" in tp_raw:
        return patient, "12mo_post", 365
    return patient, tp_raw, -1


def _process_cart_raw(
    raw_dir: Path,
    max_cells_per_sample: int | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Process raw GSE290722 CAR-T files into an AnnData object.

    Reads per-sample expression CSV files, combines them, applies QC,
    normalisation, computes embeddings, and annotates cell types via
    Leiden clustering + Wilcoxon marker scoring.
    """
    import scanpy as sc

    rna_files = sorted(raw_dir.glob("GSM*_*_rna.csv.gz"))
    if not rna_files:
        raise ValueError(f"No RNA expression files found in {raw_dir}")

    samples: list[dict] = []
    for f in rna_files:
        patient, timepoint, days = _parse_cart_sample_info(f.name)
        if patient is None:
            continue
        samples.append(
            {
                "file": f,
                "patient": patient,
                "timepoint": timepoint,
                "days": days,
                "sample_id": f"{patient}_{timepoint}",
            }
        )

    if not samples:
        raise ValueError("No valid CAR-T samples found.")

    logger.info(f"Found {len(samples)} CAR-T RNA expression files")

    # Load first sample to get gene names
    first_df = pd.read_csv(samples[0]["file"])
    gene_col = first_df.columns[-1]
    gene_names = first_df[gene_col].tolist()
    logger.info(f"Detected {len(gene_names)} genes")

    # Load all samples
    rng = np.random.default_rng(seed)
    all_X: list[np.ndarray] = []
    all_obs: list[pd.DataFrame] = []

    for sample in samples:
        df = pd.read_csv(sample["file"])
        gc_col = df.columns[-1]
        genes = df[gc_col].tolist()
        expr_df = df.drop(columns=[gc_col])
        cells = list(expr_df.columns)
        X = expr_df.values.T.astype(np.float32)  # cells × genes

        # Re-order genes if needed
        if genes != gene_names:
            if set(genes) == set(gene_names):
                g2i = {g: i for i, g in enumerate(genes)}
                new_order = [g2i[g] for g in gene_names]
                X = X[:, new_order]
            else:
                logger.warning(f"Gene mismatch for {sample['sample_id']}, skipping")
                continue

        # Subsample
        if max_cells_per_sample and X.shape[0] > max_cells_per_sample:
            idx = rng.choice(X.shape[0], max_cells_per_sample, replace=False)
            X = X[idx]
            cells = [cells[i] for i in idx]

        obs_df = pd.DataFrame(
            {
                "cell_id": cells,
                "sample_id": sample["sample_id"],
                "patient_id": sample["patient"],
                "timepoint": sample["timepoint"],
                "days_post_treatment": sample["days"],
            },
            index=[f"{sample['sample_id']}_{c}" for c in cells],
        )

        all_X.append(X)
        all_obs.append(obs_df)
        del df, expr_df
        gc.collect()

    if not all_X:
        raise ValueError("No CAR-T samples could be loaded.")

    X_combined = sp.csr_matrix(np.vstack(all_X))
    obs_combined = pd.concat(all_obs, axis=0)
    adata = ad.AnnData(X=X_combined, obs=obs_combined, var=pd.DataFrame(index=gene_names))
    del all_X
    gc.collect()

    # ── QC and normalisation ──────────────────────────────────────────
    adata.layers["counts"] = adata.X.copy()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    if "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info(f"QC: {n_before:,} → {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)

    # ── Standardised sctrial obs columns ──────────────────────────────
    obs = adata.obs
    obs["participant_id"] = obs["patient_id"].astype(str)
    obs["visit"] = obs["timepoint"].apply(lambda t: "Pre" if t == "Leukapheresis" else "Post")
    obs["is_paired"] = False  # filled below
    obs["response"] = "CAR-T"
    adata.obs = obs

    # ── Embeddings + clustering ───────────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", subset=False)
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    # Clustering — try Leiden, fallback to KMeans
    try:
        sc.tl.leiden(adata, resolution=0.8)
    except Exception:
        from sklearn.cluster import MiniBatchKMeans

        X_pca = adata.obsm["X_pca"][:, :20]
        kmeans = MiniBatchKMeans(n_clusters=15, random_state=seed, batch_size=1000)
        clusters = kmeans.fit_predict(X_pca)
        adata.obs["leiden"] = pd.Categorical([str(c) for c in clusters])

    # ── Cell type annotation (marker scoring) ─────────────────────────
    logger.info("Annotating CAR-T cell types...")
    adata.obs["cell_type"] = _annotate_immune_celltypes(adata)

    # Mark paired patients
    patients = adata.obs["participant_id"].unique()
    paired_patients = []
    for p in patients:
        tps = set(adata.obs.loc[adata.obs["participant_id"] == p, "timepoint"].unique())
        if "Leukapheresis" in tps and len(tps) > 1:
            paired_patients.append(p)
    adata.obs["is_paired"] = adata.obs["participant_id"].isin(paired_patients)

    adata.uns["paired_patients"] = paired_patients
    adata.uns["dataset"] = "GSE290722"
    adata.uns["trial"] = "ZUMA-1"
    adata.uns["description"] = "CAR-T therapy longitudinal scRNA-seq"
    return adata


def load_cart(
    data_dir: str | None = None,
    processed_name: str = "gse290722_cart_processed.h5ad",
    max_cells_per_sample: int | None = None,
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load the CAR-T cell therapy dataset (GSE290722).

    This dataset contains pre/post-CAR-T infusion samples with cell-type
    annotations and treatment-response metadata from the ZUMA-1 trial.

    Parameters
    ----------
    data_dir : str
        Directory containing (or to store) the data files.
        Raw files go in ``<data_dir>/raw/`` and the processed cache in
        ``<data_dir>/processed/``.
    processed_name : str
        Filename for the cached processed h5ad file.
    max_cells_per_sample : int
        Maximum cells to keep per sample after subsampling.
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download raw data from GEO when not found locally.
    force_reprocess : bool
        If True, reprocess even when a cached file exists.

    Returns
    -------
    AnnData
        The processed AnnData object.

    Notes
    -----
    The raw data is automatically downloaded from GEO when
    ``allow_download=True``:
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290722

    Reference: GSE290722 CAR-T therapy dataset (ZUMA-1 trial).

    Examples
    --------
    >>> adata = sctrial.load_cart(allow_download=True)
    """
    data_dir = data_dir or _default_data_dir("cart")
    data_dir_path = Path(data_dir)

    processing_params = {
        "version": "v2",
        "max_cells_per_sample": max_cells_per_sample,
        "seed": seed,
    }

    # ── Try to load cached processed file ─────────────────────────────
    processed_path = data_dir_path / "processed" / processed_name

    if not force_reprocess and processed_path.exists():
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if prev:
            if _params_match(prev, processing_params):
                logger.info(
                    f"Loaded CAR-T dataset (GSE290722): {adata.n_obs:,} cells, {adata.n_vars:,} genes"
                )
                return adata
            logger.info("Processed file parameters differ; reprocessing.")
        else:
            warnings.warn(
                "Cached file lacks processing_params metadata; cannot verify it matches "
                "current settings. Consider reprocessing with force_reprocess=True.",
                UserWarning,
                stacklevel=2,
            )
            logger.info(
                f"Loaded CAR-T dataset (GSE290722): {adata.n_obs:,} cells, {adata.n_vars:,} genes"
            )
            return adata

    # ── Locate or download raw files ──────────────────────────────────
    try:
        import scanpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "scanpy is required for CAR-T dataset processing. "
            "Install with: pip install sctrial[plots]  or  pip install scanpy"
        ) from None

    raw_dir = data_dir_path / "raw"
    found_raw = raw_dir if raw_dir.is_dir() and list(raw_dir.glob("GSM*_*_rna.csv.gz")) else None

    if found_raw is None:
        if not allow_download:
            raise FileNotFoundError(
                f"CAR-T dataset not found. Searched for raw files and "
                f"'{processed_name}' in several locations including {data_dir_path}. "
                "Download from GEO: "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290722 "
                "or set allow_download=True to fetch automatically."
            )
        raw_dir.mkdir(parents=True, exist_ok=True)
        tar_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE290722&format=file"
        tar_dest = raw_dir / "GSE290722_RAW.tar"
        _download_file(tar_url, tar_dest, "GSE290722 supplementary tar")
        logger.info("Extracting raw files...")
        with tarfile.open(tar_dest, "r") as tf:
            tf.extractall(path=raw_dir)
        tar_dest.unlink(missing_ok=True)
        found_raw = raw_dir

    # ── Process raw data ──────────────────────────────────────────────
    logger.info("Processing raw CAR-T data (this may take several minutes)...")
    adata = _process_cart_raw(found_raw, max_cells_per_sample=max_cells_per_sample, seed=seed)
    adata.uns["processing_params"] = processing_params

    processed_dir = data_dir_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / processed_name
    adata.write_h5ad(out_path)
    logger.info(f"Saved processed CAR-T file: {out_path}")
    logger.info(f"Loaded CAR-T dataset (GSE290722): {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata

def _process_tnbc_raw(
    raw_dir: Path,
    max_cells_per_participant_visit: int | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Process raw GSE169246 TNBC files into an AnnData object.

    Reads MTX expression matrix, barcodes, features, and GEO metadata CSV,
    applies QC, normalisation, computes embeddings, and annotates cell types.
    Processing is identical to the AML and CAR-T pipelines.
    """
    import scanpy as sc

    logger.info("Loading MTX matrix...")
    mat = mmread(str(raw_dir / "GSE169246_TNBC_RNA.counts.mtx.gz")).T.tocsc()
    logger.info(f"  Raw matrix: {mat.shape[0]} cells x {mat.shape[1]} genes")

    with gzip.open(str(raw_dir / "GSE169246_TNBC_RNA.barcode.tsv.gz"), "rt") as f:
        barcodes = [line.strip() for line in f]
    with gzip.open(str(raw_dir / "GSE169246_TNBC_RNA.feature.tsv.gz"), "rt") as f:
        features_raw = [line.strip().split("\t") for line in f]
    gene_ids   = [feat[0] for feat in features_raw]
    gene_names = [feat[1] if len(feat) > 1 else feat[0] for feat in features_raw]

    adata = ad.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )
    adata.var["gene_ids"] = gene_ids
    adata.var_names_make_unique()

    obs = adata.obs.copy()
    obs["barcode_full"]  = obs.index
    obs["sample_id"]     = obs["barcode_full"].str.split(".").str[1]
    obs["timepoint_raw"] = obs["sample_id"].str.split("_").str[0]
    obs["patient_id"]    = obs["sample_id"].str.extract(r"(P\d+)")
    obs["tissue_type"]   = obs["sample_id"].str.split("_").str[-1]

    # Treatment arm from GEO metadata CSV
    # (generated from GSE169246_family.soft.gz via parse_geo_soft.py)
    geo_meta_path = raw_dir / "geo_metadata.csv"
    if not geo_meta_path.exists():
        raise FileNotFoundError(
            f"geo_metadata.csv not found at {geo_meta_path}. "
            "This file maps patient IDs to treatment arms and must be present "
            "alongside the MTX files. See the TNBC tutorial for instructions "
            "on generating it using parse_geo_soft.py."
        )
    geo_meta = pd.read_csv(geo_meta_path)
    geo_meta["treatment"] = geo_meta["treatment"].replace(
        {"Anti-PD-L1+Chemo": "anti-PDL1+Chemo"}
    )
    geo_meta["patient_id"] = geo_meta["title"].str.extract(r"_?(P\d+)_")
    patient_arm = (
        geo_meta.drop_duplicates("patient_id")
        .set_index("patient_id")["treatment"]
    )
    obs["arm"] = obs["patient_id"].map(patient_arm)

    # Clinical response from mmc3.xlsx (Zhang et al. 2021, Table S2).
    # CR/PR → R (responder), PD/SD → NR (non-responder).
    tnbc_response = _load_tnbc_clinical(raw_dir)
    obs["response"] = obs["patient_id"].map(tnbc_response)

    adata.obs = obs
    logger.info(f"  All cells: {adata.n_obs:,}")

    # ── 5. Filter to tumor biopsies, Pre/Post only ────────────────
    adata = adata[adata.obs["tissue_type"] == "t"].copy()
    logger.info(f"  Tumor biopsies: {adata.n_obs:,}")

    adata = adata[adata.obs["timepoint_raw"].isin(["Pre", "Post"])].copy()
    logger.info(f"  Pre+Post only: {adata.n_obs:,}")

    adata.obs["visit"]          = adata.obs["timepoint_raw"].astype(str)
    adata.obs["participant_id"] = adata.obs["patient_id"].astype(str)

    # Remove participants with <50 cells in any visit (unreliable data)
    remove_pids: set[str] = set()
    for pid in adata.obs["participant_id"].unique():
        for visit in ["Pre", "Post"]:
            n = int(
                ((adata.obs["participant_id"] == pid) &
                 (adata.obs["visit"] == visit)).sum()
            )
            if 0 < n < 50:
                remove_pids.add(pid)
                logger.warning(
                    f"  {pid} {visit}: only {n} cells -- removing participant"
                )
    if remove_pids:
        adata = adata[~adata.obs["participant_id"].isin(remove_pids)].copy()

    # Keep only paired participants (have both Pre and Post)
    paired: set[str] = set()
    for pid in adata.obs["participant_id"].unique():
        visits = set(
            adata.obs.loc[adata.obs["participant_id"] == pid, "visit"].unique()
        )
        if {"Pre", "Post"}.issubset(visits):
            paired.add(pid)
    adata = adata[adata.obs["participant_id"].isin(paired)].copy()
    logger.info(f"  Paired participants: {len(paired)}")
    logger.info(f"  Cells after pairing filter: {adata.n_obs:,}")

    # Verify every paired patient has a clinical response label
    n_missing = int(adata.obs["response"].isna().sum())
    if n_missing > 0:
        missing_pids = sorted(
            adata.obs.loc[adata.obs["response"].isna(), "participant_id"].unique()
        )
        logger.warning(f"  {n_missing} cells have no response label.")
        logger.warning(f"  Missing patient IDs: {missing_pids}")
        logger.warning("  These patients are missing from mmc3.xlsx or have 'Na' efficacy.")

    # Optional: subsample to max_cells_per_participant_visit
    if max_cells_per_participant_visit is not None:
        rng = np.random.default_rng(seed)
        keep: list = []
        for (pid, visit), grp in adata.obs.groupby(
            ["participant_id", "visit"], observed=True
        ):
            if len(grp) > max_cells_per_participant_visit:
                chosen = rng.choice(
                    grp.index,
                    size=max_cells_per_participant_visit,
                    replace=False,
                )
            else:
                chosen = grp.index.values
            keep.extend(chosen)
        adata = adata[keep].copy()
        logger.info(
            f"  After subsampling (max {max_cells_per_participant_visit} "
            f"per participant-visit): {adata.n_obs:,} cells"
        )

    logger.info("QC filtering...")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, inplace=True
    )

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info(
        f"  QC: {n_before:,} -> {adata.n_obs:,} cells, {adata.n_vars:,} genes"
    )

    logger.info("Normalising (normalize_total 1e4 + log1p)...")
    adata.layers["counts"]     = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)

    logger.info("Computing embeddings (HVG -> PCA -> neighbors -> UMAP)...")
    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000, flavor="seurat", subset=False
    )
    sc.tl.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    # Clustering -- try Leiden, fallback to KMeans
    # (flavor="igraph" may be needed on newer scanpy versions;
    #  KMeans fallback ensures the loader works across environments)
    logger.info("Leiden clustering (resolution=0.8)...")
    try:
        sc.tl.leiden(adata, resolution=0.8)
    except Exception:
        try:
            sc.tl.leiden(adata, resolution=0.8, flavor="igraph")
        except Exception:
            from sklearn.cluster import MiniBatchKMeans
            X_pca = adata.obsm["X_pca"][:, :20]
            kmeans = MiniBatchKMeans(n_clusters=15, random_state=seed, batch_size=1000)
            clusters = kmeans.fit_predict(X_pca)
            adata.obs["leiden"] = pd.Categorical([str(c) for c in clusters])

    # Cell type annotation
    logger.info("Annotating cell types (Wilcoxon + weighted scoring)...")
    adata.obs["cell_type"] = _annotate_immune_celltypes(adata)

    # Dataset metadata
    adata.uns["dataset"]     = "GSE169246"
    adata.uns["paper"]       = "Zhang et al., Cancer Cell 2021"
    adata.uns["description"] = (
        "TNBC immunotherapy trial: anti-PDL1+Chemo vs Chemo, "
        "12 paired patients, Pre/Post tumor biopsies"
    )

    # Log final summary
    logger.info(f"  Final: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    logger.info(f"  Participants: {adata.obs['participant_id'].nunique()}")
    logger.info(f"  Arms: {dict(adata.obs['arm'].value_counts())}")
    logger.info(f"  Visits: {dict(adata.obs['visit'].value_counts())}")
    logger.info("  Cell types:")
    for ct, n in adata.obs["cell_type"].value_counts().items():
        logger.info(f"    {ct}: {n:,}")

    gc.collect()
    return adata


def load_tnbc_zhang(
    data_dir: str | None = None,
    processed_name: str = "tnbc_zhang_processed.h5ad",
    max_cells_per_participant_visit: int | None = None,
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load and preprocess Zhang et al. TNBC immunotherapy dataset (GSE169246).

    Randomized trial comparing anti-PDL1+Chemo vs Chemo in triple-negative
    breast cancer. 12 paired patients with Pre- and Post-treatment tumor biopsies.

    Processing pipeline (identical to AML and CAR-T loaders):
    - QC: min_genes=200, max_genes=6000, pct_mt<20, min_cells=10
    - Normalization: normalize_total(1e4) + log1p -> log1p_norm layer
    - Embedding: HVG(2000) -> PCA(50) -> neighbors(15, 30PCs) -> UMAP
    - Clustering: Leiden(0.8)
    - Cell type annotation: Wilcoxon markers + weighted scoring

    Parameters
    ----------
    data_dir : str or None
        Directory containing (or to store) the raw data files.
        Raw files go in ``<data_dir>/raw/`` and the processed cache in
        ``<data_dir>/processed/``. Defaults to ``datasets/tnbc_zhang/``
        relative to the repository root.
    processed_name : str
        Filename for the cached processed h5ad file.
    max_cells_per_participant_visit : int or None
        Maximum number of cells to retain per participant-visit pair.
        If None, all cells are kept (recommended for full analysis).
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download missing raw files from GEO automatically.
        Note: geo_metadata.csv cannot be auto-downloaded and must be
        generated manually. See the TNBC tutorial for instructions.
    force_reprocess : bool
        If True, reprocess from raw files even when a valid cache exists.

    Returns
    -------
    AnnData
        Processed AnnData with the following structure:

        - ``adata.X``: log1p-normalized expression (same as log1p_norm layer)
        - ``adata.layers["counts"]``: raw counts
        - ``adata.layers["log1p_norm"]``: log1p-normalized expression
        - ``adata.obs["participant_id"]``: patient ID (e.g. "P019")
        - ``adata.obs["visit"]``: "Pre" or "Post"
        - ``adata.obs["arm"]``: "anti-PDL1+Chemo" or "Chemo"
        - ``adata.obs["response"]``: "R" or "NR" (CR/PR → R, PD/SD → NR, from mmc3.xlsx Table S2)
        - ``adata.obs["cell_type"]``: annotated immune cell type
        - ``adata.obsm["X_pca"]``, ``adata.obsm["X_umap"]``: embeddings

    Notes
    -----
    Unlike the melanoma dataset, no response harmonization is needed --
    arm labels are assigned at randomization and are clean.

    Raw files required in ``<data_dir>/raw/``:

    - ``GSE169246_TNBC_RNA.counts.mtx.gz``   (auto-downloadable)
    - ``GSE169246_TNBC_RNA.barcode.tsv.gz``  (auto-downloadable)
    - ``GSE169246_TNBC_RNA.feature.tsv.gz``  (auto-downloadable)
    - ``geo_metadata.csv``                   (must be generated manually)
    - ``mmc3.xlsx``                          (must be downloaded manually)

    Download raw files from:
    https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169246

    Reference: Zhang et al., Cancer Cell 2021.

    Examples
    --------
    >>> import sctrial as st
    >>> adata = st.load_tnbc_zhang(allow_download=True)
    >>> print(adata)
    """
    processing_params = {
        "version": "v4",
        "max_cells_per_participant_visit": max_cells_per_participant_visit,
        "seed": seed,
    }

    data_dir       = data_dir or _default_data_dir("tnbc_zhang")
    data_dir_path  = Path(data_dir)
    processed_path = data_dir_path / "processed" / processed_name

    # ── Try to load cached processed file ─────────────────────────
    if not force_reprocess and processed_path.exists():
        adata = ad.read_h5ad(processed_path)
        prev  = adata.uns.get("processing_params", {})
        if prev:
            if _params_match(prev, processing_params):
                logger.info(
                    f"Loaded TNBC Zhang dataset (GSE169246): "
                    f"{adata.n_obs:,} cells, {adata.n_vars:,} genes"
                )
                return adata
            logger.info("Processed file parameters differ; reprocessing.")
            logger.debug(f"  Stored:  {prev}")
            logger.debug(f"  Current: {processing_params}")
        else:
            warnings.warn(
                "Cached file lacks processing_params metadata; cannot verify it "
                "matches current settings. Consider reprocessing with "
                "force_reprocess=True.",
                UserWarning,
                stacklevel=2,
            )
            logger.info(
                f"Loaded TNBC Zhang dataset (GSE169246): "
                f"{adata.n_obs:,} cells, {adata.n_vars:,} genes"
            )
            return adata

    # Check scanpy is available before downloading
    try:
        import scanpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "scanpy is required for TNBC dataset processing. "
            "Install with: pip install sctrial[plots]  or  pip install scanpy"
        ) from None

    #_ Locate or download raw files
    raw_dir = data_dir_path / "raw"
    _REQUIRED_RAW = [
        "GSE169246_TNBC_RNA.counts.mtx.gz",
        "GSE169246_TNBC_RNA.barcode.tsv.gz",
        "GSE169246_TNBC_RNA.feature.tsv.gz",
        "geo_metadata.csv",
    ]
    raw_dir_resolved = _resolve_dir_with_files(str(raw_dir), _REQUIRED_RAW)
    missing = [f for f in _REQUIRED_RAW if not (raw_dir_resolved / f).exists()]

    if missing:
        if not allow_download:
            raise FileNotFoundError(
                f"Missing raw file(s): {missing}\n"
                f"Expected in: {raw_dir_resolved}\n"
                "Download from GEO: "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169246\n"
                "Or set allow_download=True to fetch automatically.\n\n"
                "Note: geo_metadata.csv must be generated manually from the GEO "
                "SOFT file. See the TNBC tutorial for instructions."
            )

        # Download the three GEO MTX files (geo_metadata.csv cannot be auto-downloaded)
        raw_dir_resolved.mkdir(parents=True, exist_ok=True)
        _GEO_BASE = (
            "https://www.ncbi.nlm.nih.gov/geo/download/"
            "?acc=GSE169246&format=file&file="
        )
        _tnbc_geo_files = [
            (
                "GSE169246_TNBC_RNA.counts.mtx.gz",
                _GEO_BASE + "GSE169246%5FTNBC%5FRNA%2Ecounts%2Emtx%2Egz",
                "counts matrix",
            ),
            (
                "GSE169246_TNBC_RNA.barcode.tsv.gz",
                _GEO_BASE + "GSE169246%5FTNBC%5FRNA%2Ebarcode%2Etsv%2Egz",
                "barcodes file",
            ),
            (
                "GSE169246_TNBC_RNA.feature.tsv.gz",
                _GEO_BASE + "GSE169246%5FTNBC%5FRNA%2Efeature%2Etsv%2Egz",
                "features file",
            ),
        ]
        for fname, url, label in _tnbc_geo_files:
            dest = raw_dir_resolved / fname
            if not dest.exists():
                _download_file(url, dest, label)

        # geo_metadata.csv cannot be auto-downloaded -- raise a clear error
        if not (raw_dir_resolved / "geo_metadata.csv").exists():
            raise FileNotFoundError(
                "geo_metadata.csv is missing and cannot be downloaded automatically.\n"
                "This file maps patient IDs to treatment arms and must be generated "
                "from the GEO SOFT file (GSE169246_family.soft.gz).\n"
                "See the TNBC tutorial notebook for instructions on generating it "
                "using parse_geo_soft.py."
            )

    logger.info("Processing raw TNBC data (this may take several minutes)...")
    adata = _process_tnbc_raw(
        raw_dir_resolved,
        max_cells_per_participant_visit=max_cells_per_participant_visit,
        seed=seed,
    )
    adata.uns["processing_params"] = processing_params

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_path)
    logger.info(f"Saved processed file: {processed_path}")
    logger.info(
        f"Loaded TNBC Zhang dataset (GSE169246): "
        f"{adata.n_obs:,} cells, {adata.n_vars:,} genes"
    )
    return adata

def harmonize_response(adata: ad.AnnData, *, force: bool = False) -> ad.AnnData:
    """Create a ``response_harmonized`` column with consistent labels.

    Maps various responder/non-responder column names and label formats
    (e.g. "R"/"NR", "Responder"/"Non-responder") to a standard vocabulary:
    ``"Responder"`` and ``"Non-responder"``.

    Parameters
    ----------
    adata : AnnData
        Must contain one of: ``response``, ``Response``, or
        ``clinical_response`` in ``.obs``.
    force : bool
        If True, recompute even when the column already exists.

    Returns
    -------
    AnnData
        The input AnnData with ``response_harmonized`` added to ``.obs``.
    """
    if force and "response_harmonized" in adata.obs.columns:
        del adata.obs["response_harmonized"]

    if "response_harmonized" in adata.obs.columns:
        if "participant_id" in adata.obs.columns:
            n_per = adata.obs.groupby("participant_id")["response_harmonized"].nunique()
            if (n_per > 1).any():
                pid_resp = adata.obs.groupby("participant_id")["response_harmonized"].agg(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
                )
                adata.obs["response_harmonized"] = adata.obs["participant_id"].map(pid_resp)
        return adata

    mapping = {
        "responder": "Responder",
        "Responder": "Responder",
        "R": "Responder",
        "non-responder": "Non-responder",
        "Non-responder": "Non-responder",
        "NR": "Non-responder",
        "nonresponder": "Non-responder",
    }
    for col in ("response", "Response", "clinical_response"):
        if col in adata.obs.columns:
            adata.obs["response_harmonized"] = (
                adata.obs[col].astype(str).map(lambda x: mapping.get(x, x))
            )
            break

    if "response_harmonized" in adata.obs.columns and "participant_id" in adata.obs.columns:
        pid_resp = adata.obs.groupby("participant_id")["response_harmonized"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        adata.obs["response_harmonized"] = adata.obs["participant_id"].map(pid_resp)

    return adata


def ensure_fdr(df: pd.DataFrame, p_col: str = "p_time", fdr_col: str = "FDR_time") -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR column for a p-value column.

    Parameters
    ----------
    df
        DataFrame containing the p-value column.
    p_col
        Column name in `df` to use for p-value column.
    fdr_col
        Column name in `df` to use for FDR-corrected p-value column.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with the FDR-corrected p-value column added.
    """
    if df.empty:
        return df
    if fdr_col in df.columns:
        return df
    if p_col in df.columns:
        mask = df[p_col].notna()
        df[fdr_col] = np.nan
        if mask.sum() > 0:
            df.loc[mask, fdr_col] = multipletests(df.loc[mask, p_col], method="fdr_bh")[1]
    return df
