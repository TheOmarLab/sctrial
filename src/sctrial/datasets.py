from __future__ import annotations

import gzip
import logging
import urllib.error
import urllib.request
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from statsmodels.stats.multitest import multipletests

from .utils import get_counts_matrix

logger = logging.getLogger(__name__)


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
        # G1 markers from publication: IGKC, LTB, LY9, SELL, TCF7, CCR7
        "MS4A1", "CD79A", "CD79B", "BANK1", "CD74", "CD19",
        "PAX5", "BLK", "IGKC", "LTB", "LY9",
    },
    "Plasma cell": {
        # G2 cluster
        "MZB1", "SDC1", "XBP1", "JCHAIN", "PRDM1",
        "IGKC", "IGHG1",
    },
    "NK cell": {
        "KLRD1", "KLRF1", "KLRB1", "GNLY", "PRF1",
        "NKG7", "GZMB", "NCAM1", "FCGR3A",
    },
    "Monocyte/Macrophage": {
        # G3 cluster
        "CD14", "CD68", "LYZ", "CST3", "S100A8", "S100A9",
        "C1QA", "C1QB", "C1QC", "MRC1", "CSF1R", "FCGR3A",
    },
    "Dendritic cell": {
        # G4 cluster
        "FCER1A", "CLEC10A", "CD1C", "ITGAX",
        "HLA-DRA", "HLA-DQA1",
    },
}

# Annotation parameters (following Diab_duod conventions)
_ANNOT_TOP_N = 50          # top markers per cluster from Wilcoxon
_ANNOT_MIN_LFC = 0.25      # minimum log fold-change
_ANNOT_MAX_FDR = 0.1       # maximum adjusted p-value
_ANNOT_SECOND_DELTA = 0.25 # delta to flag ambiguous clusters
_ANNOT_MIN_ACCEPT = 0.3    # minimum weighted score to accept a label


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
    1. Compute Leiden clusters on log1p-normalised expression.
    2. Find differentially expressed markers per cluster (Wilcoxon).
    3. Score each cluster against canonical immune marker sets using
       a rank-weighted scoring function.
    4. Assign the best-scoring cell type to each cluster.

    Parameters
    ----------
    adata : AnnData
        Must contain expression values (TPM) in ``adata.X`` and gene names
        in ``adata.var_names``.

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

    # PCA -> neighbors -> Leiden
    logger.info("    Computing PCA for cell-type annotation...")
    sc.pp.highly_variable_genes(aw, n_top_genes=2000, flavor="seurat")
    sc.tl.pca(aw, n_comps=30)
    sc.pp.neighbors(aw, n_neighbors=15, n_pcs=20)
    sc.tl.leiden(aw, resolution=1.0)

    # Wilcoxon marker finding per cluster
    logger.info("    Finding cluster markers (Wilcoxon)...")
    sc.tl.rank_genes_groups(aw, groupby="leiden", method="wilcoxon",
                            n_genes=_ANNOT_TOP_N)

    clusters = sorted(aw.obs["leiden"].unique(), key=int)
    cluster_labels: dict[str, str] = {}

    for cl in clusters:
        # Extract marker table for this cluster
        result = aw.uns["rank_genes_groups"]
        idx = list(result["names"].dtype.names).index(cl)
        names = [result["names"][i][idx] for i in range(len(result["names"]))]
        lfcs = [result["logfoldchanges"][i][idx]
                for i in range(len(result["logfoldchanges"]))]
        padjs = [result["pvals_adj"][i][idx]
                 for i in range(len(result["pvals_adj"]))]

        df_markers = pd.DataFrame({
            "names": names,
            "logfoldchanges": lfcs,
            "pvals_adj": padjs,
            "rank": np.arange(1, len(names) + 1),
        })

        # Filter by LFC and FDR
        df_markers = df_markers[
            (df_markers["logfoldchanges"] >= _ANNOT_MIN_LFC)
            & (df_markers["pvals_adj"] <= _ANNOT_MAX_FDR)
        ]

        # If strict filtering yields no markers, use unfiltered markers
        if df_markers.empty:
            df_markers = pd.DataFrame({
                "names": names,
                "logfoldchanges": lfcs,
                "pvals_adj": padjs,
                "rank": np.arange(1, len(names) + 1),
            })

        # Score against each cell type
        label_scores: dict[str, float] = {}
        for ct, gene_set in _IMMUNE_MARKERS.items():
            score, _ = _weighted_marker_score(df_markers, gene_set)
            if score > 0:
                label_scores[ct] = score

        if not label_scores:
            # Fallback: score with unfiltered markers (no LFC/FDR filter)
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
    return (data.min() >= 0) and (data.max() < 50) and (not np.allclose(data, np.round(data), atol=1e-3))


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
    data_dir: str = "data/sade_feldman",
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
    data_dir_path = _resolve_dir_with_files(
        data_dir,
        [
            "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz",
            "GSE120575_patient_ID_single_cells.txt.gz",
        ],
    )
    processed_path = data_dir_path.parent / "processed" / processed_name

    processing_params = {
        "version": "v6",
        "max_cells_per_participant_visit": max_cells_per_participant_visit,
        "seed": seed,
        "assay": "TPM",
    }

    if processed_path.exists() and not force_reprocess:
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if _params_match(prev, processing_params):
            logger.info(f"Loaded processed Sade-Feldman dataset: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
            return adata
        # Show what's different for debugging
        logger.info("Processed file parameters differ; reprocessing.")
        logger.debug(f"  Stored: {prev}")
        logger.debug(f"  Current: {processing_params}")

    tpm_path = data_dir_path / "GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz"
    meta_path = data_dir_path / "GSE120575_patient_ID_single_cells.txt.gz"

    _GEO_BASE = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120575&format=file&file="
    _sade_feldman_files = [
        (tpm_path, _GEO_BASE + "GSE120575%5FSade%5FFeldman%5Fmelanoma%5Fsingle%5Fcells%5FTPM%5FGEO%2Etxt%2Egz", "TPM file"),
        (meta_path, _GEO_BASE + "GSE120575%5Fpatient%5FID%5Fsingle%5Fcells%2Etxt%2Egz", "metadata file"),
    ]
    missing = [(p, url, label) for p, url, label in _sade_feldman_files if not p.exists()]
    if missing:
        if not allow_download:
            names = ", ".join(str(p) for p, _, _ in missing)
            raise FileNotFoundError(
                f"Missing file(s): {names}. Download from GEO: "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575"
            )
        data_dir_path.mkdir(parents=True, exist_ok=True)
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
    meta = meta.rename(columns={
        "title": "sample_id",
        "characteristics: patinet ID (Pre=baseline; Post= on treatment)": "patient_raw",
        "characteristics: response": "response",
    })
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
        logger.info(f"Stratified sampling: {adata.n_obs:,} cells (max {max_cells_per_participant_visit} per participant-visit)")
    else:
        logger.info(f"Using full dataset: {adata.n_obs:,} cells (no subsampling)")

    adata.layers["tpm"] = adata.X.copy()
    adata.layers["log1p_tpm"] = adata.X.copy() if _looks_log1p(adata.X) else np.log1p(adata.X)

    # Annotate cell types using cluster-level weighted marker scoring
    # (adapted from Diab_duod project methodology).
    # Requires scanpy (optional dependency in [plots] extra).
    try:
        import scanpy as sc  # noqa: F401 — check availability before heavy work
    except ImportError:
        raise ImportError(
            "scanpy is required for Sade-Feldman cell type annotation. "
            "Install with: pip install sctrial[plots]"
        ) from None
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
    data_path: str = "data/stephenson/covid_portal_210320_with_raw.h5ad",
    processed_name: str = "stephenson_covid19_v3.h5ad",
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load and preprocess Stephenson COVID-19 dataset (E-MTAB-10026).

    Parameters
    ----------
    data_path : str
        Path to the raw h5ad data file.
    processed_name : str
        Filename for the cached processed h5ad file.
    seed : int
        Random seed for reproducibility.
    allow_download : bool
        If True, download the data file automatically when missing.
    force_reprocess : bool
        If True, reprocess even when a cached file exists.

    Returns
    -------
    AnnData
        The processed AnnData object.
    """
    data_path_resolved = _resolve_file(data_path)
    # Derive processed cache location from the user-provided path so it
    # respects custom directory layouts even when the raw file is missing.
    data_root = Path(data_path).parent.parent if not data_path_resolved.exists() else data_path_resolved.parent.parent
    processed_path = data_root / "processed" / processed_name

    if processed_path.exists() and not force_reprocess:
        adata = ad.read_h5ad(processed_path)
        logger.info(f"Loaded cached file: {processed_path}")
        logger.info(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
        return adata

    if not data_path_resolved.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"Data not found at {data_path_resolved}. Download from: https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/"
            )
        data_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        url = "https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/covid_portal_210320_with_raw.h5ad"
        _download_file(url, data_path_resolved, "Stephenson COVID-19 data")
    logger.info("Processing raw data...")
    adata = ad.read_h5ad(data_path_resolved)

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

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(processed_path)
    logger.info(f"  Saved: {processed_path}")
    logger.info(f"  Final: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata


def load_vaccine_gse171964(
    data_dir: str = "data/vaccine_gse171964",
    processed_name: str = "vaccine_gse171964_day0_day7.h5ad",
    max_participants: int | None = 30,
    max_cells_per_group: int | None = 200,
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
    data_dir_path = _resolve_dir_with_files(
        data_dir,
        [
            "GSE171964_barcodes_v2.tsv.gz",
            "GSE171964_feats_v2.tsv.gz",
            "GSE171964_geo_pheno_v2.csv.gz",
            "GSE171964_countsmatrix_v2.mtx.gz",
        ],
    )
    processed_path = data_dir_path.parent / "processed" / processed_name

    processing_params = {
        "version": "v2",
        "max_participants": max_participants,
        "max_cells_per_group": max_cells_per_group,
        "seed": seed,
        "days": [0, 7],
    }

    if processed_path.exists() and not force_reprocess:
        adata = ad.read_h5ad(processed_path)
        prev = adata.uns.get("processing_params", {})
        if _params_match(prev, processing_params):
            logger.info(f"Loaded processed vaccine dataset (GSE171964): {adata.n_obs} cells, {adata.n_vars} genes")
            logger.info(f"Processed file: {processed_path}")
            logger.info(f"Days: {adata.obs['day'].unique()}")
            logger.info(f"Participants: {adata.obs['pt_id'].nunique()}")
            logger.info(f"Cell types: {adata.obs['clustnm'].nunique()}")
            return adata
        logger.info("Processed file parameters differ; reprocessing.")
        logger.debug(f"  Stored: {prev}")
        logger.debug(f"  Current: {processing_params}")

    barcodes_path = data_dir_path / "GSE171964_barcodes_v2.tsv.gz"
    feats_path = data_dir_path / "GSE171964_feats_v2.tsv.gz"
    pheno_path = data_dir_path / "GSE171964_geo_pheno_v2.csv.gz"
    mtx_path = data_dir_path / "GSE171964_countsmatrix_v2.mtx.gz"

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
            raise FileNotFoundError(f"Missing file(s): {names}")
        data_dir_path.mkdir(parents=True, exist_ok=True)
        for dest, url, label in missing:
            _download_file(url, dest, label)
    barcodes = (
        pd.read_csv(barcodes_path, sep="\\s+", header=None, engine="python", skiprows=1)[1]
        .astype(str)
        .str.strip('\"')
        .tolist()
    )
    features = (
        pd.read_csv(feats_path, sep="\\s+", header=None, engine="python", skiprows=1)[1]
        .astype(str)
        .str.strip('\"')
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
        sampled = (
            adata.obs.groupby(grp, observed=True, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max_cells_per_group), random_state=seed))
        )
        adata = adata[sampled.index].copy()

    adata.layers["counts"] = adata.X.copy()
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
    participant_col: str = "participant_id"
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
    wide = (
        obs.groupby([participant_col, visit_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )
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
    wide = (
        obs.groupby([participant_col, visit_col], observed=True)
        .size()
        .unstack(fill_value=0)
    )
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
        valid_ids = None
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
