from __future__ import annotations

import gzip
import urllib.error
import logging
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


def _get_counts_matrix(adata: ad.AnnData) -> tuple[np.ndarray | None, str | None]:
    return get_counts_matrix(adata)



def load_sade_feldman(
    data_dir: str = "data/sade_feldman",
    processed_name: str = "sade_feldman_processed_v5.h5ad",
    max_cells_per_participant_visit: int | None = None,
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
) -> ad.AnnData:
    """Load and preprocess Sade-Feldman melanoma immunotherapy dataset (GSE120575).
    Args:
        data_dir: Directory to store the data.
        processed_name: Name of the processed file.
        max_cells_per_participant_visit: Maximum number of cells per participant-visit.
        seed: Random seed.
        allow_download: Whether to allow downloading the data from GEO.
        force_reprocess: Whether to force reprocessing the data.
    Returns:
        ad.AnnData: The processed AnnData object.
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
        "version": "v5",
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

    for p in [tpm_path, meta_path]:
        if not p.exists():
            if not allow_download:
                raise FileNotFoundError(
                    f"Missing file: {p}. Download from GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575"
                )
            data_dir_path.mkdir(parents=True, exist_ok=True)
            url1 = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120575&format=file&file=GSE120575%5FSade%5FFeldman%5Fmelanoma%5Fsingle%5Fcells%5FTPM%5FGEO%2Etxt%2Egz'
            print(f"Downloading from {url1}...")
            try:
                urllib.request.urlretrieve(url1, str(tpm_path))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if tpm_path.exists():
                    tpm_path.unlink()
                raise RuntimeError(f"Failed to download TPM file from {url1}: {e}. Please download manually from {url1} and place it in {data_dir_path}") from e
            print(f'Successfully downloaded TPM file: {tpm_path}')
            url2 = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE120575&format=file&file=GSE120575%5Fpatient%5FID%5Fsingle%5Fcells%2Etxt%2Egz'
            print(f"Downloading from {url2}...")
            try:
                urllib.request.urlretrieve(url2, str(meta_path))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if meta_path.exists():
                    meta_path.unlink()
                raise RuntimeError(f"Failed to download metadata file from {url2}: {e}. Please download manually from {url2} and place it in {data_dir_path}") from e
            print(f'Successfully downloaded metadata file: {meta_path}')
    print("Processing raw data (this may take a minute)...")
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

    adata.obs["cell_type"] = "Immune"

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
    Args:
        data_path: Path to the raw data file.
        processed_name: Name of the processed file.
        seed: Random seed.
        allow_download: Whether to allow downloading the data from https://www.ebi.ac.uk/biostudies/files/E-MTAB-10026/covid_portal_210320_with_raw.h5ad.
        force_reprocess: Whether to force reprocessing the data.
    Returns:
        ad.AnnData: The processed AnnData object.
    """
    data_path_resolved = _resolve_file(data_path)
    data_root = data_path_resolved.parent.parent if data_path_resolved.exists() else Path("data")
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
        print(f"Downloading from {url}...")
        try:
            urllib.request.urlretrieve(url, str(data_path_resolved))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if data_path_resolved.exists():
                data_path_resolved.unlink()
            raise RuntimeError(f"Failed to download data from {url}: {e}. Please download manually from {url} and place it in {data_path_resolved.parent}") from e
        print(f'Successfully downloaded data: {data_path_resolved}')
    print("Processing raw data...")
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
    Args:
        data_dir: Directory to store the data.
        processed_name: Name of the processed file.
        max_participants: Maximum number of participants.
        max_cells_per_group: Maximum number of cells per group.
        seed: Random seed.
        allow_download: Whether to allow downloading the data from GEO.
        force_reprocess: Whether to force reprocessing the data.
    Returns:
        ad.AnnData: The processed AnnData object.
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

    for p in [barcodes_path, feats_path, pheno_path, mtx_path]:
        if not p.exists():
            if not allow_download:
                raise FileNotFoundError(f"Missing file: {p}")
            data_dir_path.mkdir(parents=True, exist_ok=True)
            url1 = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE171964&format=file&file=GSE171964%5Fbarcodes%5Fv2%2Etsv%2Egz'
            print(f"Downloading from {url1}...")
            try:
                urllib.request.urlretrieve(url1, str(barcodes_path))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if barcodes_path.exists():
                    barcodes_path.unlink()
                raise RuntimeError(f"Failed to download barcodes file from {url1}: {e}. Please download manually from {url1} and place it in {data_dir_path}") from e
            print(f'Successfully downloaded barcodes file: {barcodes_path}')
            url2 = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE171964&format=file&file=GSE171964%5Ffeats%5Fv2%2Etsv%2Egz'
            print(f"Downloading from {url2}...")
            try:
                urllib.request.urlretrieve(url2, str(feats_path))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if feats_path.exists():
                    feats_path.unlink()
                raise RuntimeError(f"Failed to download features file from {url2}: {e}. Please download manually from {url2} and place it in {data_dir_path}") from e
            print(f'Successfully downloaded features file: {feats_path}')
            url3 = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE171964&format=file&file=GSE171964%5Fgeo%5Fpheno%5Fv2%2Ecsv%2Egz'
            print(f"Downloading from {url3}...")
            try:
                urllib.request.urlretrieve(url3, str(pheno_path))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if pheno_path.exists():
                    pheno_path.unlink()
                raise RuntimeError(f"Failed to download pheno file from {url3}: {e}. Please download manually from {url3} and place it in {data_dir_path}") from e
            print(f'Successfully downloaded pheno file: {pheno_path}')
            url4 = 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE171964&format=file&file=GSE171964%5Fcountsmatrix%5Fv2%2Emtx%2Egz'
            print(f"Downloading from {url4}...")
            try:
                urllib.request.urlretrieve(url4, str(mtx_path))
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                if mtx_path.exists():
                    mtx_path.unlink()
                raise RuntimeError(f"Failed to download mtx file from {url4}: {e}. Please download manually from {url4} and place it in {data_dir_path}") from e
            print(f'Successfully downloaded mtx file: {mtx_path}')
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
    """Count participants with data at both visits."""
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

    Returns:
      - paired_ids: set of participant IDs with both visits (and non-NaN features if provided)
      - dropped_ids: list of participant IDs dropped by validation
      - n_paired: count of paired_ids
      - n_total: total unique participants
    """
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
        df_pv = (
            obs.groupby([participant_col, visit_col], observed=True)[list(features)]
            .mean()
            .reset_index()
        )
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
    """Map fine-grained cell types to coarse lineages (COVID-19 example)."""
    ct_lower = str(ct).lower()
    if "cd4" in ct_lower or "th1" in ct_lower or "th2" in ct_lower or "treg" in ct_lower:
        return "CD4_T"
    if "cd8" in ct_lower or "cytotoxic" in ct_lower:
        return "CD8_T"
    if "nk" in ct_lower or "natural killer" in ct_lower:
        return "NK"
    if "b cell" in ct_lower or "plasma" in ct_lower or "b_cell" in ct_lower:
        return "B_cells"
    if "mono" in ct_lower or "cd14" in ct_lower or "cd16" in ct_lower:
        return "Monocytes"
    if "dc" in ct_lower or "dendritic" in ct_lower:
        return "DCs"
    return "Other"


def ensure_fdr(df: pd.DataFrame, p_col: str = "p_time", fdr_col: str = "FDR_time") -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR column for a p-value column."""
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
