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
from collections.abc import Callable, Sequence
from io import StringIO
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from statsmodels.stats.multitest import multipletests

from .preprocessing import add_log1p_cpm_layer, drop_artifact_genes
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


# mmc3.xlsx is ~33 MB / 489,490 rows and takes ~1 min to parse, but is needed
# for both clinical response and per-cell annotations. Cache it per raw_dir.
_TNBC_MMC3_CACHE: dict[str, pd.DataFrame] = {}


def _read_tnbc_mmc3(raw_dir: Path) -> pd.DataFrame:
    """Read (and memoise) the per-cell sheet of mmc3.xlsx (Zhang 2021 Table S2).

    Columns include ``Cell barcode``, ``Patient``, ``Efficacy``, ``Origin``,
    ``Group``, ``Major celltype`` (4 immune categories) and ``Cluster`` (97
    author clusters). Row 0 is the table title, so the real header is row 1.
    """
    key = str(raw_dir.resolve())
    cached = _TNBC_MMC3_CACHE.get(key)
    if cached is not None:
        return cached
    mmc3_path = raw_dir / "mmc3.xlsx"
    if not mmc3_path.exists():
        raise FileNotFoundError(
            f"mmc3.xlsx not found at {mmc3_path}. "
            "Download it from the Zhang et al. 2021 paper supplementary materials "
            "and place it in the raw data directory."
        )
    df = pd.read_excel(mmc3_path, sheet_name="Single cell clustering", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    _TNBC_MMC3_CACHE[key] = df
    return df


def _load_tnbc_clinical(raw_dir: Path) -> pd.Series:
    """Read per-patient response from mmc3.xlsx (Zhang et al. 2021, Table S2).

    Returns a Series indexed by patient ID (e.g. 'P001') with values 'R' or 'NR'.
    CR/PR → R (responder), PD/SD → NR (non-responder), Na → NaN.
    """
    df = _read_tnbc_mmc3(raw_dir)
    pt_eff = (
        df[["Patient", "Efficacy"]]
        .drop_duplicates("Patient")
        .set_index("Patient")["Efficacy"]
    )
    return pt_eff.map(_TNBC_EFFICACY_MAP)


def _load_tnbc_published_labels(raw_dir: Path) -> pd.DataFrame:
    """Per-cell author annotations for GSE169246, indexed by cell barcode.

    The same mmc3.xlsx the loader already opens for clinical response also
    carries the authors' own annotations, which were previously discarded:
    ``Major celltype`` (T / B / Myeloid / ILC — every cell is immune, the
    deposit being CD45+ sorted) and ``Cluster`` (97 clusters such as
    ``t_CD8-CXCL13``). Barcodes are the same ``<16bp>.<Sample>`` strings the
    GEO matrix uses, so the join needs no identifier munging.
    """
    df = _read_tnbc_mmc3(raw_dir)
    cols = {c.lower(): c for c in df.columns}
    bc = cols.get("cell barcode")
    major = cols.get("major celltype")
    clus = cols.get("cluster")
    if bc is None or major is None:
        raise KeyError(
            f"mmc3.xlsx is missing expected annotation columns; found {list(df.columns)}"
        )
    keep = [c for c in (bc, major, clus) if c is not None]
    out = df[keep].drop_duplicates(subset=bc).set_index(bc)
    out.index = out.index.astype(str).str.strip()
    rename = {major: "cell_type_published"}
    if clus is not None:
        rename[clus] = "cluster_published"
    return out.rename(columns=rename)


def _tnbc_cluster_lineage(cluster: object) -> str | None:
    """Map a Zhang 2021 cluster name (e.g. ``t_CD4_Treg-FOXP3``) to a lineage.

    The authors' ``Major celltype`` column has only four levels (T / B / Myeloid
    / ILC), which is coarser than the analysis needs — it cannot separate CD4
    from CD8 or resolve Tregs. Their 97 ``Cluster`` labels are systematically
    named, so the lineage is recovered from the cluster name instead, keeping
    the authors' resolution. Clusters the authors deliberately leave unresolved
    (``t_Tn-LEF1`` naive T, ``t_Tact-IFI6`` activated T, ``t_Tprf-MKI67``
    proliferating T, and the ambiguous ``Mix`` bin) are NOT forced onto CD4 or
    CD8; they map to "T cell (unresolved)" / "Unassigned".
    """
    if cluster is None or (isinstance(cluster, float) and pd.isna(cluster)):
        return None
    name = str(cluster).strip()
    if not name:
        return None
    if name.lower() == "mix":
        return "Unassigned"
    # Strip the origin prefix ("t_" tumour / "b_" blood)
    body = re.sub(r"^[tb]_", "", name)
    low = body.lower()
    if low.startswith("cd4_treg") or low.startswith("treg"):
        return "Treg"
    if "treg" in low and low.startswith("cd4"):
        return "Treg"
    if low.startswith("cd4"):
        return "CD4 T cell"
    if low.startswith("cd8"):
        return "CD8 T cell"
    if low.startswith(("tn-", "tact-", "tprf-", "tem-", "tcm-")):
        return "T cell (unresolved)"
    if low.startswith("pb-"):
        return "Plasma cell"
    if low.startswith(("bn-", "bmem-", "bfoc-", "b-")):
        return "B cell"
    if low.startswith(("mono-", "macro-", "mphi-", "mφ-")):
        return "Monocyte/Macrophage"
    if low.startswith(("cdc", "mdc", "pdc", "dc")):
        return "Dendritic cell"
    if low.startswith("mast"):
        return "Mast cell"
    if low.startswith("ilc1"):
        # Group 1 ILC is the NK-cell compartment in this study's nomenclature.
        return "NK cell"
    if low.startswith(("ilc2", "ilc3")):
        return "ILC"
    return "Unassigned"


def _load_cart_published_metadata(raw_dir: Path) -> pd.DataFrame | None:
    """Per-cell author metadata for GSE290722, indexed by ``Cell.id``.

    This series-level supplementary file carries far more than cell types: it
    holds the authors' three-level annotation (``Compartment`` / ``Major_Alias``
    / ``Alias``), their per-cell clinical ``Response`` (LtR / R / NR / Unknown)
    — which is the publication's primary outcome and cannot be reconstructed
    from the expression matrices — the exact ``TimePoint_Final``, and ``Axicel``
    CAR-transgene read counts. Written by R's ``write.table``, so it is
    space-delimited with quoted fields, not comma-separated.

    Returns None when the file is absent, so the loader degrades to de-novo
    annotation rather than failing.
    """
    path = raw_dir / "GSE290722_metadata.csv.gz"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=" ")
    df.columns = [str(c).strip() for c in df.columns]
    key = next((c for c in df.columns if c.lower() in ("cell.id", "cell_id")), None)
    if key is None:
        logger.warning("GSE290722_metadata.csv.gz has no Cell.id column; ignoring.")
        return None
    return df.drop_duplicates(subset=key).set_index(key)


# ---------------------------------------------------------------------------
# Published per-cell annotations
# ---------------------------------------------------------------------------
# Where the source publication deposited per-cell cell-type labels, those labels
# are the primary annotation: they are the authors' own, peer-reviewed, and far
# finer than a generic marker panel can recover. The de-novo marker-based labels
# are retained alongside (``cell_type_denovo``) purely as a reproducibility
# check, and ``annotation_source`` records which is which.

# Sade-Feldman et al., Cell 2018, Table S1 (mmc1.xlsx), sheet
# "Gene marker-Fig1B-C" column headers -- read verbatim from the workbook.
_SF_CLUSTER_NAMES: dict[int, str] = {
    1: "B cells",
    2: "Plasma cells",
    3: "Monocytes/Macrophages",
    4: "Dendritic cells",
    5: "Lymphocytes",
    6: "Exhausted CD8 T cells",
    7: "Regulatory T cells",
    8: "Cytotoxicity (Lymphocytes)",
    9: "Exhausted/HS CD8 T cells",
    10: "Memory T cells",
    11: "Lymphocytes exhausted/cell-cycle",
}

# GEO cell names carry FACS sort-fraction / lane suffixes that the author tables
# either omit or replace with the sort gate (e.g. GEO "A10_P1_M39_T_enriched" vs
# author "A10_P1_M39_DN1"). Stripping these yields a comparable base key.
_SF_SUFFIX_RE = re.compile(
    r"_(T_enriched|myeloid_enriched|T_cell_enriched|DN1|DN2|DN|DP1|DP2|DP|L00\d)$",
    re.IGNORECASE,
)


def _sf_cell_base(name: str) -> str:
    """Strip trailing sort-fraction/lane suffixes from a Sade-Feldman cell name."""
    prev = str(name)
    while True:
        cur = _SF_SUFFIX_RE.sub("", prev)
        if cur == prev:
            return cur
        prev = cur


def _join_published_labels(
    target_names: Sequence[str],
    label_map: dict[str, object],
    *,
    normalise: Callable[[str], str] | None = None,
    label_desc: str = "published labels",
) -> list[object]:
    """Align *label_map* onto *target_names*, exact matches first.

    A naive exact join loses cells whenever the deposited matrix and the author
    table spell the same cell differently. A naive normalised join instead
    creates collisions, because stripping a suffix can map two distinct cells
    onto one key. Doing both in order -- exact first, then normalised on the
    unmatched remainder only -- is unambiguous: every cell that already matched
    exactly is removed from contention before the fuzzy pass runs, which for
    Sade-Feldman leaves exactly one candidate per base key and recovers 100%.

    Returns a list aligned to *target_names*, with ``None`` where no unambiguous
    label exists (never a guess).
    """
    out: list[object] = [None] * len(target_names)
    matched_keys: set[str] = set()

    # Stage 1 -- exact
    for i, name in enumerate(target_names):
        if name in label_map:
            out[i] = label_map[name]
            matched_keys.add(name)
    n_exact = sum(v is not None for v in out)

    n_fuzzy = 0
    n_ambiguous = 0
    if normalise is not None and n_exact < len(target_names):
        # Remaining author entries, bucketed by normalised key
        rem_by_base: dict[str, list[object]] = {}
        for key, val in label_map.items():
            if key in matched_keys:
                continue
            rem_by_base.setdefault(normalise(key), []).append(val)
        # Remaining target cells, bucketed by normalised key
        tgt_by_base: dict[str, list[int]] = {}
        for i, name in enumerate(target_names):
            if out[i] is None:
                tgt_by_base.setdefault(normalise(name), []).append(i)
        for base_key, idxs in tgt_by_base.items():
            cands = rem_by_base.get(base_key, [])
            if len(idxs) == 1 and len(cands) == 1:
                out[idxs[0]] = cands[0]
                n_fuzzy += 1
            else:
                n_ambiguous += len(idxs)

    n_total = sum(v is not None for v in out)
    logger.info(
        "  %s: matched %d/%d cells (%.2f%%) [exact=%d, suffix-normalised=%d, "
        "ambiguous/unmatched=%d]",
        label_desc, n_total, len(target_names),
        100.0 * n_total / max(len(target_names), 1),
        n_exact, n_fuzzy, len(target_names) - n_total,
    )
    if n_ambiguous:
        logger.warning(
            "  %s: %d cell(s) left unlabelled because the normalised key was "
            "ambiguous; they are NOT guessed.", label_desc, n_ambiguous,
        )
    return out


def _load_sade_feldman_published_labels(
    raw_dir: Path, cell_names: Sequence[str]
) -> dict[str, list[object]]:
    """Per-cell author labels for GSE120575 from the Cell 2018 supplements.

    Table S1 (mmc1) gives the unsupervised G1-G11 cluster for all 16,291 cells;
    Table S2 (mmc2) gives the CD8_G (memory-like) vs CD8_B (dysfunctional)
    dichotomy that is the publication's headline result; Table S4 (mmc4) gives
    its six-way refinement CD8_1..CD8_6. Returns empty lists for any table that
    is absent, so the loader degrades gracefully rather than failing.
    """
    out: dict[str, list[object]] = {}

    mmc1 = raw_dir / "mmc1.xlsx"
    if mmc1.exists():
        df = pd.read_excel(mmc1, sheet_name="Cluster annotation-Fig1B-C")
        df.columns = [str(c).strip() for c in df.columns]
        cmap = {
            str(n).strip(): int(c)
            for n, c in zip(df["Cell Name"], df["Cluster number"])
            if pd.notna(c)
        }
        clusters = _join_published_labels(
            cell_names, cmap, normalise=_sf_cell_base,
            label_desc="Table S1 G1-G11 clusters",
        )
        out["cluster_published"] = [
            f"G{c}" if c is not None else None for c in clusters
        ]
        out["cell_type_published"] = [
            _SF_CLUSTER_NAMES.get(c) if c is not None else None for c in clusters
        ]

    for fname, sheet, key in (
        ("mmc2.xlsx", "Cluster annotation-Fig2A-B", "cd8_state"),
        ("mmc4.xlsx", "Cluster annotation-Fig4A-B", "cd8_subcluster"),
    ):
        path = raw_dir / fname
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        name_col = next(c for c in df.columns if "name" in c.lower())
        clus_col = next(c for c in df.columns if "cluster" in c.lower())
        cmap = {
            str(n).strip(): str(c).strip()
            for n, c in zip(df[name_col], df[clus_col])
            if pd.notna(c)
        }
        out[key] = _join_published_labels(
            cell_names, cmap, normalise=_sf_cell_base,
            label_desc=f"{fname} {key}",
        )
    return out


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
        "version": "v8",
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

    # Author per-cell annotations live in the Cell 2018 supplements, not in GEO
    # (the GEO deposit carries no cell-type field at all). Fetched from the
    # publisher so the annotation stays reproducible from public sources alone.
    # Non-fatal: the loader still works without them, falling back to de-novo
    # labels only, so a publisher outage cannot break the dataset.
    _ELS_BASE = "https://ars.els-cdn.com/content/image/1-s2.0-S0092867418313941-"
    for fname, label in (
        ("mmc1.xlsx", "Table S1 (G1-G11 cluster annotation)"),
        ("mmc2.xlsx", "Table S2 (CD8_G/CD8_B states)"),
        ("mmc4.xlsx", "Table S4 (CD8_1-6 subclusters)"),
    ):
        dest = raw_dir_resolved / fname
        if dest.exists() or not allow_download:
            continue
        try:
            _download_file(_ELS_BASE + fname, dest, label)
        except Exception as exc:  # noqa: BLE001 - optional enrichment
            logger.warning(
                "Could not download %s (%s); published labels for this table "
                "will be unavailable.", fname, exc,
            )
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

    # NOTE: do NOT filter cells on a `sample_id` name pattern. An earlier
    # `^[A-Z]\d+_P\d+_M\d+` filter silently discarded 3,108 of the 16,291
    # published QC-passing cells (19.1%) and 7 of 32 patients, purely because
    # some lesions use a different naming convention (the whole `MMD*` lesion
    # series, plus `m15` whose sample token is lower-case). The drop was not
    # compositionally neutral -- it depleted cluster G6 (exhaustion-enriched),
    # which is the publication's primary axis. Validity is already enforced
    # downstream by the response filter (which removes the 38 non-cell footer
    # rows) and by the intersection with the TPM matrix columns below, so
    # removing the pattern filter recovers exactly the published cohort:
    # 16,291 cells / 32 participants / 11 Pre+Post paired.
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
    # De-novo marker-based labels are computed first and always retained as a
    # reproducibility check, then superseded by the authors' own per-cell labels
    # where those are available (see _load_sade_feldman_published_labels).
    # Uses the Leiden clusters computed above (same embedding as UMAP).
    logger.info("Annotating cell types from marker genes (de-novo)...")
    adata.obs["cell_type_denovo"] = _annotate_immune_celltypes(adata)

    logger.info("Joining published per-cell annotations (Cell 2018 supplements)...")
    published = _load_sade_feldman_published_labels(
        raw_dir_resolved, list(adata.obs_names)
    )
    for col, values in published.items():
        adata.obs[col] = pd.Series(values, index=adata.obs_names, dtype="object")

    if "cell_type_published" in adata.obs and adata.obs["cell_type_published"].notna().any():
        # Authors' unsupervised G1-G11 annotation is the primary cell type.
        adata.obs["cell_type"] = (
            adata.obs["cell_type_published"].fillna("Unassigned").astype(str)
        )
        adata.obs["annotation_source"] = "publication"
        adata.uns["annotation_source"] = (
            "Sade-Feldman et al., Cell 2018 — Table S1 (mmc1.xlsx) unsupervised "
            "clusters G1-G11; Table S2 CD8_G/CD8_B; Table S4 CD8_1-6"
        )
    else:
        adata.obs["cell_type"] = adata.obs["cell_type_denovo"].astype(str)
        adata.obs["annotation_source"] = "de-novo"
        adata.uns["annotation_source"] = (
            "de-novo marker scoring (published supplements unavailable)"
        )
        logger.warning(
            "Published Sade-Feldman labels unavailable; falling back to de-novo "
            "annotation. Re-run with allow_download=True to fetch mmc1/2/4."
        )

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
    processed_name: str = "stephenson_covid19_v5.h5ad",
    seed: int = 42,
    allow_download: bool = False,
    force_reprocess: bool = False,
    *,
    severity_keep: Sequence[str] = ("Mild", "Severe"),
    require_numeric_dfo: bool = True,
    data_path: str | None = None,
) -> ad.AnnData:
    """Load and preprocess Stephenson COVID-19 dataset (E-MTAB-10026).

    Reference: Stephenson et al., "Single-cell multi-omics analysis of the immune
    response in COVID-19", Nat Med 2021;27(5):904-916. PMID 33879890,
    doi:10.1038/s41591-021-01329-2.

    .. warning::
       **This loader retains a minority of the deposited atlas.** The deposited
       object holds ~647k cells across ~120 individuals spanning asymptomatic,
       mild, moderate, severe and critical COVID-19 plus healthy, non-COVID
       respiratory-illness and IV-LPS control groups. Two filters are applied:

       1. ``severity_keep`` (default ``("Mild", "Severe")``) keeps two strata,
          which **excludes the asymptomatic, moderate and critical patients** and
          collapses an ordinal clinical variable to a binary contrast.
       2. ``require_numeric_dfo`` drops cells whose ``Days_from_onset`` is not
          numerically parseable (they would otherwise land in a ``"nan"`` bin).

       Together these retain roughly a third of the deposited cells. The exact
       per-step counts are recorded in ``adata.uns["cohort_funnel"]``. Widen
       ``severity_keep`` to analyse the full severity spectrum.

    Parameters
    ----------
    data_dir : str
        Directory containing (or to store) the raw data files.
    processed_name : str
        Filename for the cached processed h5ad file.
    seed : int
        Unused; this loader is deterministic and performs no subsampling. Kept
        for signature symmetry with the other loaders.
    severity_keep : Sequence[str]
        ``Status_on_day_collection_summary`` levels to retain.
    require_numeric_dfo : bool
        Drop cells whose days-from-onset is not numerically parseable.
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

    # This loader previously served ANY cached file whose processing_params was
    # merely non-empty, never comparing it to the current settings. The cache key
    # was effectively the filename, so a file produced by older code -- before the
    # artifact-gene QC, for instance -- was returned as if current. Match the
    # params like every sibling loader does.
    processing_params = {
        "version": "v5",
        "severity_keep": list(severity_keep),
        "dfo_bins": ["DFO_0-7", "DFO_8-14", "DFO_15+"],
        "qc_artifact_flag": True,
    }

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
        if _params_match(prev, processing_params):
            logger.info(f"Loaded cached file: {processed_path}")
            logger.info(f"  {adata.n_obs:,} cells, {adata.n_vars:,} genes")
            return adata
        logger.info("Processed file parameters differ; reprocessing.")

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
    # Record the cohort funnel so the (substantial) loss is recoverable from the
    # cached artefact instead of only appearing in transient log lines.
    funnel: dict[str, object] = {
        "n_cells_deposited": int(adata.n_obs),
        "n_individuals_deposited": int(obs["patient_id"].nunique()),
    }

    obs["severity"] = obs["Status_on_day_collection_summary"].astype(str)
    funnel["severity_levels_deposited"] = sorted(obs["severity"].unique().tolist())
    funnel["severity_levels_kept"] = list(severity_keep)
    funnel["severity_levels_dropped"] = sorted(
        set(obs["severity"].unique()) - set(severity_keep)
    )
    obs = obs[obs["severity"].isin(list(severity_keep))].copy()
    funnel["n_cells_after_severity"] = int(len(obs))
    logger.info(f"  After severity filter: {len(obs):,} cells")

    obs["dfo"] = pd.to_numeric(obs["Days_from_onset"], errors="coerce")
    n_unparseable = int(obs["dfo"].isna().sum())
    obs["dfo_bin"] = pd.cut(
        obs["dfo"],
        bins=[-np.inf, 7, 14, np.inf],
        labels=["DFO_0-7", "DFO_8-14", "DFO_15+"],
    ).astype(str)

    if require_numeric_dfo:
        # NOTE: pd.to_numeric(errors="coerce") turns non-numeric day values into
        # NaN, which .astype(str) then renders as the literal string "nan". Those
        # cells are dropped here for a METADATA-FORMATTING reason, not for QC, so
        # the count is logged and persisted rather than silently absorbed.
        if n_unparseable:
            logger.warning(
                "  %d cell(s) have non-numeric Days_from_onset and are dropped "
                "(require_numeric_dfo=True).", n_unparseable,
            )
        valid_dfo = obs["dfo_bin"].isin(["DFO_0-7", "DFO_8-14", "DFO_15+"])
        obs = obs[valid_dfo].copy()
    funnel["n_cells_dropped_unparseable_dfo"] = n_unparseable
    funnel["n_cells_after_dfo"] = int(len(obs))
    logger.info(f"  After DFO filter: {len(obs):,} cells")

    if "Collection_Day" in obs.columns:
        obs["collection_day"] = obs["Collection_Day"].astype(str)

    obs["participant_id"] = obs["patient_id"].astype(str)
    obs["celltype"] = obs["full_clustering"].astype(str)

    # `severity` is recorded per SAMPLE (status on day of collection) but is used
    # downstream as a fixed participant-level arm. Verify that assumption instead
    # of trusting it: a participant whose status changes between collections would
    # otherwise be silently assigned to two arms.
    per_pid = obs.groupby("participant_id")["severity"].nunique()
    inconsistent = sorted(per_pid[per_pid > 1].index.tolist())
    if inconsistent:
        logger.warning(
            "  %d participant(s) have >1 severity level across samples and are "
            "ambiguous as a fixed arm: %s", len(inconsistent), inconsistent,
        )
    funnel["participants_with_inconsistent_severity"] = inconsistent
    funnel["n_cells_final"] = int(len(obs))
    funnel["n_individuals_final"] = int(obs["participant_id"].nunique())

    adata = adata[obs.index].copy()
    adata.obs = obs
    # Normalise BEFORE dropping artifact genes: drop_artifact_genes documents that
    # it must run after normalisation so per-cell library sizes are computed over
    # the full gene set. This loader does no normalisation of its own, so the
    # log1p_cpm layer is built here rather than downstream from a matrix whose
    # ribosomal/histone counts have already been removed.
    add_log1p_cpm_layer(adata, counts_layer="counts", out_layer="log1p_cpm")
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)
    # The deposited object's own layer is a duplicate of `counts`; keep one copy.
    if "raw" in adata.layers:
        del adata.layers["raw"]
    adata.uns["cohort_funnel"] = funnel
    adata.uns["annotation_source"] = (
        "Stephenson et al., Nat Med 2021 — deposited `full_clustering` labels "
        "(atlas-provided, used verbatim)"
    )
    adata.obs["annotation_source"] = "publication"
    adata.uns["paper"] = "Stephenson et al., Nat Med 2021"
    adata.uns["pmid"] = "33879890"
    adata.uns["doi"] = "10.1038/s41591-021-01329-2"
    adata.uns["processing_params"] = processing_params

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
    *,
    days: Sequence[int] = (0, 7),
    min_cells_per_participant_visit: int = 50,
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
        "version": "v4",
        "max_participants": max_participants,
        "max_cells_per_group": max_cells_per_group,
        "seed": seed,
        "days": list(days),
        "adt_split": True,
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

    adata = adata[adata.obs["day"].isin(list(days))].copy()

    import scanpy as sc  # local import (optional dependency), like the other loaders

    # ── Split off CITE-seq ADT protein features ───────────────────────────────
    # GSE171964 is CITE-seq: the deposited feature space mixes mRNA genes with
    # antibody-derived tags. Left in place they are treated as genes, so their
    # counts enter the CP10K library-size denominator and every gene value is
    # scaled by (1 - ADT fraction). That fraction differs between Day 0 and Day 7,
    # so the bias does NOT cancel in the paired within-arm contrast -- it is
    # confounded with the loader's only estimand. Proteins are moved to
    # `.obsm["protein"]` (raw) and excluded from the gene matrix entirely.
    is_adt = adata.var_names.str.contains("_ADT", case=False, regex=False)
    n_adt = int(is_adt.sum())
    if n_adt:
        X_adt = adata[:, is_adt].X
        X_adt = X_adt.toarray() if sp.issparse(X_adt) else np.asarray(X_adt)
        adata.obsm["protein"] = pd.DataFrame(
            X_adt,
            index=adata.obs_names,
            columns=adata.var_names[is_adt].tolist(),
        )
        adata = adata[:, ~is_adt].copy()
        logger.info(
            "Moved %d CITE-seq ADT protein feature(s) to .obsm['protein']; "
            "%d gene features remain.", n_adt, adata.n_vars,
        )
    adata.uns["n_adt_features"] = n_adt

    # ── QC (10x UMI): mito %, gene/cell filters -- matches the other 10x loaders ──
    # Run BEFORE the pairing filter so pairing is computed on QC-passing cells.
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
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

    # A participant-visit represented by a handful of cells yields a pseudobulk
    # that is effectively one cell's profile, yet the pairing guard above (which
    # only checks that >=2 days are present) would pass it. With n=6 participants
    # such a group can dominate both the paired delta and its variance.
    vc = adata.obs.groupby(["pt_id", "day"], observed=True).size()
    logger.info("Cells per participant-visit:\n%s", vc.to_string())
    thin = vc[vc < min_cells_per_participant_visit]
    if len(thin):
        logger.warning(
            "  %d participant-visit group(s) below %d cells: %s",
            len(thin), min_cells_per_participant_visit, thin.to_dict(),
        )
    adata.uns["cells_per_participant_visit"] = {
        f"{p}_d{d}": int(n) for (p, d), n in vc.items()
    }

    adata.layers["counts"] = adata.X.copy()
    # Normalize (GSE171964 matrix is raw integer UMI counts): CP10K + log1p, so
    # .X is log-normalized like the other 10x datasets (counts kept in the layer).
    # Gene filtering follows normalisation so per-cell library sizes are computed
    # over the full gene set, matching drop_artifact_genes' documented contract.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.filter_genes(adata, min_cells=10)
    adata.layers["log1p_norm"] = adata.X.copy()
    drop_artifact_genes(adata)  # QC: remove hemoglobin/ribosomal/histone genes (keep cell-cycle)
    adata.obs["annotation_source"] = "publication"
    adata.uns["annotation_source"] = (
        "GSE171964 deposited `clustnm` cluster names (source-provided, verbatim)"
    )
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


# Magnetically-enriched / FACS-sorted fractions deposited as separate samples.
# van Galen et al. sorted healthy donor BM5 into CD34+ and CD34+CD38- fractions;
# both are the SAME donor and the same aspirate. Treating them as two independent
# participants fabricates a donor and double-counts one person in every
# participant-level statistic, while also presenting enriched fractions as if
# they were unsorted whole marrow.
_AML_SORTED_FRACTIONS: dict[str, tuple[str, str]] = {
    "BM5-34p": ("BM5", "CD34+"),
    "BM5-34p38n": ("BM5", "CD34+CD38-"),
}


def _parse_aml_sample_info(sample_name: str) -> tuple[str, str, int, str]:
    """Parse sample name, participant ID, day, and sorted fraction.

    Returns ``(sample_name, patient, day, sorted_fraction)`` where
    *sorted_fraction* is ``"unsorted"`` unless the sample is a deposited
    enrichment fraction (see ``_AML_SORTED_FRACTIONS``).
    """
    if sample_name in _AML_SORTED_FRACTIONS:
        patient, fraction = _AML_SORTED_FRACTIONS[sample_name]
        return sample_name, patient, 0, fraction
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
    return sample_name, patient, day, "unsorted"


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

    # GSE116256 also deposits two immortalised leukaemia CELL LINES alongside the
    # primary marrow aspirates. Their sample names do not start with "AML", so the
    # sample_type heuristic below would file them as "Healthy" and they would
    # contaminate the healthy-donor reference group with cultured tumour lines.
    # They are not patient material and are excluded outright.
    _CELL_LINES = {"MUTZ3", "MUTZ-3", "OCI-AML3", "OCIAML3"}
    excluded = [s for s in matched if s.upper() in {c.upper() for c in _CELL_LINES}]
    if excluded:
        matched = [s for s in matched if s not in excluded]
        logger.info(f"Excluded {len(excluded)} cell-line sample(s): {excluded}")
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

        _, patient, day, sorted_fraction = _parse_aml_sample_info(sname)
        unique_cells = [f"{sname}_{c}" for c in common_cells]
        obs_df = anno_df.copy()
        obs_df.index = unique_cells
        obs_df["sample_id"] = sname
        obs_df["patient_id"] = patient
        obs_df["day"] = day
        obs_df["sorted_fraction"] = sorted_fraction

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
    n_after_genes = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
    n_after_mt = adata.n_obs
    sc.pp.filter_genes(adata, min_cells=10)
    logger.info(f"QC: {n_before:,} → {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    # Persist the QC funnel so the loss is auditable from the cached h5ad alone.
    adata.uns["qc_summary"] = {
        "n_before": int(n_before),
        "n_after_gene_filters": int(n_after_genes),
        "n_after_mt_filter": int(n_after_mt),
        "n_final": int(adata.n_obs),
        "filters": "min_genes=200, max_genes=6000, pct_counts_mt<20, genes min_cells=10",
    }

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["log1p_norm"] = adata.X.copy()
    # drop_hb=False: this is BONE MARROW. Erythroid precursors (earlyEry/lateEry)
    # are published populations here, and hemoglobin/ALAS2/AHSP/SLC4A1 is their
    # defining identity programme -- not ambient RBC-lysis contamination as it
    # would be in peripheral blood. Dropping it would delete the biology.
    drop_artifact_genes(adata, drop_hb=False)

    # ── Standardised sctrial obs columns ──────────────────────────────
    obs = adata.obs
    obs["participant_id"] = obs["patient_id"].astype(str)
    # Keep the raw day AND an ordered timepoint. Collapsing every post-treatment
    # marrow to a single "Post" pools day-14 residual-disease aspirates with
    # long-remission ones for the same patient, which participant x visit
    # aggregation then averages together.
    obs["visit"] = obs["day"].apply(lambda d: "Pre" if d == 0 else "Post")
    obs["timepoint"] = obs["day"].apply(
        lambda d: "Pre" if d == 0 else ("EarlyPost" if d <= 45 else "LatePost")
    )
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
        # Nullable: an unclassified cell is unknown, not "not malignant".
        obs["malignant_status"] = obs["PredictionRefined"].apply(
            lambda x: pd.NA
            if pd.isna(x)
            else ("malignant" if "malignant" in str(x).lower() else "normal")
        )
    else:
        obs["malignant_status"] = pd.NA
    obs["is_malignant"] = obs["malignant_status"].eq("malignant")
    # `disease_group` is a cohort label derived from the sample name, NOT a
    # clinical outcome. `response` is kept as an alias for backwards
    # compatibility but must not be read as an endpoint: GSE116256 deposits no
    # per-patient response variable.
    obs["disease_group"] = (
        obs["sample_type"].map({"AML": "Treatment", "Healthy": "Control"}).fillna("Unknown")
    )
    obs["response"] = obs["disease_group"]
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
    adata.obs["annotation_source"] = "publication"
    adata.uns["annotation_source"] = (
        "van Galen et al., Cell 2019 — deposited .anno CellType / PredictionRefined "
        "labels (used verbatim; no de-novo clustering)"
    )
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
        "version": "v3",
        "max_cells_per_sample": max_cells_per_sample,
        "seed": seed,
        "exclude_cell_lines": True,
        "keep_hemoglobin": True,
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
    # `visit` is the binary Pre/Post contrast the DiD estimand needs; the three
    # distinct post-infusion timepoints remain separable via `timepoint` and
    # `days_post_treatment`, which are NOT collapsed.
    obs["visit"] = obs["timepoint"].apply(lambda t: "Pre" if t == "Leukapheresis" else "Post")
    obs["is_paired"] = False  # filled below
    adata.obs = obs

    # ── Author per-cell metadata (annotations + clinical response) ─────
    cart_meta = _load_cart_published_metadata(raw_dir)
    if cart_meta is not None:
        cid = adata.obs["cell_id"].astype(str)
        n_match = int(cid.isin(cart_meta.index).sum())
        logger.info(
            "  GSE290722_metadata: matched %d/%d cells (%.2f%%)",
            n_match, adata.n_obs, 100.0 * n_match / max(adata.n_obs, 1),
        )
        for src, dst in (
            # Major_Alias is the primary lineage: unlike Compartment it keeps
            # "T regs" as a distinct population, which the analysis needs.
            ("Major_Alias", "cell_type_published"),
            ("Compartment", "cell_type_compartment_published"),
            ("Alias", "cluster_published"),
            ("Response", "response_published"),
            ("TimePoint_Final", "timepoint_published"),
            ("Axicel", "car_transgene_counts"),
        ):
            if src in cart_meta.columns:
                adata.obs[dst] = cid.map(cart_meta[src]).astype("object")
        if "car_transgene_counts" in adata.obs:
            adata.obs["is_car_positive"] = (
                pd.to_numeric(adata.obs["car_transgene_counts"], errors="coerce").fillna(0) > 0
            )
        # The authors' primary outcome. Previously this column was hard-coded to
        # the constant "CAR-T", which silently discarded the durable-responder /
        # responder / non-responder contrast the publication is built on.
        if "response_published" in adata.obs:
            adata.obs["response"] = (
                adata.obs["response_published"].fillna("Unknown").astype(str)
            )
        else:
            adata.obs["response"] = "Unknown"
    else:
        adata.obs["response"] = "Unknown"
        logger.warning(
            "GSE290722_metadata.csv.gz not found: per-cell response and author "
            "annotations unavailable. Re-run with allow_download=True."
        )

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

    # ── Cell type annotation ──────────────────────────────────────────
    # De-novo marker scoring is retained as a reproducibility check, but the
    # authors' own annotation is primary: on this dataset the generic immune
    # panel mislabels the majority of T cells (cytotoxic CD8 clusters score
    # onto the NK set via shared GZMB/PRF1/NKG7, and Tregs are not recovered).
    logger.info("Annotating CAR-T cell types (de-novo)...")
    adata.obs["cell_type_denovo"] = _annotate_immune_celltypes(adata)

    if "cell_type_published" in adata.obs and adata.obs["cell_type_published"].notna().any():
        adata.obs["cell_type"] = (
            adata.obs["cell_type_published"].fillna("Unassigned").astype(str)
        )
        adata.obs["annotation_source"] = "publication"
        adata.uns["annotation_source"] = (
            "Cheloni et al., Nat Commun 2025 — GSE290722_metadata.csv.gz "
            "Major_Alias (primary, 15 levels incl. T regs), Compartment (12), "
            "Alias (73 populations)"
        )
        # The authors flagged and removed doublet-enriched clusters; keep the
        # flag so downstream analyses can exclude them explicitly.
        adata.obs["is_doublet"] = (
            adata.obs["cell_type_published"].astype(str).str.lower() == "doublets"
        )
    else:
        adata.obs["cell_type"] = adata.obs["cell_type_denovo"].astype(str)
        adata.obs["annotation_source"] = "de-novo"
        adata.uns["annotation_source"] = "de-novo marker scoring (author metadata unavailable)"
        adata.obs["is_doublet"] = False

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
        "version": "v3",
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

    # Series-level per-cell metadata (author annotations, clinical response,
    # CAR-transgene counts). It sits beside the RAW tar rather than inside it,
    # so it must be fetched separately. Non-fatal if unavailable.
    meta_dest = found_raw / "GSE290722_metadata.csv.gz"
    if not meta_dest.exists() and allow_download:
        try:
            _download_file(
                "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE290nnn/GSE290722/suppl/"
                "GSE290722_metadata.csv.gz",
                meta_dest,
                "GSE290722 per-cell metadata",
            )
        except Exception as exc:  # noqa: BLE001 - optional enrichment
            logger.warning(
                "Could not download GSE290722_metadata.csv.gz (%s); author "
                "annotations and per-cell response will be unavailable.", exc,
            )

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

    # Cell type annotation -- de-novo first (retained as a reproducibility
    # check), then superseded by the authors' own per-cell labels from mmc3.
    logger.info("Annotating cell types (Wilcoxon + weighted scoring, de-novo)...")
    adata.obs["cell_type_denovo"] = _annotate_immune_celltypes(adata)

    logger.info("Joining published per-cell annotations (mmc3.xlsx, Table S2)...")
    pub = _load_tnbc_published_labels(raw_dir)
    bc = adata.obs["barcode_full"].astype(str)
    rename = {"cell_type_published": "cell_type_major_published"}
    for col in ("cell_type_published", "cluster_published"):
        if col in pub.columns:
            adata.obs[rename.get(col, col)] = bc.map(pub[col]).astype("object")
    n_pub = int(adata.obs["cell_type_major_published"].notna().sum())
    logger.info(
        "  published labels matched %d/%d cells (%.2f%%)",
        n_pub, adata.n_obs, 100.0 * n_pub / max(adata.n_obs, 1),
    )
    if n_pub:
        # Lineage is derived from the authors' 97 fine clusters rather than
        # their 4 major categories, which cannot separate CD4/CD8/Treg.
        adata.obs["cell_type_published"] = (
            adata.obs["cluster_published"].map(_tnbc_cluster_lineage).astype("object")
        )
        adata.obs["cell_type"] = (
            adata.obs["cell_type_published"].fillna("Unassigned").astype(str)
        )
        adata.obs["annotation_source"] = "publication"
        adata.uns["annotation_source"] = (
            "Zhang et al., Cancer Cell 2021 — Table S2 (mmc3.xlsx) 'Major "
            "celltype' (primary) and 'Cluster' (97 author clusters)"
        )
    else:
        adata.obs["cell_type"] = adata.obs["cell_type_denovo"].astype(str)
        adata.obs["annotation_source"] = "de-novo"
        adata.uns["annotation_source"] = "de-novo marker scoring (mmc3 labels unavailable)"

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
        "version": "v5",
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
