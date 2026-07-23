from __future__ import annotations

import logging
import re

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

__all__ = [
    "add_log1p_cpm_layer",
    "flag_artifact_genes",
    "is_artifact_gene",
    "drop_artifact_genes",
    "exclude_artifacts_from_hvg",
    "add_qc_class_metrics",
]

logger = logging.getLogger(__name__)


def add_log1p_cpm_layer(
    adata: AnnData,
    *,
    counts_layer: str | None = "counts",
    out_layer: str = "log1p_cpm",
    layer_out: str | None = None,  # alias for out_layer
    scale: float = 1e6,
    overwrite: bool = False,
    inplace: bool = True,
) -> AnnData:
    """Add log1p(CPM) normalization as a layer.

    Parameters
    ----------
    adata
        AnnData object with raw counts.
    counts_layer
        Layer name containing raw counts. If None, uses adata.X.
    out_layer
        Output layer name.
    layer_out
        Backwards-compatible alias for out_layer.
    scale
        CPM scale factor (default 1e6). Must be finite and positive.
    overwrite
        Overwrite if out_layer already exists.
    inplace
        Modify the input AnnData if True, else return a copy.

    Returns
    -------
    AnnData
        The AnnData object with the new log1p(CPM) layer added.

    Raises
    ------
    ValueError
        If ``scale`` is not a finite positive number, or if the counts
        matrix contains negative values.
    KeyError
        If ``counts_layer`` is not found in ``adata.layers``.
    """
    if layer_out is not None:
        out_layer = layer_out

    # --- Validate scale ---
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a finite positive number, got {scale!r}.")

    ad = adata if inplace else adata.copy()

    if (out_layer in ad.layers) and (not overwrite):
        logger.info(
            "Layer '%s' already exists; returning unchanged (pass overwrite=True to recompute).",
            out_layer,
        )
        return ad

    # Fetch counts matrix
    if counts_layer is None:
        X = ad.X
    else:
        if counts_layer not in ad.layers:
            raise KeyError(
                f"counts_layer='{counts_layer}' not found in adata.layers. "
                f"Available layers: {list(ad.layers.keys())}. "
                f"Either add adata.layers['{counts_layer}']=<counts> or "
                f"pass counts_layer=None to use adata.X."
            )
        X = ad.layers[counts_layer]

    # --- Validate counts are non-negative and finite ---
    if sp.issparse(X):
        vals = X.data
    else:
        vals = np.asarray(X).ravel()

    if vals.size > 0:
        if not np.all(np.isfinite(vals)):
            raise ValueError(
                "Counts matrix contains non-finite values (NaN or inf). "
                "log1p(CPM) requires finite non-negative counts."
            )
        if np.min(vals) < 0:
            raise ValueError(
                f"Counts matrix contains negative values (min={np.min(vals):.4g}). "
                "log1p(CPM) requires non-negative counts. "
                "Pass raw counts, not log-transformed or z-scored data."
            )

    # Compute log1p(CPM)
    if sp.issparse(X):
        if not isinstance(X, sp.csr_matrix):
            X = X.tocsr()
        libsize = np.asarray(X.sum(axis=1)).reshape(-1)
        n_zero = int(np.sum(libsize == 0))
        if n_zero > 0:
            logger.warning(
                "%d cell(s) have zero total counts and will have all-zero "
                "CPM values. Consider filtering these cells before "
                "normalization.",
                n_zero,
            )
        # Safe division: 0/0 → 0 via replacing zero libsizes with inf
        safe_libsize = libsize.astype(float, copy=True)
        safe_libsize[safe_libsize == 0] = np.inf
        X_cpm = X.multiply(scale / safe_libsize.reshape(-1, 1))
        X_log = X_cpm.tocsr()
        X_log.data = np.log1p(X_log.data)
    else:
        X = np.asarray(X)
        libsize = X.sum(axis=1, keepdims=True)
        n_zero = int(np.sum(libsize == 0))
        if n_zero > 0:
            logger.warning(
                "%d cell(s) have zero total counts and will have all-zero "
                "CPM values. Consider filtering these cells before "
                "normalization.",
                n_zero,
            )
        # Safe division: 0/0 → 0 via replacing zero libsizes with inf
        safe_libsize = libsize.astype(float, copy=True)
        safe_libsize[safe_libsize == 0] = np.inf
        X_cpm = X / safe_libsize * scale
        X_log = np.log1p(X_cpm)

    ad.layers[out_layer] = X_log
    # Store provenance for reproducibility
    ad.uns.setdefault("sctrial", {})["log1p_cpm_scale"] = float(scale)
    return ad


# ---------------------------------------------------------------------------
# Technical-artifact gene QC (ambient / housekeeping correction)
# ---------------------------------------------------------------------------
# Gene classes that dominate UNBIASED gene-level differential expression and GSEA
# leading-edges as technical/housekeeping signal rather than biology, so they are
# flagged here (upstream, at preprocessing) and excluded from the HVG/analysis
# feature set. Ambient hemoglobin reflects red-blood-cell lysis; ribosomal and
# replication-histone genes are highly-expressed housekeeping that dominate rankings.
# Cell-cycle / proliferation genes are deliberately NOT flagged -- they are real
# biology and are used by the proliferation gene signatures. Patterns are anchored
# to END so signaling genes that merely share a prefix (e.g. RPS6KA*, the ribosomal
# protein S6 kinases) are NOT flagged. Genes are kept in the object; only flagged.
_HB_RE = re.compile(
    r"^(HBA[12]|HBB|HBD|HBE1|HBG[12]|HBM|HBQ1|HBZ|ALAS2|SLC4A1|EPB42|CA1|AHSP)$",
    re.IGNORECASE,
)
_RIBO_RE = re.compile(r"^(RP[SL]\d+[A-Z]?\d?|RPLP\d|RPSA|FAU|UBA52)$", re.IGNORECASE)
_HIST_RE = re.compile(r"^HIST\d[0-9A-Z]*$", re.IGNORECASE)


def is_artifact_gene(name: str) -> bool:
    """True if *name* is a technical-artifact gene (hemoglobin/erythroid, ribosomal,
    or replication-histone) to be excluded from unbiased HVG/DE/GSEA. Cell-cycle
    genes are NOT artifacts. Anchored so prefix-sharing signaling genes are kept."""
    g = str(name)
    return bool(_HB_RE.match(g) or _RIBO_RE.match(g) or _HIST_RE.match(g))


def flag_artifact_genes(adata: AnnData, *, inplace: bool = True) -> AnnData:
    """Flag technical-artifact gene classes on ``adata.var`` for QC exclusion.

    Sets boolean columns ``is_hb`` (hemoglobin/erythroid), ``is_ribo`` (ribosomal
    proteins), ``is_histone`` (replication histones), and their union ``is_artifact``.
    Depends only on ``adata.var_names`` (gene symbols), so it never touches counts or
    library sizes and behaves identically for TPM, raw-count, and log-normalized data.
    Genes are kept in the object; only flagged, so signature scoring (which uses
    explicit curated gene lists, none of which are artifacts) is unaffected. Cell-cycle
    genes are deliberately not flagged (used by the proliferation signatures).
    """
    ad = adata if inplace else adata.copy()
    names = [str(g) for g in ad.var_names]
    ad.var["is_hb"] = np.array([bool(_HB_RE.match(g)) for g in names])
    ad.var["is_ribo"] = np.array([bool(_RIBO_RE.match(g)) for g in names])
    ad.var["is_histone"] = np.array([bool(_HIST_RE.match(g)) for g in names])
    ad.var["is_artifact"] = ad.var["is_hb"] | ad.var["is_ribo"] | ad.var["is_histone"]
    logger.info(
        "Flagged %d artifact genes for QC exclusion (hb=%d, ribo=%d, histone=%d).",
        int(ad.var["is_artifact"].sum()), int(ad.var["is_hb"].sum()),
        int(ad.var["is_ribo"].sum()), int(ad.var["is_histone"].sum()),
    )
    return ad


def exclude_artifacts_from_hvg(adata: AnnData) -> AnnData:
    """Remove flagged artifact genes from the highly-variable set in place.

    Sets ``var['highly_variable'] &= ~var['is_artifact']`` so every HVG-driven step
    (PCA/clustering, gene-level DE, GSEA rankings) skips technical-artifact genes.
    Runs :func:`flag_artifact_genes` first if the flag is missing. No-op if
    ``highly_variable`` has not been computed yet.
    """
    if "is_artifact" not in adata.var:
        flag_artifact_genes(adata)
    if "highly_variable" in adata.var:
        adata.var["highly_variable"] = (
            adata.var["highly_variable"].to_numpy() & ~adata.var["is_artifact"].to_numpy()
        )
    return adata


def drop_artifact_genes(adata: AnnData) -> AnnData:
    """Remove technical-artifact gene classes from *adata* in place (upstream QC).

    Removes hemoglobin/erythroid, ribosomal, and replication-histone genes so that
    EVERY downstream analysis inherits a clean gene set with no per-analysis handling:
    HVG/PCA/clustering, gene-level differential expression, and GSEA (which ranks the
    full ``var_names`` list, so it would otherwise be dominated by these housekeeping/
    ambient genes in its leading edges). Cell-cycle genes are NOT removed -- they are
    real biology and are used by the proliferation gene signatures.

    Must be run AFTER normalization (per-cell library sizes must be computed over all
    genes, including the ambient/housekeeping ones) and after any QC-metric computation.
    The removed gene symbols are recorded in ``adata.uns['artifact_genes_removed']``.
    """
    if "is_artifact" not in adata.var:
        flag_artifact_genes(adata)
    keep = ~adata.var["is_artifact"].to_numpy()
    removed = adata.var_names[~keep].tolist()
    adata._inplace_subset_var(keep)
    adata.uns["artifact_genes_removed"] = removed
    logger.info("Removed %d technical-artifact genes (kept %d).", len(removed), int(keep.sum()))
    return adata


def add_qc_class_metrics(adata: AnnData, *, counts_layer: str | None = "counts") -> AnnData:
    """Add ``pct_counts_hb`` / ``pct_counts_ribo`` (and ``pct_counts_mt``) QC metrics.

    Uses ``counts_layer`` if present in ``adata.layers`` else ``adata.X``. Optional and
    informational -- the analysis exclusion is driven by :func:`flag_artifact_genes`.
    Requires scanpy.
    """
    import scanpy as sc

    if "is_hb" not in adata.var:
        flag_artifact_genes(adata)
    adata.var["hb"] = adata.var["is_hb"].to_numpy()
    adata.var["ribo"] = adata.var["is_ribo"].to_numpy()
    if "mt" not in adata.var:
        adata.var["mt"] = np.array([str(g).upper().startswith("MT-") for g in adata.var_names])
    layer = counts_layer if (counts_layer and counts_layer in adata.layers) else None
    qc_vars = [v for v in ("mt", "ribo", "hb") if bool(adata.var[v].any())]
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, layer=layer, percent_top=None, log1p=False, inplace=True
    )
    return adata
