from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

if TYPE_CHECKING:
    from ctxcore.genesig import GeneSignature

__all__ = ["score_gene_sets", "score_gene_sets_aucell", "ScoreMethod"]

logger = logging.getLogger(__name__)

ScoreMethod = Literal["zmean", "mean"]

try:
    from pyscenic.aucell import aucell, create_rankings
except ImportError:  # pragma: no cover
    create_rankings = None
    aucell = None


def score_gene_sets(
        adata: AnnData,
        gene_sets: dict[str, list[str]],
        *,
        layer: str | None = None,
        method: ScoreMethod = "zmean",
        prefix: str = "",
        min_genes: int = 3,
        overwrite: bool = True,
) -> AnnData:
    """Score gene sets and store results in `adata.obs`.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    gene_sets
        Dictionary mapping set names to lists of gene names.
        Each value must be a ``list`` (not a bare string).  Duplicate gene
        names within a set are automatically removed.
    layer
        Expression matrix source. If None, uses `adata.X`.
        For log1p-CPM workflows, use layer="log1p_cpm".
    method
        Scoring method:

        - "mean": mean expression across genes.
        - "zmean": z-score each gene across cells (within the current AnnData),
          then average z-scores across genes. This is the recommended method
          as it accounts for different expression scales across genes.
    prefix
        Prefix to add to column names (e.g., ``ms_`` for module scores).
    min_genes
        Minimum number of genes from the set that must be present in the data.
        If fewer genes overlap, the score is set to NaN and a warning is
        logged.  Default is 3.
    overwrite
        If False, skip gene sets that already have a column in adata.obs.

    Returns
    -------
    AnnData
        The input AnnData with new columns added to obs.

    Notes
    -----
    **Zero-variance gene handling (zmean method):**
    Genes with zero or near-zero variance (std < 1e-12) are excluded from
    the z-mean calculation. If ALL genes in a set have zero variance, the
    score is NaN. This prevents division by zero and ensures meaningful scores.

    The zmean method computes: mean(z_i) where z_i = (x_i - mean(x_i)) / std(x_i)
    for each gene i across all cells.

    **Non-finite expression values:**
    NaN and inf values in the expression matrix are excluded from score
    computation (treated as missing).  A warning is logged when non-finite
    values are detected.  If *all* values for a cell are non-finite the
    resulting score will be NaN.
    """
    if method not in ("zmean", "mean"):
        raise ValueError(f"Unknown method '{method}'. Use 'zmean' or 'mean'.")
    if not isinstance(gene_sets, dict) or len(gene_sets) == 0:
        raise ValueError("gene_sets must be a non-empty dict of name -> gene list.")
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string.")
    if min_genes < 1:
        raise ValueError("min_genes must be >= 1.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer '{layer}' not found in adata.layers.")

    # Validate per-set gene list types up front
    for name, gset in gene_sets.items():
        if not isinstance(gset, (list, tuple, set, frozenset)):
            raise TypeError(
                f"Gene set '{name}' must be a list of gene names, "
                f"got {type(gset).__name__}."
            )

    X = adata.layers[layer] if layer is not None else adata.X
    var_names = adata.var_names
    idx = {g: i for i, g in enumerate(var_names)}

    is_sparse = sp.issparse(X)
    if is_sparse:
        if not isinstance(X, sp.csr_matrix):
            X = X.tocsr()

    for name, gset in gene_sets.items():
        # Deduplicate while preserving order
        seen: set[str] = set()
        use: list[str] = []
        for g in gset:
            if g in idx and g not in seen:
                use.append(g)
                seen.add(g)

        col = f"{prefix}{name}"

        if (not overwrite) and (col in adata.obs.columns):
            continue

        # Count unique requested genes (after dedup) for accurate logging
        n_unique_requested = len({g for g in gset})
        n_found = len(use)

        if n_found < min_genes:
            logger.warning(
                "Gene set '%s': only %d/%d unique genes found in data "
                "(min_genes=%d); setting score to NaN.",
                name, n_found, n_unique_requested, min_genes,
            )
            adata.obs[col] = np.nan
            continue

        if n_found < n_unique_requested:
            logger.info(
                "Gene set '%s': %d/%d unique genes found in data.",
                name, n_found, n_unique_requested,
            )

        gidx = np.array([idx[g] for g in use], dtype=int)

        if method == "mean" and is_sparse:
            sub = X[:, gidx]
            dense_sub = np.asarray(sub.todense())
            finite_mask = np.isfinite(dense_sub)
            n_nonfinite = int(np.sum(~finite_mask))
            if n_nonfinite > 0:
                logger.warning(
                    "Gene set '%s': %d non-finite value(s) in expression "
                    "matrix; these are excluded from the mean.",
                    name, n_nonfinite,
                )
            # nanmean ignores NaN; replace inf with NaN first
            dense_sub = np.where(finite_mask, dense_sub, np.nan)
            with np.errstate(all="ignore"):
                score = np.nanmean(dense_sub, axis=1)
            adata.obs[col] = score
            continue

        # For zmean, or dense mean, compute dense submatrix for gene-set only
        sub = X[:, gidx].toarray() if is_sparse else np.asarray(X[:, gidx], dtype=np.float64)

        # Detect and mask non-finite values
        finite_mask = np.isfinite(sub)
        n_nonfinite_expr = int(np.sum(~finite_mask))
        if n_nonfinite_expr > 0:
            logger.warning(
                "Gene set '%s': %d non-finite value(s) in expression "
                "matrix; these are excluded from scoring.",
                name, n_nonfinite_expr,
            )
            sub = np.where(finite_mask, sub, np.nan)

        if method == "mean":
            with np.errstate(all="ignore"):
                score = np.nanmean(sub, axis=1)
        else:  # zmean (already validated above)
            # Compute stats ignoring NaN
            with np.errstate(all="ignore"):
                mu = np.nanmean(sub, axis=0, keepdims=True)
                sd = np.nanstd(sub, axis=0, ddof=1, keepdims=True)

            # Mask zero-variance genes to prevent division by zero
            valid_genes = (sd > 1e-12).ravel()
            n_valid = valid_genes.sum()
            if n_valid == 0:
                # All genes have zero variance - return NaN
                score = np.full(sub.shape[0], np.nan)
            else:
                # Z-score only non-zero-variance genes; nanmean
                # handles any per-cell NaN from the expression data.
                z = (sub[:, valid_genes] - mu[:, valid_genes]) / sd[:, valid_genes]
                with np.errstate(all="ignore"):
                    score = np.nanmean(z, axis=1)

        adata.obs[col] = score

    return adata


def score_gene_sets_aucell(
    adata: AnnData,
    gene_sets: dict[str, list[str]] | dict[str, GeneSignature],
    *,
    layer: str | None = None,
    prefix: str = "aucell_",
    overwrite: bool = False,
) -> AnnData:
    """Score gene sets using AUCell (pySCENIC).

    Requires pyscenic to be installed.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    gene_sets
        Dictionary mapping set names to lists of genes (or GeneSignature objects).
    layer
        Expression layer to use. If None, uses `adata.X`.
    prefix
        Prefix to add to output columns (default: ``aucell_``).
    overwrite
        If False, skip sets that already exist in `adata.obs`.

    Returns
    -------
    AnnData
        The input AnnData with AUCell scores added to `adata.obs`.
    """
    if create_rankings is None or aucell is None:
        raise ImportError("pyscenic is required for AUCell scoring. Install with 'pip install pyscenic'.")

    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df = pd.DataFrame(X, columns=adata.var_names, index=adata.obs_names)

    rankings = create_rankings(df)
    from ctxcore.genesig import GeneSignature

    for name, genes in gene_sets.items():
        col = f"{prefix}{name}"
        if (not overwrite) and col in adata.obs.columns:
            continue
        if isinstance(genes, GeneSignature):
            gs = genes
        else:
            genes_present = [g for g in genes if g in adata.var_names]
            if not genes_present:
                adata.obs[col] = np.nan
                continue
            gs = GeneSignature(name, genes_present)
        scores = aucell(rankings, [gs])[0]
        adata.obs[col] = scores
    return adata
