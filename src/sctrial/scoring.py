from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

__all__ = ["score_gene_sets", "ScoreMethod"]

ScoreMethod = Literal["zmean", "mean"]


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
        If fewer genes overlap, the score is set to NaN.
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
    """
    if method not in ("zmean", "mean"):
        raise ValueError(f"Unknown method '{method}'. Use 'zmean' or 'mean'.")
    X = adata.layers[layer] if layer is not None else adata.X
    var_names = adata.var_names
    idx = {g: i for i, g in enumerate(var_names)}

    is_sparse = sp.issparse(X)
    if is_sparse:
        if not isinstance(X, sp.csr_matrix):
            X = X.tocsr()

    for name, gset in gene_sets.items():
        use = [g for g in gset if g in idx]
        col = f"{prefix}{name}"

        if (not overwrite) and (col in adata.obs.columns):
            continue

        if len(use) < min_genes:
            adata.obs[col] = np.nan
            continue

        gidx = np.array([idx[g] for g in use], dtype=int)

        if method == "mean" and is_sparse:
            sub = X[:, gidx]
            score = np.asarray(sub.mean(axis=1)).ravel()
            adata.obs[col] = score
            continue

        # For zmean, or dense mean, compute dense submatrix for gene-set only
        sub = X[:, gidx].toarray() if is_sparse else np.asarray(X[:, gidx], dtype=np.float64)

        if method == "mean":
            score = sub.mean(axis=1)
        elif method == "zmean":
            mu = sub.mean(axis=0, keepdims=True)
            sd = sub.std(axis=0, ddof=1, keepdims=True)

            # Mask zero-variance genes to prevent NaNs
            mask = (sd > 1e-12).ravel()
            n_valid = mask.sum()
            if n_valid == 0:
                # All genes have zero variance - return NaN
                score = np.full(sub.shape[0], np.nan)
            else:
                # Only average over genes with non-zero variance
                z = (sub[:, mask] - mu[:, mask]) / sd[:, mask]
                score = z.mean(axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        adata.obs[col] = score

    return adata
