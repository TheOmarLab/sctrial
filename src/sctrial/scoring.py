from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

ScoreMethod = Literal["zmean", "mean"]


def score_gene_sets(
        adata: AnnData,
        gene_sets: Dict[str, List[str]],
        *,
        layer: Optional[str] = None,
        method: ScoreMethod = "zmean",
        prefix: str = "",
        min_genes: int = 3,
        overwrite: bool = True,
) -> AnnData:
    """Score gene sets and store results in `adata.obs`.

    Parameters
    ----------
    layer
        Expression matrix source. If None, uses `adata.X`.
        For log1p-CPM workflows, use layer="log1p_cpm".
    method
        - "mean": mean expression across genes.
        - "zmean": z-score each gene across cells (within the current AnnData),
          then average z across genes.
    min_genes
        Minimum overlap genes required; otherwise score is NaN.
    """
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