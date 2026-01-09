from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

__all__ = ["extract_gene_vector"]

def extract_gene_vector(adata: AnnData, gene: str, layer: str | None = None) -> np.ndarray:
    """Return a 1D numpy array of gene expression (length = n_obs)."""
    if gene not in adata.var_names:
        raise KeyError(gene)
    j = int(adata.var_names.get_loc(gene))
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        # Ensure it's subscriptable (COO is not)
        if isinstance(X, sp.coo_matrix):
            X = X.tocsr()
        return np.asarray(X[:, j].toarray()).ravel()
    return np.asarray(X[:, j]).ravel()
