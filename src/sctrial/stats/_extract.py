from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

__all__ = ["extract_gene_matrix", "extract_gene_vector"]


def extract_gene_matrix(
    adata: AnnData,
    genes: Sequence[str],
    layer: str | None = None,
) -> np.ndarray:
    """Return a 2D numpy array of gene expression (n_obs × n_genes).

    Parameters
    ----------
    adata
        AnnData object.
    genes
        Sequence of gene names to extract.
    layer
        Layer name in `adata.layers` to use for expression data.
        If None, uses `adata.X`.

    Returns
    -------
    np.ndarray
        A 2D numpy array of gene expression (n_obs × n_genes).

    Raises
    ------
    KeyError
        If genes are not found in `adata.var_names`.
    """
    if not genes:
        return np.empty((adata.n_obs, 0), dtype=float)
    missing = [g for g in genes if g not in adata.var_names]
    if missing:
        raise KeyError(missing[:5])
    idx = adata.var_names.get_indexer(genes)
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        if isinstance(X, sp.coo_matrix):
            X = X.tocsr()
        return np.asarray(X[:, idx].toarray())
    return np.asarray(X[:, idx])


def extract_gene_vector(adata: AnnData, gene: str, layer: str | None = None) -> np.ndarray:
    """Return a 1D numpy array of gene expression (length = n_obs).

    Parameters
    ----------
    adata
        AnnData object.
    gene
        Gene name to extract.
    layer
        Layer name in `adata.layers` to use for expression data.
        If None, uses `adata.X`.

    Returns
    -------
    np.ndarray
        A 1D numpy array of gene expression (length = n_obs).
    """
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
