from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData


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
    counts_layer
        Layer name containing raw counts. If None, uses adata.X.
    out_layer
        Output layer name.
    layer_out
        Backwards-compatible alias for out_layer.
    scale
        CPM scale factor (default 1e6).
    overwrite
        Overwrite if out_layer already exists.
    inplace
        Modify the input AnnData if True, else return a copy.
    """
    if layer_out is not None:
        out_layer = layer_out

    ad = adata if inplace else adata.copy()

    if (out_layer in ad.layers) and (not overwrite):
        return ad

    # Fetch counts matrix
    if counts_layer is None:
        X = ad.X
    else:
        if counts_layer not in ad.layers:
            raise KeyError(
                f"counts_layer='{counts_layer}' not found in adata.layers. "
                f"Available layers: {list(ad.layers.keys())}. "
                f"Either add adata.layers['{counts_layer}']=<counts> or pass counts_layer=None to use adata.X."
            )
        X = ad.layers[counts_layer]

    # Compute log1p(CPM)
    if sp.issparse(X):
        if not isinstance(X, sp.csr_matrix):
            X = X.tocsr()
        libsize = np.asarray(X.sum(axis=1)).reshape(-1, 1)
        X_cpm = X.multiply(scale / (libsize + 1e-12))
        X_log = X_cpm.tocsr()
        X_log.data = np.log1p(X_log.data)
    else:
        X = np.asarray(X)
        libsize = X.sum(axis=1, keepdims=True)
        X_cpm = X / (libsize + 1e-12) * scale
        X_log = np.log1p(X_cpm)

    ad.layers[out_layer] = X_log
    # optional provenance:
    # ad.uns.setdefault("sctrial", {})["log1p_cpm_scale"] = float(scale)
    return ad
