from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

__all__ = ["add_log1p_cpm_layer"]

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
        raise ValueError(
            f"scale must be a finite positive number, got {scale!r}."
        )

    ad = adata if inplace else adata.copy()

    if (out_layer in ad.layers) and (not overwrite):
        logger.info(
            "Layer '%s' already exists; returning unchanged "
            "(pass overwrite=True to recompute).",
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
