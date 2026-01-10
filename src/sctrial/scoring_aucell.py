from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

if TYPE_CHECKING:
    from pyscenic.aucell import aucell

try:
    from pyscenic.aucell import create_rankings, aucell
except ImportError:  # pragma: no cover
    create_rankings = None
    aucell = None

__all__ = ["score_gene_sets_aucell"]


def score_gene_sets_aucell(
    adata: AnnData,
    gene_sets: dict[str, Sequence[str]] | dict[str, "GeneSignature"],
    *,
    layer: str | None = None,
    prefix: str = "aucell_",
    overwrite: bool = False,
) -> AnnData:
    """Score gene sets using AUCell (pySCENIC).

    Requires pyscenic to be installed.
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
