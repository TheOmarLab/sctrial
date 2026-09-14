"""What each benchmark method receives, and what it is scored against.

Gate F. Two published conclusions have already been produced by getting this
wrong, both times because each half was correct in isolation and only the seam
between them was wrong:

* dreamlet's "substantial effect-size bias" was a log2-versus-natural-log unit
  error at the harvest seam;
* roughly two thirds of dreamlet's "empirical-Bayes inflation" was a
  normalisation-scope artifact -- every method normalised against the 50-2000
  gene *panel* rather than the transcriptome, so a coordinated signal moved the
  denominator it was being measured against.

This module makes both seams explicit and testable. Nothing here is inferred at
call time: each method class has a written contract for its input representation
and a named estimand, and ``tests/test_benchmark_contracts.py`` asserts them.

The contracts
-------------
============================  =========================================  ==============
Method                        Input                                      Estimand
============================  =========================================  ==============
``sctrial_did``               participant x visit ``log(1 + CPM)``,      ``log1p_cpm``
                              CPM denominator = FULL transcriptome,
                              panel selected AFTER normalisation
``wilcoxon_paired``           identical outcome to ``sctrial_did``       ``log1p_cpm``
``dreamlet``                  raw summed pseudobulk counts + full-       ``count_link``
                              transcriptome ``lib.size``; dreamlet's
                              own TMM/voom normalisation on top
``limma_voom``                as dreamlet                                ``count_link``
``edger_qlf``                 as dreamlet                                ``count_link``
``nebula``                    cell-level raw counts + full-              ``count_link``
                              transcriptome library size as the
                              offset, on the LINEAR scale
============================  =========================================  ==============

Why two estimands
-----------------
These methods do not estimate the same functional. ``sctrial`` and the Wilcoxon
change score target the difference-in-differences of ``log(1 + CPM)``; the
count-based models target a log-link coefficient. Because
``d/dx log(1+x) = 1/(1+x)``, the two coincide only when CPM >> 1 and diverge for
low-expression genes at realistic depth. Scoring every method against the
injected ``beta`` therefore penalises whichever estimand differs most from it --
which is a property of the estimand, not of the method.

Each method is scored against its OWN oracle. Cross-method comparison is retained
for Type I error, calibration, FDR, power, direction accuracy, convergence and
runtime, all of which are estimand-invariant, and is NOT retained for a bias/RMSE
ranking, which is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "METHOD_ESTIMAND",
    "METHOD_INPUT",
    "prepare_inputs",
    "prepare_inputs_from_adata",
    "participant_log1p_cpm",
]

METHOD_ESTIMAND: dict[str, str] = {
    "sctrial_did": "log1p_cpm",
    "sctrial_mixed": "log1p_cpm",
    "wilcoxon_paired": "log1p_cpm",
    "dreamlet": "count_link",
    "limma_voom": "count_link",
    "edger_qlf": "count_link",
    "nebula": "count_link",
}

METHOD_INPUT: dict[str, str] = {
    "sctrial_did": "participant_log1p_cpm",
    "sctrial_mixed": "participant_log1p_cpm",
    "wilcoxon_paired": "participant_log1p_cpm",
    "dreamlet": "pseudobulk_counts",
    "limma_voom": "pseudobulk_counts",
    "edger_qlf": "pseudobulk_counts",
    "nebula": "cell_counts",
}

_META_COLS = ("participant", "visit", "arm")


def participant_log1p_cpm(
    pseudobulk_counts: pd.DataFrame,
    panel_genes: list[str],
    gene_cols: list[str] | None = None,
) -> pd.DataFrame:
    """``log(1 + CPM)`` per participant-visit, normalised on the TRANSCRIPTOME.

    The denominator is the sum over ``gene_cols`` (the full transcriptome), not
    over ``panel_genes``. Normalising within the tested panel makes the reference
    move with the signal: under a coordinated effect the panel total shifts, and
    every null gene in the panel acquires an offsetting apparent effect. That
    artifact was previously attributed to empirical-Bayes moderation.

    Panel selection happens AFTER normalisation, which is what a real workflow
    does and what makes panel size separable from normalisation scope.
    """
    genes = list(pseudobulk_counts.columns) if gene_cols is None else list(gene_cols)
    genes = [g for g in genes if g not in _META_COLS and g != "n_cells"]
    mat = pseudobulk_counts[genes].to_numpy(dtype=np.float64)
    total = mat.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    cpm = mat / total * 1e6

    pos = {g: i for i, g in enumerate(genes)}
    keep = [g for g in panel_genes if g in pos]
    meta = pseudobulk_counts[[c for c in _META_COLS if c in pseudobulk_counts.columns]]
    # Built in one concat rather than column by column: inserting 2,000 columns
    # individually fragments the frame and dominates the iteration's runtime,
    # which would then be attributed to the method being timed.
    panel_df = pd.DataFrame(
        np.log1p(cpm[:, [pos[g] for g in keep]]), columns=keep, index=pseudobulk_counts.index
    )
    return pd.concat([meta, panel_df], axis=1)


def prepare_inputs(sim: dict, panel_genes: list[str]) -> dict:
    """Build every method's contracted input from one simulated dataset.

    Returns
    -------
    dict with
        ``participant_log1p_cpm``  outcome frame for sctrial / Wilcoxon
        ``pseudobulk_counts``      panel raw counts for the count-based models
        ``lib_size``               full-transcriptome total per pseudobulk sample
        ``cell_counts``            cell-level AnnData restricted to the panel
        ``cell_lib_size``          full-transcriptome library size per cell
        ``oracle``                 per-gene truth keyed by estimand name
        ``panel_genes``            the tested panel
    """
    pb = sim["pseudobulk_counts"]
    gene_cols = list(sim["gene_names"])
    panel = [g for g in panel_genes if g in set(gene_cols)]

    mat = pb[gene_cols].to_numpy(dtype=np.float64)
    lib_size = mat.sum(axis=1)

    outcome = participant_log1p_cpm(pb, panel, gene_cols=gene_cols)

    counts = pd.concat(
        [pb[[c for c in _META_COLS if c in pb.columns]], pb[panel].astype(np.int64)], axis=1
    )

    adata = sim["adata"]
    cell_lib = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
    cell_view = adata[:, panel]

    oracle = sim.get("oracle", {})
    return {
        "participant_log1p_cpm": outcome,
        "pseudobulk_counts": counts,
        "lib_size": lib_size,
        "cell_counts": cell_view,
        "cell_lib_size": cell_lib,
        "oracle": oracle,
        "panel_genes": panel,
        "gene_cols": gene_cols,
    }


def truth_for(method: str, oracle: dict, gene: str, injected: float) -> float:
    """The value ``method`` should be scored against for ``gene``.

    Falls back to the injected effect when no oracle is available (which is the
    correct behaviour for the count-link estimand, and is flagged for the
    ``log1p_cpm`` estimand because there the two genuinely differ).
    """
    scale = METHOD_ESTIMAND.get(method, "count_link")
    table = oracle.get(scale)
    if table is None:
        return injected
    return float(table.get(gene, injected))


def prepare_inputs_from_adata(
    adata,
    panel_genes: list[str],
    participant_col: str = "participant",
    visit_col: str = "visit",
    arm_col: str = "arm",
    counts_layer: str = "counts",
) -> dict:
    """The same contracts, applied to REAL data.

    The permutation and subsampling analyses run the identical methods on real
    cohorts and must therefore hand them the identical representations. They
    previously built their own pseudobulk and normalised inside the tested panel,
    so the real-data results were produced under a different normalisation scope
    from the simulation used to characterise those same methods. Sharing this
    function is what stops the two drifting apart again.

    The CPM denominator is the whole measured transcriptome, not the tested
    panel, exactly as in :func:`prepare_inputs`.
    """
    obs = adata.obs
    X = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    gene_cols = [str(g) for g in adata.var_names]
    panel = [g for g in panel_genes if g in set(gene_cols)]

    keys = [participant_col, visit_col, arm_col]
    keys = [k for k in keys if k in obs.columns]
    groups = obs.groupby(keys, observed=True).indices

    rows, mats = [], []
    for key, idx in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        sub = X[idx]
        block = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        mats.append(block.sum(axis=0))
        meta = dict(zip(keys, [str(k) for k in key]))
        rows.append(
            {
                "participant": meta.get(participant_col, ""),
                "visit": meta.get(visit_col, ""),
                "arm": meta.get(arm_col, "Treated"),
                "n_cells": len(idx),
            }
        )
    mat = np.vstack(mats)
    pb = pd.concat(
        [pd.DataFrame(rows), pd.DataFrame(mat, columns=gene_cols)], axis=1
    )

    sim_like = {
        "pseudobulk_counts": pb,
        "gene_names": gene_cols,
        "adata": _canonical_adata(
            adata, participant_col, visit_col, arm_col, counts_layer
        ),
        # No truth exists for real data. An empty oracle is correct and explicit;
        # the permutation analysis is about the null distribution of p-values.
        "oracle": {},
    }
    return prepare_inputs(sim_like, panel)


def _canonical_adata(adata, participant_col, visit_col, arm_col, counts_layer):
    """A raw-count AnnData carrying the canonical obs column names.

    The runners take column names as arguments, but the contract layer needs one
    vocabulary. Renaming once here is safer than threading four more column-name
    arguments through every call site, where a mismatch fails silently.
    """
    import anndata as ad

    obs = adata.obs.rename(
        columns={participant_col: "participant", visit_col: "visit", arm_col: "arm"}
    )
    keep = [c for c in ("participant", "visit", "arm") if c in obs.columns]
    X = adata.layers[counts_layer] if counts_layer in adata.layers else adata.X
    out = ad.AnnData(X=X, obs=obs[keep].copy())
    out.var_names = adata.var_names
    return out
