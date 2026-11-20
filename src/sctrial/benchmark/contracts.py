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
    "CANONICAL_ARMS",
    "CANONICAL_VISITS",
    "canonical_cohort",
    "require_raw_counts",
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
"""The estimand scale each method targets: ``log1p_cpm`` or ``count_link``."""

METHOD_INPUT: dict[str, str] = {
    "sctrial_did": "participant_log1p_cpm",
    "sctrial_mixed": "participant_log1p_cpm",
    "wilcoxon_paired": "participant_log1p_cpm",
    "dreamlet": "pseudobulk_counts",
    "limma_voom": "pseudobulk_counts",
    "edger_qlf": "pseudobulk_counts",
    "nebula": "cell_counts",
}
"""The input representation each method expects: ``participant_log1p_cpm``, ``pseudobulk_counts``, or ``cell_counts``."""

_META_COLS = ("participant", "visit", "arm")

CANONICAL_VISITS = ("Pre", "Post")
"""The visit labels every runner reads: (baseline, follow-up)."""

CANONICAL_ARMS = ("Treated", "Control")
"""The arm labels every runner reads: (treated, control)."""


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


def _count_matrix(adata, counts_layer: str | None):
    """The matrix the caller named as raw counts; ``None`` names ``adata.X``.

    A named layer that is absent raises. Falling back to ``adata.X`` would hand a
    normalised matrix to the count models of a cohort that has no counts.
    """
    if counts_layer is None:
        return adata.X
    if counts_layer not in adata.layers:
        raise KeyError(
            f"counts layer {counts_layer!r} not found (layers: {sorted(adata.layers)}). "
            "Name the layer that holds the counts, or pass counts_layer=None if they "
            "are in adata.X."
        )
    return adata.layers[counts_layer]


def _require_vocabulary(obs: pd.DataFrame, col: str, allowed: tuple[str, str]) -> None:
    """Refuse labels the runners do not read: they would match neither group."""
    values = obs[col]
    other = sorted(str(v) for v in pd.unique(values.dropna()) if v not in allowed)
    n_missing = int(values.isna().sum())
    if other or n_missing:
        found = ([f"labels {other}"] if other else []) + (
            [f"{n_missing} missing labels"] if n_missing else []
        )
        raise ValueError(
            f"obs[{col!r}] holds {' and '.join(found)}; the benchmark runners read "
            f"{allowed[0]!r} and {allowed[1]!r} only. Map a real cohort with "
            "canonical_cohort() first."
        )


def canonical_cohort(
    adata,
    *,
    participant_col: str,
    visit_col: str,
    visits: tuple[str, str],
    arm_col: str | None = None,
    arm_treated: str | None = None,
    arm_control: str | None = None,
    counts_layer: str | None = "counts",
):
    """A real cohort in the vocabulary of the benchmark.

    The runners read the columns ``participant``, ``visit`` and ``arm`` and the
    labels of :data:`CANONICAL_VISITS` and :data:`CANONICAL_ARMS`. A real cohort
    names them differently (``response`` with ``Responder``/``Non-responder``),
    and a label a runner does not read matches neither group: every method then
    fails for every gene. This function maps a cohort once, from labels the caller
    states; nothing is guessed.

    Parameters
    ----------
    adata
        The cohort.
    participant_col, visit_col
        Columns of ``adata.obs``.
    visits
        (baseline, follow-up) labels of ``visit_col``. Rows of other visits are
        left out.
    arm_col, arm_treated, arm_control
        The arm column and its two labels, for a two-arm cohort. Rows of any other
        arm, or with a missing arm label, are left out with one warning
        (:func:`sctrial.subset_arms`). ``arm_col=None`` declares a single-arm
        cohort.
    counts_layer
        Layer that holds raw counts; ``None`` names ``adata.X``.

    Returns
    -------
    AnnData
        Counts in ``X`` and ``obs`` columns ``participant``, ``visit``, ``arm``
        with the canonical labels.

    Raises
    ------
    KeyError
        If a column or the counts layer is absent.
    ValueError
        If a named visit or arm has no rows, if the arm labels of a two-arm
        cohort are not given, or if a participant label is missing.
    """
    import anndata as ad

    from ..adata_tools import subset_arms
    from ..design import TrialDesign

    obs = adata.obs
    absent_cols = [
        c for c in (participant_col, visit_col, arm_col) if c is not None and c not in obs.columns
    ]
    if absent_cols:
        raise KeyError(f"columns {absent_cols} not in adata.obs (columns: {list(obs.columns)})")
    baseline, followup = visits
    if baseline == followup:
        raise ValueError(f"visits must name two different labels, got {visits!r}")
    absent_visits = [v for v in (baseline, followup) if not obs[visit_col].isin([v]).any()]
    if absent_visits:
        present = sorted(str(v) for v in pd.unique(obs[visit_col].dropna()))
        raise ValueError(
            f"No rows of visit {' or '.join(repr(v) for v in absent_visits)} in "
            f"obs[{visit_col!r}] (labels present: {present})."
        )
    cohort = adata[obs[visit_col].isin([baseline, followup]).to_numpy()]

    if arm_col is None:
        arm_labels = np.full(cohort.n_obs, CANONICAL_ARMS[0], dtype=object)
    else:
        if arm_treated is None or arm_control is None:
            raise ValueError(
                "A two-arm cohort needs arm_treated and arm_control; pass arm_col=None "
                "for a single-arm cohort."
            )
        design = TrialDesign(
            participant_col=participant_col,
            visit_col=visit_col,
            arm_col=arm_col,
            arm_treated=arm_treated,
            arm_control=arm_control,
            celltype_col=None,
        )
        cohort = subset_arms(cohort, design, stacklevel=3)
        arm = cohort.obs[arm_col]
        absent_arms = [a for a in (arm_treated, arm_control) if not arm.isin([a]).any()]
        if absent_arms:
            raise ValueError(
                f"No rows of arm {' or '.join(repr(a) for a in absent_arms)} in "
                f"obs[{arm_col!r}] at visits {visits!r}."
            )
        arm_labels = np.where(arm.isin([arm_treated]).to_numpy(), *CANONICAL_ARMS)

    participant = cohort.obs[participant_col]
    if participant.isna().any():
        raise ValueError(f"obs[{participant_col!r}] has missing participant labels")
    canonical_obs = pd.DataFrame(
        {
            "participant": participant.astype(str).to_numpy(),
            "visit": np.where(
                cohort.obs[visit_col].isin([baseline]).to_numpy(), *CANONICAL_VISITS
            ),
            "arm": arm_labels,
        },
        index=cohort.obs_names,
    )
    out = ad.AnnData(X=_count_matrix(cohort, counts_layer).copy(), obs=canonical_obs)
    out.var_names = adata.var_names
    return out


def require_raw_counts(methods: list[str], cohort) -> None:
    """Refuse to run a count model on a matrix that does not hold counts.

    ``dreamlet``, ``limma_voom``, ``edger_qlf`` and ``nebula`` model raw counts.
    A cohort measured as TPM, or a normalised matrix named as the counts layer,
    has non-integer values.
    """
    from scipy import sparse

    count_models = [m for m in methods if METHOD_ESTIMAND.get(m) == "count_link"]
    if not count_models:
        return
    X = cohort.X
    values = X.data if sparse.issparse(X) else np.asarray(X)
    if values.size and (np.any(values < 0) or np.any(values != np.floor(values))):
        raise ValueError(
            f"{count_models} model raw counts, but the count matrix holds non-integer or "
            "negative values. Name a raw-count layer, or run only the methods that take "
            "log(1 + CPM) (METHOD_ESTIMAND == 'log1p_cpm')."
        )


def prepare_inputs_from_adata(
    adata,
    panel_genes: list[str],
    participant_col: str = "participant",
    visit_col: str = "visit",
    arm_col: str = "arm",
    counts_layer: str | None = "counts",
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

    The visit and arm labels must be those of :data:`CANONICAL_VISITS` and
    :data:`CANONICAL_ARMS`, which the runners read; any other label raises.
    :func:`canonical_cohort` maps a real cohort. ``counts_layer`` names the raw
    counts (``None`` for ``adata.X``) and raises when the layer is absent.
    """
    obs = adata.obs
    if visit_col in obs.columns:
        _require_vocabulary(obs, visit_col, CANONICAL_VISITS)
    if arm_col in obs.columns:
        _require_vocabulary(obs, arm_col, CANONICAL_ARMS)
    X = _count_matrix(adata, counts_layer)
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
    out = ad.AnnData(X=_count_matrix(adata, counts_layer), obs=obs[keep].copy())
    out.var_names = adata.var_names
    return out
