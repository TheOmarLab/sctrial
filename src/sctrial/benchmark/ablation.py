"""Ablation study — progressive component addition.

Tests what each component of sctrial contributes by running increasingly
complete variants:

1. Cell-level OLS (pseudoreplication baseline)
2. Pseudobulk OLS (aggregation only)
3. Pseudobulk + FE (fixed effects)
4. Pseudobulk + FE + CRSE (cluster-robust SE)
5. Full sctrial (FE + interaction + bootstrap)

EVERY RUNG USES THE SAME OUTCOME: ``log(1 + CPM)`` normalised on the full
transcriptome, computed per cell for the cell-level rung and per
participant-visit for the pseudobulk rungs. Previously the cell-level rung ran on
raw counts and the pseudobulk rungs on raw mean counts, so the ladder varied the
outcome scale at the same time as the aggregation and could not isolate either.
The whole point of an ablation is that exactly one thing changes per step.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cell_level_ols(adata, gene_cols: list[str], cell_lib_size=None) -> dict:
    """Naive OLS on cell-level data (the pseudoreplication baseline).

    The outcome is ``log(1 + CPM)`` per cell, with the CPM denominator taken from
    the FULL transcriptome via ``cell_lib_size`` — the same quantity and the same
    scope the pseudobulk rungs use. This rung differs from the next one only in
    treating each cell as an independent observation, which is the entire claim
    being demonstrated.
    """
    import statsmodels.api as sm

    obs = adata.obs.copy()
    obs["post"] = (obs["visit"] == "Post").astype(float)
    obs["treat"] = (obs["arm"] == "Treated").astype(float)
    obs["interact"] = obs["post"] * obs["treat"]

    X_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    X_mat = np.asarray(X_mat, dtype=float)
    if cell_lib_size is None:
        raise ValueError(
            "cell_lib_size is required: a panel-derived denominator makes the "
            "normalisation reference move with the signal (see "
            "sctrial.benchmark.contracts)"
        )
    denom = np.asarray(cell_lib_size, dtype=float).copy()
    denom[denom <= 0] = 1.0
    X_mat = np.log1p(X_mat / denom[:, None] * 1e6)

    out = {}
    for i, gene in enumerate(gene_cols):
        try:
            y = X_mat[:, i].astype(float)
            X = sm.add_constant(obs[["post", "treat", "interact"]].values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit()
            out[gene] = {
                "beta": float(fit.params[3]),
                "pvalue": float(fit.pvalues[3]),
                "ci_lo": float(fit.conf_int()[3, 0]),
                "ci_hi": float(fit.conf_int()[3, 1]),
            }
        except Exception:
            out[gene] = {"beta": np.nan, "pvalue": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    return out


def _pseudobulk_ols(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """OLS on pseudobulk without fixed effects."""
    import statsmodels.api as sm

    pb = pb.copy()
    pb["post"] = (pb["visit"] == "Post").astype(float)
    pb["treat"] = (pb["arm"] == "Treated").astype(float)
    pb["interact"] = pb["post"] * pb["treat"]

    out = {}
    for gene in gene_cols:
        try:
            y = pb[gene].values.astype(float)
            X = sm.add_constant(pb[["post", "treat", "interact"]].values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit()
            out[gene] = {
                "beta": float(fit.params[3]),
                "pvalue": float(fit.pvalues[3]),
                "ci_lo": float(fit.conf_int()[3, 0]),
                "ci_hi": float(fit.conf_int()[3, 1]),
            }
        except Exception:
            out[gene] = {"beta": np.nan, "pvalue": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    return out


def _pseudobulk_fe(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """OLS on pseudobulk WITH participant fixed effects."""
    import statsmodels.api as sm

    pb = pb.copy()
    pb["post"] = (pb["visit"] == "Post").astype(float)
    pb["treat"] = (pb["arm"] == "Treated").astype(float)
    pb["interact"] = pb["post"] * pb["treat"]
    dummies = pd.get_dummies(pb["participant"], drop_first=True, dtype=float)

    out = {}
    for gene in gene_cols:
        try:
            y = pb[gene].values.astype(float)
            X_base = pb[["post", "treat", "interact"]].values
            X = np.column_stack([np.ones(len(y)), X_base, dummies.values])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit()
            # interaction is at index 3
            out[gene] = {
                "beta": float(fit.params[3]),
                "pvalue": float(fit.pvalues[3]),
                "ci_lo": float(fit.conf_int()[3, 0]),
                "ci_hi": float(fit.conf_int()[3, 1]),
            }
        except Exception:
            out[gene] = {"beta": np.nan, "pvalue": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    return out


def _pseudobulk_fe_crse(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """OLS on pseudobulk + FE + cluster-robust SE."""
    import statsmodels.api as sm

    pb = pb.copy()
    pb["post"] = (pb["visit"] == "Post").astype(float)
    pb["treat"] = (pb["arm"] == "Treated").astype(float)
    pb["interact"] = pb["post"] * pb["treat"]
    dummies = pd.get_dummies(pb["participant"], drop_first=True, dtype=float)

    out = {}
    for gene in gene_cols:
        try:
            y = pb[gene].values.astype(float)
            X_base = pb[["post", "treat", "interact"]].values
            X = np.column_stack([np.ones(len(y)), X_base, dummies.values])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.OLS(y, X).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": pb["participant"].values},
                )
            out[gene] = {
                "beta": float(fit.params[3]),
                "pvalue": float(fit.pvalues[3]),
                "ci_lo": float(fit.conf_int()[3, 0]),
                "ci_hi": float(fit.conf_int()[3, 1]),
            }
        except Exception:
            out[gene] = {"beta": np.nan, "pvalue": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    return out


ABLATION_VARIANTS = {
    "cell_ols": ("Cell-level OLS", _cell_level_ols, "adata"),
    "pb_ols": ("Pseudobulk OLS", _pseudobulk_ols, "pseudobulk_means"),
    "pb_fe": ("Pseudobulk + FE", _pseudobulk_fe, "pseudobulk_means"),
    "pb_fe_crse": ("Pseudobulk + FE + CRSE", _pseudobulk_fe_crse, "pseudobulk_means"),
    "sctrial_full": ("Full sctrial", None, "adata"),  # uses sctrial runner
}


def run_ablation(
    inputs: dict,
    gene_cols: list[str],
    variants: list[str] | None = None,
) -> dict[str, dict]:
    """Run ablation variants on a single simulated/real dataset.

    Parameters
    ----------
    inputs : dict
        From :func:`sctrial.benchmark.contracts.prepare_inputs`. Supplying the
        prepared contract rather than a raw ``sim`` dict is what guarantees every
        rung sees the same outcome.
    gene_cols : list[str]
        Genes to test.
    variants : list[str]
        Which ablation variants to run. Default: all.

    Returns
    -------
    dict : variant_name → {gene → {"beta", "pvalue", "ci_lo", "ci_hi"}}
    """
    if variants is None:
        variants = list(ABLATION_VARIANTS.keys())

    results = {}
    for var_name in variants:
        _label, fn, _data_key = ABLATION_VARIANTS[var_name]

        if var_name == "sctrial_full":
            from .runners.sctrial_did import run as run_sctrial

            results[var_name] = run_sctrial(
                inputs["participant_log1p_cpm"], gene_cols, from_pseudobulk=True
            )
        elif var_name == "cell_ols":
            results[var_name] = _cell_level_ols(
                inputs["cell_counts"], gene_cols, cell_lib_size=inputs["cell_lib_size"]
            )
        elif fn is not None:
            results[var_name] = fn(inputs["participant_log1p_cpm"], gene_cols)

    return results
