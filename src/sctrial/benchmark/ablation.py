"""Ablation study — progressive component addition.

Tests what each component of sctrial contributes by running
increasingly complete variants:

1. Cell-level OLS (pseudoreplication baseline)
2. Pseudobulk OLS (aggregation only)
3. Pseudobulk + FE (fixed effects)
4. Pseudobulk + FE + CRSE (cluster-robust SE)
5. Full sctrial (FE + interaction + bootstrap)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cell_level_ols(adata, gene_cols: list[str]) -> dict:
    """Naive OLS on cell-level data (pseudoreplication baseline)."""
    import statsmodels.api as sm

    obs = adata.obs.copy()
    obs["post"] = (obs["visit"] == "Post").astype(float)
    obs["treat"] = (obs["arm"] == "Treated").astype(float)
    obs["interact"] = obs["post"] * obs["treat"]

    X_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

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
            out[gene] = {"beta": np.nan, "pvalue": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan}
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
            out[gene] = {"beta": np.nan, "pvalue": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan}
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
            out[gene] = {"beta": np.nan, "pvalue": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan}
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
            out[gene] = {"beta": np.nan, "pvalue": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan}
    return out


ABLATION_VARIANTS = {
    "cell_ols": ("Cell-level OLS", _cell_level_ols, "adata"),
    "pb_ols": ("Pseudobulk OLS", _pseudobulk_ols, "pseudobulk_means"),
    "pb_fe": ("Pseudobulk + FE", _pseudobulk_fe, "pseudobulk_means"),
    "pb_fe_crse": ("Pseudobulk + FE + CRSE", _pseudobulk_fe_crse, "pseudobulk_means"),
    "sctrial_full": ("Full sctrial", None, "adata"),  # uses sctrial runner
}


def run_ablation(
    sim: dict,
    gene_cols: list[str],
    variants: list[str] | None = None,
) -> dict[str, dict]:
    """Run ablation variants on a single simulated/real dataset.

    Parameters
    ----------
    sim : dict
        Must have "adata" and "pseudobulk_means" keys (from simulate_trial).
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
        label, fn, data_key = ABLATION_VARIANTS[var_name]

        if var_name == "sctrial_full":
            from .runners.sctrial_fe import run as run_sctrial
            results[var_name] = run_sctrial(sim["adata"], gene_cols)
        else:
            results[var_name] = fn(sim[data_key], gene_cols)

    return results
