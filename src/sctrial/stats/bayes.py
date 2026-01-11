from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ..design import TrialDesign
from ._utils import apply_fdr
from .did import (
    AggregateFunc,
    AggregateMode,
    _add_feature_columns,
    _aggregate_for_did,
    _prepare_did_obs,
)

__all__ = ["did_table_bayes"]


def did_table_bayes(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    exclude_crossovers: bool = True,
    celltype: str | None = None,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    standardize: bool = True,
    agg: AggregateFunc = "mean",
    covariates: list[str] | None = None,
    draws: int = 1000,
    tune: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bayesian DiD with participant random intercepts.

    Fits a simple hierarchical model:

        y_ijt = alpha_i + beta_time * time + beta_arm * arm + beta_did * time*arm + eps

    Parameters
    ----------
    draws
        Posterior draws.
    tune
        Tuning iterations.
    """
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymc is required for did_table_bayes") from exc

    ad, obs = _prepare_did_obs(adata, design, visits, celltype, exclude_crossovers)

    cols = [design.participant_col, design.visit_col, design.arm_col, "visit_num", "arm_bin"]
    if design.celltype_col and design.celltype_col in obs.columns:
        cols.append(design.celltype_col)

    if covariates:
        for c in covariates:
            if c not in obs.columns:
                raise KeyError(f"Covariate '{c}' not found in adata.obs")
            cols.append(c)

    df = obs[cols].copy()
    df, final_features = _add_feature_columns(df, ad, features, layer)

    df_use, unit, time, arm_bin = _aggregate_for_did(
        df,
        final_features,
        design,
        visits,
        aggregate,
        agg,
        covariates,
    )

    # participant index for random intercepts
    unit_codes = df_use[unit].astype("category").cat.codes.to_numpy()
    n_units = int(df_use[unit].nunique())
    time_vals = df_use[time].to_numpy()
    arm_vals = df_use[arm_bin].to_numpy()
    interaction = time_vals * arm_vals

    rows = []
    for feat in final_features:
        y = df_use[feat].astype(float).to_numpy()
        if standardize:
            y_std = y.std(ddof=1)
            if not np.isfinite(y_std) or y_std < 1e-12:
                rows.append({"feature": feat, "beta_DiD": np.nan, "n_units": n_units})
                continue
            y = (y - y.mean()) / y_std

        with pm.Model():
            sigma = pm.HalfNormal("sigma", 1.0)
            alpha = pm.Normal("alpha", 0.0, 1.0, shape=n_units)
            beta_time = pm.Normal("beta_time", 0.0, 1.0)
            beta_arm = pm.Normal("beta_arm", 0.0, 1.0)
            beta_did = pm.Normal("beta_did", 0.0, 1.0)
            mu = alpha[unit_codes] + beta_time * time_vals + beta_arm * arm_vals + beta_did * interaction
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=2,
                target_accept=0.9,
                random_seed=seed,
                progressbar=False,
            )

        post = idata.posterior["beta_did"].to_numpy().ravel()
        beta_mean = float(post.mean())
        ci_low, ci_high = np.quantile(post, [0.025, 0.975])
        p_bayes = 2 * min((post > 0).mean(), (post < 0).mean())

        rows.append(
            {
                "feature": feat,
                "beta_DiD": beta_mean,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p_bayes": float(p_bayes),
                "n_units": n_units,
            }
        )

    res = pd.DataFrame(rows).sort_values("p_bayes")
    res = apply_fdr(res, p_col="p_bayes", fdr_col="FDR_bayes")
    return res.reset_index(drop=True)
