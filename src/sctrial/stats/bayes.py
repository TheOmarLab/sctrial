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

__all__ = ["did_table_bayes", "prior_predictive_check"]


def _build_did_model(
    unit_codes: np.ndarray,
    n_units: int,
    time_vals: np.ndarray,
    arm_vals: np.ndarray,
    y: np.ndarray,
    prior_scale: float = 1.0,
    sigma_scale: float = 1.0,
    covariate_matrix: np.ndarray | None = None,
):
    """Build the Bayesian DiD PyMC model and return it as a context manager.

    This is a shared helper used by both ``did_table_bayes`` and
    ``prior_predictive_check`` so that the prior specification stays in sync.

    Parameters
    ----------
    covariate_matrix
        Optional (n_obs, n_covariates) array of covariate values to include
        as additional fixed effects.
    """
    try:
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "pymc is required for Bayesian DiD. Install with: pip install sctrial[bayes]"
        ) from exc

    interaction = time_vals * arm_vals
    model = pm.Model()
    with model:
        sigma = pm.HalfNormal("sigma", sigma_scale)
        alpha = pm.Normal("alpha", 0.0, prior_scale, shape=n_units)
        beta_time = pm.Normal("beta_time", 0.0, prior_scale)
        beta_arm = pm.Normal("beta_arm", 0.0, prior_scale)
        beta_did = pm.Normal("beta_did", 0.0, prior_scale)
        mu = (
            alpha[unit_codes]
            + beta_time * time_vals
            + beta_arm * arm_vals
            + beta_did * interaction
        )
        # Add covariate fixed effects if provided
        if covariate_matrix is not None and covariate_matrix.shape[1] > 0:
            n_cov = covariate_matrix.shape[1]
            beta_cov = pm.Normal("beta_cov", 0.0, prior_scale, shape=n_cov)
            mu = mu + pm.math.dot(covariate_matrix, beta_cov)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    return model


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
    prior_scale: float = 1.0,
    sigma_scale: float = 1.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    max_treedepth: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Bayesian DiD with participant random intercepts.

    Fits a simple hierarchical model:

        y_ijt = alpha_i + beta_time * time + beta_arm * arm + beta_did * time*arm + X_cov * beta_cov + eps

    Prior specification
    -------------------
    - alpha_i ~ Normal(0, prior_scale) (participant random intercepts)
    - beta_time ~ Normal(0, prior_scale)
    - beta_arm ~ Normal(0, prior_scale)
    - beta_did ~ Normal(0, prior_scale)
    - sigma ~ HalfNormal(sigma_scale)

    These weakly-informative priors are designed to stabilize estimation for
    small samples while remaining agnostic about effect direction.  Use
    ``prior_scale`` and ``sigma_scale`` to widen or tighten them.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    features
        Feature names (genes or ``obs`` columns) to test.
    design
        A ``TrialDesign`` specifying participant, visit, and arm columns.
    visits
        Tuple of ``(baseline, followup)`` visit labels.
    exclude_crossovers
        Whether to exclude crossover participants.
    celltype
        If provided, subset to this cell type before analysis.
    aggregate
        Aggregation mode (``"participant_visit"`` or ``"cell"``).
    layer
        Expression layer for gene features (``None`` uses ``adata.X``).
    standardize
        Whether to z-score outcomes before model fitting.
    agg
        Aggregation function (``"mean"``, ``"sum"``, etc.).
    covariates
        Optional covariate columns to include as fixed effects.
    prior_scale
        Scale (standard deviation) for all Normal priors on coefficients and
        random intercepts.  Increase to make priors more diffuse.
    sigma_scale
        Scale for the HalfNormal prior on the observation noise ``sigma``.
    draws
        Number of posterior draws (after tuning).
    tune
        Number of tuning (warm-up) iterations per chain.
    chains
        Number of MCMC chains.
    target_accept
        Target acceptance rate for the NUTS sampler.
    max_treedepth
        Maximum tree depth for the NUTS sampler.
    seed
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
        - feature: Feature name
        - beta_DiD: Posterior mean of the DiD effect
        - ci_low: 2.5th percentile of posterior
        - ci_high: 97.5th percentile of posterior
        - p_bayes: Two-sided posterior tail probability (not a frequentist p-value)
        - n_units: Number of paired units used
        - FDR_bayes: BH-adjusted p_bayes
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

    cov_cols = covariates or []
    rows = []
    for feat in final_features:
        select_cols = [unit, time, arm_bin, feat] + cov_cols
        df_feat = df_use[select_cols].dropna()
        if df_feat.empty:
            rows.append({"feature": feat, "beta_DiD": np.nan, "n_units": 0})
            continue

        # participant index for random intercepts (remove unused categories)
        unit_cat = df_feat[unit].astype("category").cat.remove_unused_categories()
        unit_codes = unit_cat.cat.codes.to_numpy()
        n_units = int(unit_cat.cat.categories.size)
        time_vals = df_feat[time].to_numpy()
        arm_vals = df_feat[arm_bin].to_numpy()

        y = df_feat[feat].astype(float).to_numpy()
        if standardize:
            y_std = y.std(ddof=1)
            if not np.isfinite(y_std) or y_std < 1e-12:
                rows.append({"feature": feat, "beta_DiD": np.nan, "n_units": n_units})
                continue
            y = (y - y.mean()) / y_std

        # Build covariate matrix if covariates are specified
        cov_matrix = None
        if cov_cols:
            non_numeric = [
                c for c in cov_cols
                if not pd.api.types.is_numeric_dtype(df_feat[c])
            ]
            if non_numeric:
                raise TypeError(
                    f"Bayesian DiD requires numeric covariates but got "
                    f"non-numeric column(s): {non_numeric}. "
                    f"Encode categorical covariates (e.g. via pd.get_dummies) "
                    f"before passing them."
                )
            cov_matrix = df_feat[cov_cols].to_numpy(dtype=float)

        model = _build_did_model(
            unit_codes, n_units, time_vals, arm_vals, y,
            prior_scale=prior_scale, sigma_scale=sigma_scale,
            covariate_matrix=cov_matrix,
        )
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts={"max_treedepth": max_treedepth},
                random_seed=seed,
                progressbar=False,
            )

        post = idata.posterior["beta_did"].to_numpy().ravel()
        beta_mean = float(post.mean())
        ci_low, ci_high = np.quantile(post, [0.025, 0.975])
        n_post = len(post)
        p_bayes = max(2 * min((post > 0).mean(), (post < 0).mean()), 1.0 / n_post)

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

    res = pd.DataFrame(rows).sort_values("p_bayes", na_position="last")
    res = apply_fdr(res, p_col="p_bayes", fdr_col="FDR_bayes")
    return res.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prior predictive checking
# ---------------------------------------------------------------------------


def prior_predictive_check(
    adata: AnnData,
    features: list[str],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    prior_scale: float = 1.0,
    sigma_scale: float = 1.0,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """Run prior predictive check for the Bayesian DiD model.

    Samples from the prior predictive distribution to verify that the chosen
    priors produce plausible outcome values *before* fitting the model.  This
    is useful for calibrating ``prior_scale`` and ``sigma_scale``.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    features
        Feature names (genes or ``obs`` columns) to check.
    design
        A ``TrialDesign`` specifying participant, visit, and arm columns.
    visits
        Tuple of ``(baseline, followup)`` visit labels.
    prior_scale
        Scale (standard deviation) for all Normal priors on coefficients and
        random intercepts.
    sigma_scale
        Scale for the HalfNormal prior on the observation noise ``sigma``.
    n_samples
        Number of prior predictive samples to draw.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:

        - ``prior_predictive`` : np.ndarray -- prior predictive samples for
          the outcome variable (shape depends on number of observations and
          ``n_samples``).
        - ``observed_range`` : tuple[float, float] -- ``(min, max)`` of the
          observed data across all requested features.
        - ``prior_covers_data`` : bool -- whether the 95 % prior predictive
          interval covers the full observed data range.
    """
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymc is required for prior_predictive_check") from exc

    ad, obs = _prepare_did_obs(adata, design, visits, celltype=None, exclude_crossovers=True)

    cols = [design.participant_col, design.visit_col, design.arm_col, "visit_num", "arm_bin"]
    if design.celltype_col and design.celltype_col in obs.columns:
        cols.append(design.celltype_col)

    df = obs[cols].copy()
    df, final_features = _add_feature_columns(df, ad, features, layer=None)

    df_use, unit, time, arm_bin = _aggregate_for_did(
        df,
        final_features,
        design,
        visits,
        "participant_visit",
        "mean",
        None,
    )

    # Collect observed values across all requested features for the range
    all_observed: list[np.ndarray] = []
    # We build the model from the first usable feature (they share the same
    # design matrix) and draw prior predictive samples once.
    model = None
    for feat in final_features:
        df_feat = df_use[[unit, time, arm_bin, feat]].dropna()
        if df_feat.empty:
            continue

        y = df_feat[feat].astype(float).to_numpy()
        all_observed.append(y)

        if model is None:
            unit_cat = df_feat[unit].astype("category").cat.remove_unused_categories()
            unit_codes = unit_cat.cat.codes.to_numpy()
            n_units = int(unit_cat.cat.categories.size)
            time_vals = df_feat[time].to_numpy()
            arm_vals = df_feat[arm_bin].to_numpy()
            model = _build_did_model(
                unit_codes, n_units, time_vals, arm_vals, y,
                prior_scale=prior_scale, sigma_scale=sigma_scale,
            )

    if model is None:
        raise ValueError("No usable features found for prior predictive check.")

    with model:
        idata = pm.sample_prior_predictive(samples=n_samples, random_seed=seed)

    prior_pred = idata.prior_predictive["y"].to_numpy().ravel()

    obs_all = np.concatenate(all_observed)
    obs_min, obs_max = float(obs_all.min()), float(obs_all.max())

    pp_low, pp_high = np.quantile(prior_pred, [0.025, 0.975])
    prior_covers_data = bool(pp_low <= obs_min and pp_high >= obs_max)

    return {
        "prior_predictive": prior_pred,
        "observed_range": (obs_min, obs_max),
        "prior_covers_data": prior_covers_data,
    }
