from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData

from ..design import TrialDesign
from ._utils import apply_fdr, encode_visit, standardize_series
from .did import (
    MIN_CLUSTERS_FOR_ROBUST_SE,
    AggregateFunc,
    AggregateMode,
    _add_feature_columns,
    _prepare_did_obs,
)


def test_treatment_heterogeneity(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    biomarker_col: str,
    *,
    threshold: float | None = None,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    standardize: bool = True,
    agg: AggregateFunc = "mean",
    covariates: list[str] | None = None,
    fdr: Literal["feature", "all"] = "feature",
) -> pd.DataFrame:
    """Test treatment effect heterogeneity via a 3-way interaction.

    Fits: outcome ~ visit + arm + biomarker + visit:arm + visit:biomarker
         + arm:biomarker + visit:arm:biomarker + C(participant)

    The coefficient of the 3-way interaction (visit:arm:biomarker) tests whether
    the DiD effect differs by biomarker subgroup.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to test (genes or obs columns).
    design
        TrialDesign object.
    visits
        (baseline, followup) visit labels.
    biomarker_col
        Column in `adata.obs` for stratification.
    threshold
        If provided, biomarker_high = biomarker > threshold.
        If None, median split is used for numeric biomarkers. For binary
        biomarkers (0/1 or True/False), values are used directly.
    aggregate
        Aggregation mode (default: participant_visit).
    layer
        Expression layer for genes.
    standardize
        Whether to z-score outcomes before fitting.
    agg
        Aggregation function.
    covariates
        Additional covariates for the model.
    fdr
        FDR correction mode: per feature or across all results.

    Returns
    -------
    pd.DataFrame
        Table with heterogeneity effects for each feature.
    """
    if biomarker_col not in adata.obs.columns:
        raise KeyError(f"biomarker_col '{biomarker_col}' not found in adata.obs")

    ad, obs = _prepare_did_obs(adata, design, visits, celltype=None, exclude_crossovers=True)
    obs["arm_bin"] = (obs[design.arm_col] == design.arm_treated).astype(int)

    biomarker = adata.obs.loc[obs.index, biomarker_col]
    if pd.api.types.is_bool_dtype(biomarker) or set(pd.unique(biomarker.dropna())) <= {0, 1}:
        biomarker_high = biomarker.astype(int)
    else:
        biomarker = pd.to_numeric(biomarker, errors="coerce")
        if threshold is None:
            threshold = float(np.nanmedian(biomarker))
        biomarker_high = (biomarker > threshold).astype(int)

    obs["biomarker_high"] = biomarker_high.values

    # Add feature columns (genes or obs)
    obs_feat, final_features = _add_feature_columns(obs.copy(), ad, list(features), layer)

    if aggregate != "participant_visit":
        raise ValueError("test_treatment_heterogeneity supports participant_visit only.")

    grp_cols = [design.participant_col, design.visit_col, design.arm_col]
    use_cols = list(final_features) + ["biomarker_high", "arm_bin"]
    df_use = (
        obs_feat.groupby(grp_cols, observed=True)[use_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    df_use = encode_visit(df_use, design.visit_col, visits)
    unit = design.participant_col
    time = "visit_num"
    arm_bin = "arm_bin"
    obs_feat = df_use

    rows = []
    for feat in final_features:
        tmp = obs_feat.copy()
        if tmp.columns.duplicated().any():
            tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()
        tmp = tmp.dropna(subset=[feat, "biomarker_high"])
        if tmp[unit].nunique() < MIN_CLUSTERS_FOR_ROBUST_SE:
            continue

        # Handle duplicate column names by selecting the first column
        col_data = tmp[feat]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
            tmp[feat] = col_data

        if standardize:
            y_std, ok = standardize_series(tmp, feat, min_std=1e-8)
            if not ok:
                continue
            tmp["outcome_std"] = y_std
        else:
            tmp["outcome_std"] = tmp[feat].astype(float)

        df = tmp.dropna(subset=["outcome_std"])

        if df["biomarker_high"].nunique() < 2:
            continue

        formula = (
            f"outcome_std ~ {time} + {arm_bin} + biomarker_high + "
            f"{time}:{arm_bin} + {time}:biomarker_high + {arm_bin}:biomarker_high + "
            f"{time}:{arm_bin}:biomarker_high + C({unit})"
        )
        if covariates:
            formula += " + " + " + ".join(covariates)

        fit = smf.ols(formula, data=df).fit(
            cov_type="cluster",
            cov_kwds={"groups": df[unit]},
        )
        term = "visit_num:arm_bin:biomarker_high"
        rows.append(
            {
                "feature": feat,
                "beta_heterogeneity": float(fit.params.get(term, np.nan)),
                "p_heterogeneity": float(fit.pvalues.get(term, np.nan)),
                "n_units": int(df[unit].nunique()),
                "threshold": threshold,
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    if fdr in {"all", "feature"}:
        res = apply_fdr(res, p_col="p_heterogeneity", fdr_col="FDR_heterogeneity")
    return res
