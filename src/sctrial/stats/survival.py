from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ..design import TrialDesign
from ._extract import extract_gene_matrix
from ._utils import aggregate_features, apply_fdr

__all__ = ["hazard_regression_with_features"]


def hazard_regression_with_features(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    time_col: str,
    event_col: str,
    *,
    visit: str | None = None,
    layer: str | None = None,
    agg: str = "mean",
    covariates: Sequence[str] | None = None,
    standardize: bool = True,
    fdr: bool = True,
) -> pd.DataFrame:
    """Cox proportional hazards regression using single-cell features.

    This function aggregates features to the participant level and fits a Cox model
    for each feature, optionally adjusting for covariates.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to test (obs columns or gene names in var_names).
    design
        TrialDesign with participant and visit columns.
    time_col
        Column in adata.obs containing survival time.
    event_col
        Column in adata.obs indicating event (1) or censoring (0).
    visit
        Optional visit label to subset prior to aggregation (e.g., baseline).
    layer
        Expression layer to use for gene features.
    agg
        Aggregation function for features ("mean", "median", or "pct_pos").
    covariates
        Optional covariate columns to include in the Cox model.
    standardize
        If True, z-score each feature before fitting.
    fdr
        If True, add FDR correction across all features.

    Returns
    -------
    pd.DataFrame
        One row per feature with hazard ratio (HR), confidence interval, and p-value.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lifelines is required for survival analysis") from exc

    obs = adata.obs.copy()
    if visit is not None:
        obs = obs[obs[design.visit_col] == visit].copy()

    base_cols = [design.participant_col, time_col, event_col]
    if covariates:
        base_cols += list(covariates)
    missing = [c for c in base_cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required columns in adata.obs: {missing}")

    # Validate survival times are positive before building df
    valid_mask = obs[time_col] > 0
    if not valid_mask.all():
        n_bad = int((~valid_mask).sum())
        obs = obs[valid_mask].copy()
        if obs.empty:
            raise ValueError(
                f"All {n_bad} observations have non-positive survival times in '{time_col}'."
            )

    df = obs[base_cols].copy()

    obs_feats = [f for f in features if f in obs.columns]
    gene_feats = [f for f in features if f in adata.var_names and f not in obs.columns]

    for feat in obs_feats:
        df[feat] = obs[feat].values

    if gene_feats:
        mat = extract_gene_matrix(adata[obs.index], gene_feats, layer=layer)
        df_genes = pd.DataFrame(mat, columns=gene_feats, index=df.index)
        df = pd.concat([df, df_genes], axis=1)

    grp = design.participant_col

    # Enforce constant time/event/covariates within participant
    for col in base_cols[1:]:
        if df.groupby(grp, observed=True)[col].nunique().max() > 1:
            raise ValueError(f"Column '{col}' varies within participant; cannot aggregate.")

    # Aggregate features to participant level
    feat_cols = list(obs_feats) + list(gene_feats)
    df_feats = aggregate_features(df, [grp], feat_cols, agg)
    df_meta = df.groupby(grp, observed=True)[base_cols[1:]].first().reset_index()
    df_part = pd.merge(df_meta, df_feats, on=grp, how="inner")
    # Ensure survival columns are retained after merge
    for col in base_cols[1:]:
        if col not in df_part.columns:
            df_part[col] = df_meta.set_index(grp).loc[df_part[grp], col].values

    results = []
    for feat in feat_cols:
        df_model = df_part[[time_col, event_col, feat] + (list(covariates) if covariates else [])].dropna()
        if df_model.shape[0] < 5:
            results.append({"feature": feat, "HR": np.nan, "p": np.nan, "n": df_model.shape[0]})
            continue
        if standardize:
            std = df_model[feat].std(ddof=1)
            if not np.isfinite(std) or std < 1e-12:
                results.append({"feature": feat, "HR": np.nan, "p": np.nan, "n": df_model.shape[0]})
                continue
            df_model[feat] = (df_model[feat] - df_model[feat].mean()) / std

        cph = CoxPHFitter()
        cph.fit(df_model, duration_col=time_col, event_col=event_col)
        coef = cph.params_[feat]
        hr = float(np.exp(coef))
        ci = cph.confidence_intervals_.loc[feat]
        hr_low = float(np.exp(ci.iloc[0]))
        hr_high = float(np.exp(ci.iloc[1]))
        p_val = float(cph.summary.loc[feat, "p"])

        results.append(
            {
                "feature": feat,
                "HR": hr,
                "HR_low": hr_low,
                "HR_high": hr_high,
                "p": p_val,
                "n": df_model.shape[0],
            }
        )

    out = pd.DataFrame(results).sort_values("p")
    if fdr:
        out = apply_fdr(out, p_col="p", fdr_col="FDR")
    return out.reset_index(drop=True)
