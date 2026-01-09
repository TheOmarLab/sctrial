from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from ..adata_tools import subset_cells
from ..design import TrialDesign
from .did import AggregateFunc, AggregateMode, _ensure_paired
from ._utils import aggregate_features, encode_visit, standardize_series


def within_arm_comparison(
    adata: AnnData,
    arm: str,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    agg: AggregateFunc = "mean",
    standardize: bool = True,
) -> pd.DataFrame:
    """Paired within-arm pre->post contrast.

    This function tests for longitudinal changes within a single treatment arm
    using a fixed-effects model (equivalent to a paired t-test but flexible
    for single-cell data).

    Parameters
    ----------
    adata
        AnnData object.
    arm
        The arm to analyze (e.g., design.arm_treated).
    features
        List of genes or module scores.
    design
        A `TrialDesign` object.
    visits
        Tuple of (pre, post) visit labels.
    aggregate
        Aggregation mode (see `did_table`).
    layer
        Layer to use for gene expression.
    agg
        Aggregation function.
    standardize
        Whether to z-score the outcome variable.

    Returns
    -------
    pd.DataFrame
        Table with beta_time and p_time.
    """
    # Subset to arm and visits
    ad = subset_cells(adata, design, arm=arm, exclude_crossovers=False)
    ad = ad[ad.obs[design.visit_col].isin(visits)].copy()

    obs = encode_visit(ad.obs.copy(), design.visit_col, visits)

    # build dataframe
    cols = [design.participant_col, design.visit_col, "visit_num"]
    df = obs[cols].copy()

    for feat in features:
        if feat in ad.obs.columns:
            df[feat] = ad.obs[feat].values
        elif feat in ad.var_names:
            from ._extract import extract_gene_vector
            df[feat] = extract_gene_vector(ad, feat, layer=layer)
        else:
            raise KeyError(f"Feature {feat} not found.")

    # Aggregate
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col]
        df_use = aggregate_features(df, grp_cols=grp_cols, features=features, agg=agg)
        unit = design.participant_col
    else:
        df_use = df.copy()
        unit = design.participant_col

    df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
    df_use = encode_visit(df_use, design.visit_col, visits)

    rows = []
    for feat in features:
        # Create a fresh copy for each feature to avoid cross-contamination
        df_feat = df_use.copy()

        if standardize:
                y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
                if not ok:
                    # Skip features with near-zero variance
                    rows.append({
                        "feature": feat,
                        "beta_time": np.nan,
                        "p_time": np.nan,
                        "n_units": int(df_feat[unit].nunique()),
                    })
                    continue
                df_feat["_y"] = y_std
        else:
            df_feat["_y"] = df_feat[feat].astype(float)

        model = smf.ols(f"_y ~ visit_num + C({unit})", data=df_feat)
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df_feat[unit]})

        rows.append({
            "feature": feat,
            "beta_time": float(fit.params.get("visit_num", np.nan)),
            "p_time": float(fit.pvalues.get("visit_num", np.nan)),
            "n_units": int(df_use[unit].nunique()),
        })

    res = pd.DataFrame(rows)
    mask = res["p_time"].notna()
    res["FDR_time"] = np.nan
    if mask.sum() > 0:
        res.loc[mask, "FDR_time"] = multipletests(res.loc[mask, "p_time"], method="fdr_bh")[1]
    return res

def between_arm_comparison(
    adata: AnnData,
    visit: str,
    features: Sequence[str],
    design: TrialDesign,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    agg: AggregateFunc = "mean",
    standardize: bool = True,
    method: Literal["ols", "wilcoxon"] = "ols",
) -> pd.DataFrame:
    """Between-arm contrast at a fixed visit.

    This function tests if treatment arms differ at a specific visit. This
    is a cross-sectional comparison (no participant fixed effects).

    Parameters
    ----------
    adata
        AnnData object.
    visit
        The visit label to analyze.
    features
        List of genes or module scores.
    design
        A `TrialDesign` object.
    aggregate
        Aggregation mode (see `did_table`).
    layer
        Layer to use for gene expression.
    agg
        Aggregation function.
    standardize
        Whether to z-score the outcome variable (only for 'ols').
    method
        - 'ols': Ordinary Least Squares.
        - 'wilcoxon': Wilcoxon rank-sum test (Mann-Whitney U).
    """
    ad = subset_cells(adata, design, visit=visit, exclude_crossovers=False)

    obs = ad.obs.copy()
    obs["arm_bin"] = (obs[design.arm_col] == design.arm_treated).astype(int)

    cols = [design.participant_col, "arm_bin", design.arm_col]
    df = obs[cols].copy()

    for feat in features:
        if feat in ad.obs.columns:
            df[feat] = ad.obs[feat].values
        elif feat in ad.var_names:
            from ._extract import extract_gene_vector
            df[feat] = extract_gene_vector(ad, feat, layer=layer)
        else:
            raise KeyError(f"Feature {feat} not found.")

    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, "arm_bin", design.arm_col]
        df_use = aggregate_features(df, grp_cols=grp_cols, features=features, agg=agg)
    else:
        df_use = df.copy()

    rows = []
    for feat in features:
        if method == "ols":
            # Create a fresh copy for each feature to avoid cross-contamination
            df_feat = df_use.copy()

            if standardize:
                y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
                if not ok:
                    # Skip features with near-zero variance
                    rows.append({
                        "feature": feat,
                        "beta_arm": np.nan,
                        "p_arm": np.nan,
                        "n_units": int(df_feat[design.participant_col].nunique()),
                    })
                    continue
                df_feat["_y"] = y_std
            else:
                df_feat["_y"] = df_feat[feat].astype(float)

            model = smf.ols("_y ~ arm_bin", data=df_feat)
            fit = model.fit()

            rows.append({
                "feature": feat,
                "beta_arm": float(fit.params.get("arm_bin", np.nan)),
                "p_arm": float(fit.pvalues.get("arm_bin", np.nan)),
                "n_units": int(df_feat[design.participant_col].nunique()),
            })
        elif method == "wilcoxon":
            from scipy.stats import mannwhitneyu
            g1 = np.asarray(df_use[df_use["arm_bin"] == 1][feat].values, dtype=float)
            g2 = np.asarray(df_use[df_use["arm_bin"] == 0][feat].values, dtype=float)

            if len(g1) > 0 and len(g2) > 0:
                stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
                rows.append({
                    "feature": feat,
                    "beta_arm": float(np.mean(g1) - np.mean(g2)),
                    "p_arm": float(p_val),
                    "n_units": int(df_use[design.participant_col].nunique()),
                })
            else:
                rows.append({
                    "feature": feat,
                    "beta_arm": np.nan,
                    "p_arm": np.nan,
                    "n_units": int(df_use[design.participant_col].nunique()),
                })

    res = pd.DataFrame(rows)
    mask = res["p_arm"].notna()
    res["FDR_arm"] = np.nan
    if mask.sum() > 0:
        res.loc[mask, "FDR_arm"] = multipletests(res.loc[mask, "p_arm"], method="fdr_bh")[1]
    return res
