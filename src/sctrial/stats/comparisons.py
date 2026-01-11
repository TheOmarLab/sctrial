from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData
from scipy.stats import mannwhitneyu

from ..adata_tools import subset_cells
from ..design import TrialDesign
from ._utils import aggregate_features, apply_fdr, encode_visit, standardize_series
from .did import AggregateFunc, AggregateMode, MIN_CLUSTERS_FOR_ROBUST_SE, _ensure_paired


def _add_feature_columns(
    df: pd.DataFrame,
    ad: AnnData,
    features: Sequence[str],
    layer: str | None,
) -> pd.DataFrame:
    obs_feats = [f for f in features if f in ad.obs.columns]
    gene_feats = [f for f in features if f in ad.var_names and f not in ad.obs.columns]
    missing = [f for f in features if f not in ad.obs.columns and f not in ad.var_names]
    if missing:
        raise KeyError(f"Features not found in obs or var_names: {missing[:5]}")

    for feat in obs_feats:
        df[feat] = ad.obs[feat].values

    if gene_feats:
        from ._extract import extract_gene_matrix

        mat = extract_gene_matrix(ad, gene_feats, layer=layer)
        df_genes = pd.DataFrame(mat, columns=gene_feats, index=df.index)
        df = pd.concat([df, df_genes], axis=1)
    return df


def resolve_gene_name(adata: AnnData, gene_query: str) -> str:
    """Resolve a gene name in var_names, case-insensitive if needed."""
    if gene_query in adata.var_names:
        return gene_query
    candidates = [g for g in adata.var_names if g.upper() == gene_query.upper()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"Gene '{gene_query}' not found in adata.var_names.")
    raise ValueError(f"Gene '{gene_query}' is ambiguous: {candidates}")


def _prepare_between_arm_df(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visit: str,
    aggregate: AggregateMode,
    layer: str | None,
    agg: AggregateFunc,
) -> pd.DataFrame:
    ad = subset_cells(adata, design, visit=visit, exclude_crossovers=False)
    obs = ad.obs.copy()
    obs["arm_bin"] = (obs[design.arm_col] == design.arm_treated).astype(int)

    cols = [design.participant_col, "arm_bin", design.arm_col]
    df = obs[cols].copy()

    df = _add_feature_columns(df, ad, features, layer)

    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, "arm_bin", design.arm_col]
        df = aggregate_features(df, grp_cols=grp_cols, features=features, agg=agg)

    return df


def _ols_between_arm(
    df_use: pd.DataFrame,
    feat: str,
    design: TrialDesign,
    standardize: bool,
) -> dict:
    """Fit OLS model comparing arms at a single timepoint.

    Parameters
    ----------
    df_use
        DataFrame with feature values and arm assignments.
    feat
        Name of the feature column to analyze.
    design
        TrialDesign object with column specifications.
    standardize
        If True, z-score the outcome before fitting.

    Returns
    -------
    dict
        Dictionary with keys: feature, beta_arm, p_arm, n_units.
    """
    df_feat = df_use.copy()
    if standardize:
        y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
        if not ok:
            return {
                "feature": feat,
                "beta_arm": np.nan,
                "p_arm": np.nan,
                "n_units": int(df_feat[design.participant_col].nunique()),
            }
        df_feat["outcome_std"] = y_std
    else:
        df_feat["outcome_std"] = df_feat[feat].astype(float)

    model = smf.ols("outcome_std ~ arm_bin", data=df_feat)
    fit = model.fit()
    return {
        "feature": feat,
        "beta_arm": float(fit.params.get("arm_bin", np.nan)),
        "p_arm": float(fit.pvalues.get("arm_bin", np.nan)),
        "n_units": int(df_feat[design.participant_col].nunique()),
    }


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

    df = _add_feature_columns(df, ad, features, layer)

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
            df_feat["outcome_std"] = y_std
        else:
            df_feat["outcome_std"] = df_feat[feat].astype(float)

        model = smf.ols(f"outcome_std ~ visit_num + C({unit})", data=df_feat)
        n_units_feat = df_feat[unit].nunique()
        if n_units_feat < MIN_CLUSTERS_FOR_ROBUST_SE:
            warnings.warn(
                f"Only {n_units_feat} clusters (participants) available. Cluster-robust "
                f"standard errors are unreliable with fewer than {MIN_CLUSTERS_FOR_ROBUST_SE} "
                f"clusters.",
                UserWarning,
                stacklevel=2,
            )
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": df_feat[unit]})

        rows.append({
            "feature": feat,
            "beta_time": float(fit.params.get("visit_num", np.nan)),
            "p_time": float(fit.pvalues.get("visit_num", np.nan)),
            "n_units": int(df_use[unit].nunique()),
        })

    res = pd.DataFrame(rows)
    res = apply_fdr(res, p_col="p_time", fdr_col="FDR_time")
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
    df_use = _prepare_between_arm_df(
        adata=adata,
        features=features,
        design=design,
        visit=visit,
        aggregate=aggregate,
        layer=layer,
        agg=agg,
    )

    rows = []
    for feat in features:
        if method == "ols":
            rows.append(_ols_between_arm(df_use, feat, design, standardize))
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
    res = apply_fdr(res, p_col="p_arm", fdr_col="FDR_arm")
    return res


def compare_gene_in_celltype(
    adata: AnnData,
    gene: str,
    celltypes: str | Sequence[str],
    *,
    group_col: str,
    group1: str,
    group2: str,
    participant_col: str = "participant_id",
    celltype_col: str = "celltype",
    layer: str | None = "counts",
    log1p: bool = True,
    expr_threshold: float = 0.0,
    min_cells_per_patient: int = 10,
    min_patients_per_group: int = 3,
) -> tuple[dict, pd.DataFrame]:
    """Compare one gene between two groups within specified cell types.

    This aggregates expression per participant (avoids pseudoreplication) and
    tests group differences using Mann-Whitney U on participant-level means.

    Returns
    -------
    result : dict
        Summary stats including p-value and group means.
    df_patient : pd.DataFrame
        Participant-level summaries (mean, median, % expressing, n_cells).
    """
    if isinstance(celltypes, str):
        celltypes = [celltypes]

    if celltype_col not in adata.obs.columns:
        raise KeyError(f"{celltype_col} not found in adata.obs")
    if participant_col not in adata.obs.columns:
        raise KeyError(f"{participant_col} not found in adata.obs")
    if group_col not in adata.obs.columns:
        raise KeyError(f"{group_col} not found in adata.obs")

    adata_sub = adata[adata.obs[celltype_col].isin(celltypes)].copy()
    if adata_sub.n_obs == 0:
        raise ValueError("No cells found for the requested celltypes.")

    gene_name = resolve_gene_name(adata_sub, gene)
    from ._extract import extract_gene_vector

    expr = extract_gene_vector(adata_sub, gene_name, layer=layer)
    if log1p:
        expr = np.log1p(expr)

    df = pd.DataFrame({
        participant_col: adata_sub.obs[participant_col].values,
        "group": adata_sub.obs[group_col].values,
        "expr": expr,
    })
    df = df.dropna(subset=[participant_col, "group"])

    def _summarize(group_df: pd.DataFrame) -> pd.Series:
        vals = np.asarray(group_df["expr"].values, dtype=float)
        return pd.Series({
            "mean_expr": float(np.mean(vals)),
            "median_expr": float(np.median(vals)),
            "pct_expressing": float(np.mean(vals > expr_threshold) * 100.0),
            "n_cells": int(len(vals)),
        })

    df_patient = (
        df.groupby([participant_col, "group"], observed=True)
        .apply(_summarize)
        .reset_index()
    )
    df_patient = df_patient[df_patient["n_cells"] >= min_cells_per_patient].copy()

    g1 = df_patient[df_patient["group"] == group1]["mean_expr"]
    g2 = df_patient[df_patient["group"] == group2]["mean_expr"]

    if len(g1) < min_patients_per_group or len(g2) < min_patients_per_group:
        result = {
            "gene": gene_name,
            "celltypes": list(celltypes),
            "group1": group1,
            "group2": group2,
            "n_group1": int(len(g1)),
            "n_group2": int(len(g2)),
            "mean_group1": float(g1.mean()) if len(g1) else np.nan,
            "mean_group2": float(g2.mean()) if len(g2) else np.nan,
            "p_value": np.nan,
            "note": "Insufficient participants per group",
        }
        return result, df_patient

    _, p_val = mannwhitneyu(g1.values, g2.values, alternative="two-sided")
    result = {
        "gene": gene_name,
        "celltypes": list(celltypes),
        "group1": group1,
        "group2": group2,
        "n_group1": int(len(g1)),
        "n_group2": int(len(g2)),
        "mean_group1": float(g1.mean()),
        "mean_group2": float(g2.mean()),
        "delta": float(g1.mean() - g2.mean()),
        "p_value": float(p_val),
    }
    return result, df_patient
