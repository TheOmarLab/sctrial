from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData
from scipy.stats import mannwhitneyu
from scipy.stats import t as t_dist

from ..adata_tools import subset_cells
from ..design import TrialDesign
from ..utils import wild_cluster_bootstrap_t
from ._utils import apply_fdr, encode_visit, standardize_series
from .did import MIN_CLUSTERS_FOR_ROBUST_SE, AggregateFunc, AggregateMode, _ensure_paired


def _add_feature_columns(
    df: pd.DataFrame,
    ad: AnnData,
    features: Sequence[str],
    layer: str | None,
) -> pd.DataFrame:
    """Add feature columns to the DataFrame."""
    from .did import _add_feature_columns as _add_feature_columns_did

    df, _ = _add_feature_columns_did(df, ad, features, layer)
    return df


def resolve_gene_name(adata: AnnData, gene_query: str) -> str:
    """Resolve a gene name in var_names, case-insensitive if needed.

    Parameters
    ----------
    adata
        AnnData object.
    gene_query
        Gene name to resolve (case-insensitive).
    Returns
    -------
    str
        The resolved gene name (exact match or case-insensitive match).

    Raises
    ------
    ValueError
        If gene_query is not found in adata.var_names or if there are multiple case-insensitive matches.
    """
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
    covariates: list[str] | None,
) -> pd.DataFrame:
    """Prepare the data for between-arm comparison."""
    ad = subset_cells(adata, design, visit=visit, exclude_crossovers=False)
    obs = ad.obs.copy()
    obs["arm_bin"] = (obs[design.arm_col] == design.arm_treated).astype(int)

    cols = [design.participant_col, "arm_bin", design.arm_col]
    if covariates:
        for c in covariates:
            if c not in obs.columns:
                raise KeyError(f"Covariate '{c}' not found in adata.obs")
            cols.append(c)
    df = obs[cols].copy()

    df = _add_feature_columns(df, ad, features, layer)

    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, "arm_bin", design.arm_col]
        cov_agg: dict[str, str] = {}
        if covariates:
            for c in covariates:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cov_agg[c] = str(agg)
                else:
                    nunique = df.groupby(grp_cols, observed=True)[c].nunique()
                    if nunique.max() > 1:
                        raise ValueError(
                            f"Covariate '{c}' varies within participant at visit; "
                            "use numeric or constant covariates only."
                        )
                    cov_agg[c] = "first"
        df = (
            df.groupby(grp_cols, observed=True)
            .agg(
                {
                    **{f: agg for f in features},
                    **cov_agg,
                }
            )
            .reset_index()
        )

    return df


def _ols_between_arm(
    df_use: pd.DataFrame,
    feat: str,
    design: TrialDesign,
    standardize: bool,
    covariates: list[str] | None,
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
        Dictionary with keys: feature, beta_arm, se_arm, ci_lo_arm,
        ci_hi_arm, p_arm, n_units.
    """
    df_feat = df_use.copy().reset_index(drop=True)  # unique int index for .loc
    if standardize:
        y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
        if not ok:
            return {
                "feature": feat,
                "beta_arm": np.nan,
                "se_arm": np.nan,
                "ci_lo_arm": np.nan,
                "ci_hi_arm": np.nan,
                "p_arm": np.nan,
                "n_units": int(df_feat[design.participant_col].nunique()),
            }
        df_feat["outcome_std"] = y_std
    else:
        df_feat["outcome_std"] = df_feat[feat].astype(float)

    formula = "outcome_std ~ arm_bin"
    if covariates:
        formula += " + " + " + ".join(covariates)
    model = smf.ols(formula, data=df_feat)
    fit = model.fit()

    beta = float(fit.params.get("arm_bin", np.nan))
    se = float(fit.bse.get("arm_bin", np.nan))
    t_crit = float(t_dist.ppf(0.975, fit.df_resid))
    # Report effective participant count after model row drops
    model_row_idx = fit.model.data.row_labels
    n_units_eff = int(df_feat[design.participant_col].loc[model_row_idx].nunique())
    return {
        "feature": feat,
        "beta_arm": beta,
        "se_arm": se,
        "ci_lo_arm": beta - t_crit * se,
        "ci_hi_arm": beta + t_crit * se,
        "p_arm": float(fit.pvalues.get("arm_bin", np.nan)),
        "n_units": n_units_eff,
    }


def get_within_arm_aggregated_df(
    adata: AnnData,
    arm: str,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    layer: str | None = None,
    aggregate: AggregateMode = "participant_visit",
    agg: AggregateFunc = "mean",
    covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build aggregated DataFrame for within-arm comparison, for permutation tests.

    Returns (df_use, unit) where df_use has one row per participant-visit with
    feature values and visit_num. Permute visit_num within each participant
    and run OLS to get null betas.

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
    layer
        Layer to use for gene expression.
    aggregate
        Aggregation mode (see `did_table`).
    agg
        Aggregation function.
    covariates
        Optional covariate columns to include as fixed effects. Non-numeric
        covariates must be constant within participant-visit.

    Returns
    -------
    tuple[pd.DataFrame, str]
        Tuple containing the aggregated DataFrame and the unit column name.
    """
    ad = subset_cells(adata, design, arm=arm, exclude_crossovers=False)
    ad = ad[ad.obs[design.visit_col].isin(visits)].copy()
    obs = encode_visit(ad.obs.copy(), design.visit_col, visits)
    cols = [design.participant_col, design.visit_col, "visit_num"]
    if covariates:
        cols.extend(covariates)
    df = obs[cols].copy()
    df = _add_feature_columns(df, ad, features, layer)
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col]
        cov_agg: dict[str, str] = {}
        if covariates:
            for c in covariates:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cov_agg[c] = str(agg)
                else:
                    cov_agg[c] = "first"
        df_use = (
            df.groupby(grp_cols, observed=True)
            .agg({**{f: agg for f in features}, **cov_agg})
            .reset_index()
        )
    else:
        df_use = df.copy()
    df_use = _ensure_paired(
        df_use, unit=design.participant_col, time=design.visit_col, visits=visits
    )
    df_use = encode_visit(df_use, design.visit_col, visits)
    df_use = df_use.reset_index(drop=True)
    return df_use, design.participant_col


def within_arm_fit_beta(
    df: pd.DataFrame,
    feat: str,
    unit: str,
    *,
    standardize: bool = True,
) -> float:
    """Fit within-arm model and return beta_time. Used for permutation tests.

    Parameters
    ----------
    df
        DataFrame with feature values and visit_num.
    feat
        Name of the feature column to analyze.
    unit
        Name of the unit column.
    standardize
        Whether to z-score the outcome variable.

    Returns
    -------
    float
        The beta_time.
    """
    df_feat = df[[unit, "visit_num", feat]].dropna().copy()
    if len(df_feat) < 4:
        return np.nan
    if standardize:
        y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
        if not ok:
            return np.nan
        df_feat["outcome_std"] = y_std
    else:
        df_feat["outcome_std"] = df_feat[feat].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(f"outcome_std ~ visit_num + C({unit})", data=df_feat)
        clusters = np.asarray(df_feat[unit].to_numpy())
        try:
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
        except Exception:
            fit = model.fit()
    return float(fit.params.get("visit_num", np.nan))


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
    covariates: list[str] | None = None,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
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
    covariates
        Optional covariate columns to include as fixed effects. Non-numeric
        covariates must be constant within participant-visit.
    use_bootstrap
        Whether to use wild cluster bootstrap for p-values and CIs.
        Recommended when the number of participants (clusters) is small
        (< 10).  When enabled, bootstrap p-values replace the analytical
        p-values as the primary ``p_time`` column.
    n_boot
        Number of bootstrap iterations (999 or 1999 for publication).
    seed
        Random seed for bootstrap reproducibility.

    Returns
    -------
    pd.DataFrame
        Table with columns:

        - **feature** : Name of the feature.
        - **beta_time** : Estimated pre→post change (visit coefficient).
        - **se_time** : Cluster-robust standard error of the time coefficient.
        - **ci_lo_time** / **ci_hi_time** : 95 % confidence interval bounds
          (``beta ± t_crit × se`` with residual df from the OLS fit).
        - **p_time** : P-value for the time coefficient
          (cluster-robust Wald test; bootstrap if ``use_bootstrap=True``).
        - **n_units** : Number of unique participants.
        - **FDR_time** : Benjamini–Hochberg corrected p-value.

        When ``use_bootstrap=True``, additional columns are included:

        - **p_time_boot** : Bootstrap p-value.
        - **se_time_boot** : Bootstrap SE from coefficient distribution.
        - **ci_lo_boot** / **ci_hi_boot** : Bootstrap-t 95 % CI.
    """
    # Subset to arm and visits
    ad = subset_cells(adata, design, arm=arm, exclude_crossovers=False)
    ad = ad[ad.obs[design.visit_col].isin(visits)].copy()

    obs = encode_visit(ad.obs.copy(), design.visit_col, visits)

    # build dataframe
    cols = [design.participant_col, design.visit_col, "visit_num"]
    if covariates:
        for c in covariates:
            if c not in obs.columns:
                raise KeyError(f"Covariate '{c}' not found in adata.obs")
            cols.append(c)
    df = obs[cols].copy()

    df = _add_feature_columns(df, ad, features, layer)

    # Aggregate
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col]
        cov_agg: dict[str, str] = {}
        if covariates:
            for c in covariates:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cov_agg[c] = str(agg)
                else:
                    nunique = df.groupby(grp_cols, observed=True)[c].nunique()
                    if nunique.max() > 1:
                        raise ValueError(
                            f"Covariate '{c}' varies within participant-visit; "
                            "use numeric or constant covariates only."
                        )
                    cov_agg[c] = "first"
        df_use = (
            df.groupby(grp_cols, observed=True)
            .agg(
                {
                    **{f: agg for f in features},
                    **cov_agg,
                }
            )
            .reset_index()
        )
        unit = design.participant_col
    else:
        df_use = df.copy()
        unit = design.participant_col

    df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
    df_use = encode_visit(df_use, design.visit_col, visits)
    df_use = df_use.reset_index(drop=True)  # ensure unique integer index for .loc

    rows = []
    for feat in features:
        # Create a fresh copy for each feature to avoid cross-contamination
        df_feat = df_use.copy()

        if standardize:
            y_std, ok = standardize_series(df_feat, feat, min_std=1e-12)
            if not ok:
                # Skip features with near-zero variance
                rows.append(
                    {
                        "feature": feat,
                        "beta_time": np.nan,
                        "se_time": np.nan,
                        "ci_lo_time": np.nan,
                        "ci_hi_time": np.nan,
                        "p_time": np.nan,
                        "n_units": int(df_feat[unit].nunique()),
                    }
                )
                continue
            df_feat["outcome_std"] = y_std
        else:
            df_feat["outcome_std"] = df_feat[feat].astype(float)

        formula = f"outcome_std ~ visit_num + C({unit})"
        if covariates:
            formula += " + " + " + ".join(covariates)
        model = smf.ols(formula, data=df_feat)

        # Align cluster vector with fitted model rows: statsmodels may
        # drop rows with missing values during formula parsing, so
        # df_feat can be longer than model.exog.
        model_row_idx = model.data.row_labels
        clusters_aligned = np.asarray(df_feat[unit].loc[model_row_idx].to_numpy())

        n_units_feat = len(np.unique(clusters_aligned))
        if n_units_feat < MIN_CLUSTERS_FOR_ROBUST_SE:
            warnings.warn(
                f"Only {n_units_feat} clusters (participants) available. Cluster-robust "
                f"standard errors are unreliable with fewer than {MIN_CLUSTERS_FOR_ROBUST_SE} "
                f"clusters."
                + (
                    " Consider using use_bootstrap=True for more reliable p-values."
                    if not use_bootstrap
                    else ""
                ),
                UserWarning,
                stacklevel=2,
            )
        # A perfectly saturated paired design (n participants x 2 visits, with
        # participant fixed effects) leaves zero residual degrees of freedom, so
        # statsmodels' cluster-robust covariance raises ZeroDivisionError on its
        # (nobs-1)/(nobs-k_params) small-sample correction before the degenerate-SE
        # fallback below can run. Catch it and fall back to the nonrobust fit,
        # which is valid here because participant FE already absorb the
        # within-cluster correlation. Without this the exception propagates and
        # aborts the whole caller (it silently killed all five AML GSEA libraries).
        effective_cov_type = "cluster"
        try:
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": clusters_aligned})
        except (ZeroDivisionError, ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                f"Cluster-robust fit failed for feature '{feat}' with "
                f"{n_units_feat} clusters ({type(exc).__name__}: {exc}); the model "
                "is likely saturated (no residual degrees of freedom). Falling "
                "back to nonrobust (homoskedastic) SE.",
                UserWarning,
                stacklevel=2,
            )
            fit = model.fit()
            effective_cov_type = "nonrobust"

        beta = float(fit.params.get("visit_num", np.nan))
        se = float(fit.bse.get("visit_num", np.nan))
        p_val = float(fit.pvalues.get("visit_num", np.nan))

        # Fallback: if cluster-robust SE is degenerate (NaN), re-fit
        # with nonrobust SE.  Participant FE already absorbs within-
        # cluster correlation so homoskedastic SE is valid.
        if not np.isfinite(se) or not np.isfinite(p_val):
            warnings.warn(
                f"Cluster-robust SE is degenerate (NaN) for feature '{feat}' "
                f"with {n_units_feat} clusters. Falling back to nonrobust "
                f"(homoskedastic) SE.",
                UserWarning,
                stacklevel=2,
            )
            fit = model.fit()  # nonrobust
            se = float(fit.bse.get("visit_num", np.nan))
            p_val = float(fit.pvalues.get("visit_num", np.nan))
            effective_cov_type = "nonrobust"

        # Use robust conf_int() to ensure CI is consistent with the
        # cluster-robust SE / p-value (avoids df_resid mismatch).
        ci_bounds = fit.conf_int(alpha=0.05)
        if "visit_num" in ci_bounds.index:
            ci_lo = float(ci_bounds.loc["visit_num", 0])
            ci_hi = float(ci_bounds.loc["visit_num", 1])
        else:
            ci_lo = np.nan
            ci_hi = np.nan

        row_dict: dict[str, object] = {
            "feature": feat,
            "beta_time": beta,
            "se_time": se,
            "ci_lo_time": ci_lo,
            "ci_hi_time": ci_hi,
            "p_time": p_val,
            "n_units": n_units_feat,  # post-row-drop cluster count
            "cov_type_used": effective_cov_type,
        }

        # Wild cluster bootstrap
        if use_bootstrap and "visit_num" in fit.params:
            boot_res = wild_cluster_bootstrap_t(
                fit,
                X=fit.model.exog,
                clusters=clusters_aligned,  # already aligned above
                term_name="visit_num",
                B=n_boot,
                seed=seed,
                cov_type=effective_cov_type,
            )
            row_dict["p_time_boot"] = boot_res.p_boot
            row_dict["se_time_boot"] = boot_res.se_boot
            row_dict["ci_lo_boot"] = boot_res.ci_lo
            row_dict["ci_hi_boot"] = boot_res.ci_hi
            # Use bootstrap p-value as primary when available;
            # preserve analytical p_time if bootstrap returned NaN
            if np.isfinite(boot_res.p_boot):
                row_dict["p_time"] = boot_res.p_boot

        rows.append(row_dict)

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
    covariates: list[str] | None = None,
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
    covariates
        Optional covariate columns to include as fixed effects for OLS.

    Returns
    -------
    pd.DataFrame
        Table with columns:

        - **feature** : Name of the feature.
        - **beta_arm** : Effect size (treated − control difference in means).
        - **se_arm** : Standard error of the arm coefficient. For OLS this is
          the analytical SE from the model fit; for Wilcoxon it is the pooled
          SE of the difference in means (√(s₁²/n₁ + s₂²/n₂)).
        - **ci_lo_arm** / **ci_hi_arm** : 95 % confidence interval bounds.
          For OLS: ``beta ± t_crit × se`` with residual df. For Wilcoxon:
          ``beta ± t_crit × se`` with Welch–Satterthwaite df.
        - **p_arm** : P-value for the between-arm comparison.
        - **FDR_arm** : Benjamini–Hochberg corrected p-value.
        - **n_units** : Number of unique participants.
    """
    df_use = _prepare_between_arm_df(
        adata=adata,
        features=features,
        design=design,
        visit=visit,
        aggregate=aggregate,
        layer=layer,
        agg=agg,
        covariates=covariates,
    )

    rows = []
    for feat in features:
        if method == "ols":
            rows.append(_ols_between_arm(df_use, feat, design, standardize, covariates))
        elif method == "wilcoxon":
            g1 = np.asarray(df_use[df_use["arm_bin"] == 1][feat].values, dtype=float)
            g2 = np.asarray(df_use[df_use["arm_bin"] == 0][feat].values, dtype=float)

            if len(g1) > 0 and len(g2) > 0:
                stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
                beta = float(np.mean(g1) - np.mean(g2))
                n1, n2 = len(g1), len(g2)

                if n1 >= 2 and n2 >= 2:
                    # Pooled SE for the difference in means
                    v1 = np.var(g1, ddof=1) / n1
                    v2 = np.var(g2, ddof=1) / n2
                    se = float(np.sqrt(v1 + v2))
                    # Welch-Satterthwaite df for CI
                    denom = v1**2 / (n1 - 1) + v2**2 / (n2 - 1)
                    df_ws = float((v1 + v2) ** 2 / denom) if denom > 0 else max(n1 + n2 - 2, 1)
                    t_crit = float(t_dist.ppf(0.975, df_ws))
                    ci_lo = beta - t_crit * se
                    ci_hi = beta + t_crit * se
                else:
                    # Singleton arm: variance undefined with ddof=1
                    se = np.nan
                    ci_lo = np.nan
                    ci_hi = np.nan

                rows.append(
                    {
                        "feature": feat,
                        "beta_arm": beta,
                        "se_arm": se,
                        "ci_lo_arm": ci_lo,
                        "ci_hi_arm": ci_hi,
                        "p_arm": float(p_val),
                        "n_units": int(df_use[design.participant_col].nunique()),
                    }
                )
            else:
                warnings.warn(
                    f"Between-arm comparison skipped for feature '{feat}': "
                    f"empty group detected (n_treated={len(g1)}, n_control={len(g2)}).",
                    UserWarning,
                    stacklevel=2,
                )
                rows.append(
                    {
                        "feature": feat,
                        "beta_arm": np.nan,
                        "se_arm": np.nan,
                        "ci_lo_arm": np.nan,
                        "ci_hi_arm": np.nan,
                        "p_arm": np.nan,
                        "n_units": int(df_use[design.participant_col].nunique()),
                    }
                )

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

    Parameters
    ----------
    adata
        AnnData object.
    gene
        Gene name.
    celltypes
        Cell types to analyze.
    group_col
        Column name in adata.obs to use for grouping.
    group1
        First group to compare.
    group2
        Second group to compare.
    participant_col
        Column name in adata.obs to use for participant IDs.
    celltype_col
        Column name in adata.obs to use for cell types.
    layer
        Layer name in adata.layers to use for expression data.
    log1p
        Whether to log1p the expression data.
    expr_threshold
        Expression threshold to use for calculating the percentage of expressing cells. This is the minimum expression level to be considered expressing.
    min_cells_per_patient
        Minimum number of cells per participant to include in the analysis.
    min_patients_per_group
        Minimum number of participants per group to include in the analysis.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        A tuple containing a dictionary with the results and a DataFrame with the participant-level summaries
        - The dictionary contains the results of the comparison.
        - The DataFrame contains the participant-level summaries.
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

    df = pd.DataFrame(
        {
            participant_col: adata_sub.obs[participant_col].values,
            "group": adata_sub.obs[group_col].values,
            "expr": expr,
        }
    )
    df = df.dropna(subset=[participant_col, "group"])

    def _summarize(group_df: pd.DataFrame) -> pd.Series:
        vals = np.asarray(group_df["expr"].values, dtype=float)
        return pd.Series(
            {
                "mean_expr": float(np.mean(vals)),
                "median_expr": float(np.median(vals)),
                "pct_expressing": float(np.mean(vals > expr_threshold) * 100.0),
                "n_cells": int(len(vals)),
            }
        )

    df_patient = (
        df.groupby([participant_col, "group"], observed=True).apply(_summarize).reset_index()
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
