from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from anndata import AnnData

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

from ..adata_tools import subset_primary
from ..design import TrialDesign
from ..utils import wild_cluster_bootstrap_t
from ._utils import apply_fdr, encode_visit, standardize_series

# Minimum number of clusters for reliable cluster-robust standard errors
# Cameron & Miller (2015) recommend 42+, with 10 as absolute minimum
MIN_CLUSTERS_FOR_ROBUST_SE = 10


@dataclass(frozen=True)
class DiDConfig:
    """Configuration for DiD analysis.

    Use this dataclass to bundle common DiD settings and pass to did_table.
    """

    aggregate: AggregateMode = "participant_visit"
    """Aggregation mode for features (cell-level or participant-level)."""
    layer: str | None = None
    """Expression layer to use for genes (None uses adata.X)."""
    standardize: bool = True
    """Whether to z-score outcomes before model fitting."""
    agg: AggregateFunc = "mean"
    """Aggregation function applied after grouping."""
    covariates: list[str] | None = None
    """Optional covariate columns to include as fixed effects."""
    use_bootstrap: bool = False
    """If True, use wild cluster bootstrap for p-values."""
    n_boot: int = 999
    """Number of bootstrap permutations."""
    seed: int = 42
    """Random seed for bootstrap reproducibility."""
    exclude_crossovers: bool = True
    """Whether to drop crossover cells if crossover_col is set."""


def _validate_did_fit_inputs(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for did_fit: {missing}")


def _standardize_outcome(
    tmp: pd.DataFrame,
    y: str,
    standardize: bool,
    outcome_col: str = "outcome_std",
) -> pd.DataFrame | None:
    if not standardize:
        tmp[outcome_col] = tmp[y].astype(float)
        return tmp
    y_std, ok = standardize_series(tmp, y, min_std=1e-8)
    if not ok:
        return None
    tmp[outcome_col] = y_std
    return tmp


def _build_did_formula(
    time: str,
    arm_bin: str,
    unit: str,
    covariates: list[str] | None,
    outcome_col: str = "outcome_std",
) -> str:
    formula = f"{outcome_col} ~ {time} + {time}:{arm_bin} + C({unit})"
    if covariates:
        formula += " + " + " + ".join(covariates)
    return formula


def _prepare_did_obs(
    adata: AnnData,
    design: TrialDesign,
    visits: tuple[str, str],
    celltype: str | None,
    exclude_crossovers: bool,
) -> tuple[AnnData, pd.DataFrame]:
    ad = subset_primary(adata, design, visits=visits, exclude_crossovers=exclude_crossovers)
    if celltype is not None and design.celltype_col:
        ad = ad[ad.obs[design.celltype_col] == celltype].copy()
    obs = encode_visit(ad.obs.copy(), design.visit_col, visits)
    obs["arm_bin"] = (obs[design.arm_col] == design.arm_treated).astype(int)
    return ad, obs


def _add_feature_columns(
    df: pd.DataFrame,
    ad: AnnData,
    features: Sequence[str],
    layer: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    # features can be genes (in var_names) or obs columns
    # Optimization: Extract all genes from X/layer at once if they are in var_names
    genes_to_extract = [f for f in features if f in ad.var_names and f not in ad.obs.columns]
    feature_data: dict[str, np.ndarray] = {}
    final_features: list[str] = []
    missing: list[str] = []

    if genes_to_extract:
        ad_sub = ad[:, genes_to_extract]
        X = ad_sub.layers[layer] if layer is not None else ad_sub.X
        if sp.issparse(X):
            if isinstance(X, sp.coo_matrix):
                X = X.tocsr()
            X = X.toarray()
        else:
            X = np.asarray(X)
        for i, g in enumerate(genes_to_extract):
            feature_data[g] = X[:, i]

    for feat in features:
        if feat in ad.obs.columns:
            feature_data[feat] = ad.obs[feat].values
            final_features.append(feat)
        elif feat in genes_to_extract:
            final_features.append(feat)
        else:
            missing.append(feat)

    if feature_data:
        df = pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)

    if missing:
        shown = missing[:5]
        suffix = f" ({len(missing)} total)" if len(missing) > 5 else ""
        raise KeyError(f"Features not found in obs or var_names: {shown}{suffix}")
    if not final_features:
        raise ValueError("No numeric features found to analyze.")
    return df, final_features


def _aggregate_for_did(
    df: pd.DataFrame,
    final_features: list[str],
    design: TrialDesign,
    visits: tuple[str, str],
    aggregate: AggregateMode,
    agg: AggregateFunc,
    covariates: list[str] | None,
) -> tuple[pd.DataFrame, str, str, str]:
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col, design.arm_col]
        df["n_cells"] = 1
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

        df_use = df.groupby(grp_cols, observed=True).agg({
            **{f: agg for f in final_features},
            **cov_agg,
            "n_cells": "sum",
            "arm_bin": "first",
        }).reset_index()
        unit = design.participant_col
        time = "visit_num"
        arm_bin = "arm_bin"
        df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
        df_use = encode_visit(df_use, design.visit_col, visits)
        return df_use, unit, time, arm_bin

    if aggregate == "participant_visit_celltype":
        if design.celltype_col is None:
            raise ValueError("celltype_col is None; cannot use participant_visit_celltype")
        grp_cols = [design.participant_col, design.visit_col, design.arm_col, design.celltype_col]
        df["n_cells"] = 1
        cov_agg_ct: dict[str, str] = {}
        if covariates:
            for c in covariates:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cov_agg_ct[c] = str(agg)
                else:
                    nunique = df.groupby(grp_cols, observed=True)[c].nunique()
                    if nunique.max() > 1:
                        raise ValueError(
                            f"Covariate '{c}' varies within participant-visit-celltype; "
                            "use numeric or constant covariates only."
                        )
                    cov_agg_ct[c] = "first"

        df_use = df.groupby(grp_cols, observed=True).agg({
            **{f: agg for f in final_features},
            **cov_agg_ct,
            "n_cells": "sum",
            "arm_bin": "first",
        }).reset_index()
        unit = design.participant_col
        time = "visit_num"
        arm_bin = "arm_bin"
        df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
        df_use = encode_visit(df_use, design.visit_col, visits)
        return df_use, unit, time, arm_bin

    # cell-level
    df_use = df.copy()
    unit = design.participant_col
    time = "visit_num"
    arm_bin = "arm_bin"
    df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
    df_use = encode_visit(df_use, design.visit_col, visits)
    return df_use, unit, time, arm_bin

AggregateMode = Literal["cell", "participant_visit", "participant_visit_celltype"]
"""Supported aggregation modes for DiD analysis."""

AggregateFunc = Literal["mean", "median", "pct_pos"]
"""Supported aggregation functions for grouped feature summaries."""


class DidFitResult(TypedDict):
    """Structured return for did_fit."""

    beta_DiD: float
    se_DiD: float
    p_DiD: float
    beta_time: float
    p_time: float
    n_units: int
    resid_sd: NotRequired[float]
    p_DiD_boot: NotRequired[float]
    se_DiD_boot: NotRequired[float]
    ci_lo_boot: NotRequired[float]
    ci_hi_boot: NotRequired[float]
    cov_type_used: NotRequired[str]

def _ensure_paired(df: pd.DataFrame, unit: str, time: str, visits: tuple[str,str]) -> pd.DataFrame:
    wide = df.groupby([unit, time], observed=True).size().unstack(fill_value=0)
    keep = wide[(wide.get(visits[0], 0) > 0) & (wide.get(visits[1], 0) > 0)].index
    return df[df[unit].isin(keep)].copy()

def did_fit(
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    arm_bin: str,
    covariates: list[str] | None = None,
    cov_type: str = "cluster",
    standardize: bool = True,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
) -> DidFitResult:
    """Fit fixed-effects Difference-in-Differences (DiD) model.

    Mathematical Model
    ------------------
    The DiD model with participant fixed effects:

    .. math::

        Y_{it} = \\alpha_i + \\beta_1 \\cdot \\text{Post}_t + \\beta_2 \\cdot (\\text{Treat}_i \\times \\text{Post}_t) + \\epsilon_{it}

    where:
        - :math:`Y_{it}`: outcome for participant i at time t
        - :math:`\\alpha_i`: participant-specific intercept (fixed effect)
        - :math:`\\text{Post}_t`: indicator for follow-up visit (0=baseline, 1=followup)
        - :math:`\\text{Treat}_i`: treatment arm indicator (0=control, 1=treated)
        - :math:`\\beta_2`: **DiD coefficient** (the causal estimand of interest)
        - :math:`\\epsilon_{it}`: residual error

    Null Hypothesis
    ---------------
    H₀: β₂ = 0 (no differential treatment effect over time)
    H₁: β₂ ≠ 0 (treatment causes different change than control)

    The DiD estimator:

    .. math::

        \\hat{\\beta}_2 = (\\bar{Y}_{T,post} - \\bar{Y}_{T,pre}) - (\\bar{Y}_{C,post} - \\bar{Y}_{C,pre})

    Statistical Assumptions
    -----------------------
    - **Parallel trends**: In absence of treatment, both groups would follow
      same trajectory. Cannot be tested directly but can check pre-trends.
    - **No anticipation**: Treatment effect only after treatment starts.
    - **SUTVA**: No spillover between participants.
    - Requires at least 4 unique units (participants) to estimate fixed effects.
      Returns NaN for all estimates if n_units < 4.
    - Features with near-zero variance (std < 1e-8) return NaN.
    - Cluster-robust standard errors account for within-participant correlation.
    - If `n_cells` column is present, Weighted Least Squares (WLS) is used.
      Weights are proportional to n_cells (inverse-variance weighting for
      participant-level means, since Var(mean) = σ²/n).

    Parameters
    ----------
    df : DataFrame
        Long-format data with columns for unit, time, arm_bin, y, and covariates.
    y : str
        Name of the outcome column.
    unit : str
        Name of the participant/unit column.
    time : str
        Name of the time variable (numeric 0/1).
    arm_bin : str
        Name of the treatment indicator column (0/1).
    covariates : list of str, optional
        Additional covariate columns to include as fixed effects.
    cov_type : str
        Covariance type for standard errors ('cluster' recommended).
    standardize : bool
        If True, z-score the outcome before fitting (recommended for
        interpretable effect sizes).
    use_bootstrap : bool
        If True, use Wild Cluster Bootstrap for p-values (recommended
        when n_participants < 15).
    n_boot : int
        Number of bootstrap iterations (999 or 1999 for publication).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DidFitResult
        Keys:
        - beta_DiD: DiD coefficient (β₂)
        - se_DiD: Analytical standard error of β₂
        - p_DiD: P-value for H₀: β₂ = 0 (bootstrap if ``use_bootstrap=True``)
        - beta_time: Main time effect (β₁)
        - p_time: P-value for time effect
        - n_units: Number of participants used
        - cov_type_used: Covariance type used (``"cluster"`` or ``"nonrobust"``)
        - p_DiD_boot: Bootstrap p-value (only if ``use_bootstrap=True``)
        - se_DiD_boot: Bootstrap SE from coefficient distribution
          (only if ``use_bootstrap=True``)
        - ci_lo_boot: Lower 95% bootstrap-t CI
          (only if ``use_bootstrap=True``)
        - ci_hi_boot: Upper 95% bootstrap-t CI
          (only if ``use_bootstrap=True``)
    """
    if df is None or df.empty:
        raise ValueError("df must be a non-empty DataFrame.")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    cols = [unit, time, arm_bin, y]
    if covariates:
        cols.extend(covariates)
    # Include n_cells for WLS weighting if available
    if "n_cells" in df.columns:
        cols.append("n_cells")

    _validate_did_fit_inputs(df, cols)

    tmp = df[cols].dropna().copy()
    n_units = tmp[unit].nunique()
    if n_units < 4:
        return {
            "beta_DiD": np.nan,
            "se_DiD": np.nan,
            "p_DiD": np.nan,
            "beta_time": np.nan,
            "p_time": np.nan,
            "n_units": n_units,
        }

    # time is assumed numeric 0/1 already
    tmp_opt = _standardize_outcome(tmp, y, standardize, outcome_col="outcome_std")
    if tmp_opt is None:
        return {
            "beta_DiD": np.nan,
            "se_DiD": np.nan,
            "p_DiD": np.nan,
            "beta_time": np.nan,
            "p_time": np.nan,
            "n_units": n_units,
        }
    tmp = tmp_opt
    if tmp is None:
        raise ValueError("Outcome standardization failed unexpectedly.")

    formula = _build_did_formula(time, arm_bin, unit, covariates, outcome_col="outcome_std")

    # Inverse-variance weighting for pre-aggregated means:
    # Var(mean_i) = sigma^2 / n_i, so weight_i = n_i (proportional to 1/Var)
    weights = None
    if "n_cells" in tmp.columns:
        w = tmp["n_cells"].values.astype(float)
        if np.all(np.isfinite(w)) and np.all(w > 0):
            weights = w

    if weights is not None:
        model = smf.wls(formula, data=tmp, weights=weights)
    else:
        model = smf.ols(formula, data=tmp)

    # Warn if using cluster-robust SE with few clusters (unreliable inference)
    if cov_type == "cluster" and n_units < MIN_CLUSTERS_FOR_ROBUST_SE:
        warnings.warn(
            f"Only {n_units} clusters (participants) available. Cluster-robust standard "
            f"errors are unreliable with fewer than {MIN_CLUSTERS_FOR_ROBUST_SE} clusters. "
            f"Consider using use_bootstrap=True for more reliable p-values.",
            UserWarning,
            stacklevel=2,
        )

    fit = model.fit(cov_type=cov_type, cov_kwds={"groups": tmp[unit]} if cov_type == "cluster" else None)
    term = f"{time}:{arm_bin}"

    se_did = float(fit.bse.get(term, np.nan))
    p_did = float(fit.pvalues.get(term, np.nan))

    # Fallback: if cluster-robust SE is NaN (degenerate, e.g. only 2 obs per
    # cluster after participant_visit aggregation), re-fit with nonrobust SE.
    # The participant FE already absorbs within-participant correlation, so
    # homoskedastic SE is a valid (though conservative) alternative.
    #
    # Justification for nonrobust bootstrap in fallback mode:
    # When cluster-robust SE is degenerate (typically because each cluster has
    # only 2 observations after participant_visit aggregation), the wild cluster
    # bootstrap with nonrobust covariance is methodologically valid because:
    # (a) participant fixed effects already absorb within-cluster correlation,
    # (b) with only 2 obs per cluster, heteroskedasticity cannot be estimated
    #     within clusters anyway, and
    # (c) the Rademacher sign-flip at the cluster level still provides valid
    #     finite-sample inference (Webb 2023, J. Econometrics).
    effective_cov_type = cov_type
    if cov_type == "cluster" and (not np.isfinite(se_did) or not np.isfinite(p_did)):
        warnings.warn(
            f"Cluster-robust SE is degenerate (NaN) for feature '{y}' with "
            f"{int(tmp[unit].nunique())} clusters. Falling back to nonrobust "
            f"(homoskedastic) SE. This typically occurs when each cluster has "
            f"very few observations (e.g. participant_visit aggregation with "
            f"2 visits). Participant fixed effects still absorb within-cluster "
            f"correlation. Set use_bootstrap=True for more reliable p-values.",
            UserWarning,
            stacklevel=2,
        )
        fit = model.fit()  # nonrobust
        se_did = float(fit.bse.get(term, np.nan))
        p_did = float(fit.pvalues.get(term, np.nan))
        effective_cov_type = "nonrobust"

    res = {
        "beta_DiD": float(fit.params.get(term, np.nan)),
        "se_DiD": se_did,
        "p_DiD": p_did,
        "beta_time": float(fit.params.get(time, np.nan)),
        "p_time": float(fit.pvalues.get(time, np.nan)),
        "n_units": int(tmp[unit].nunique()),
        "resid_sd": float(np.sqrt(fit.scale)),
        "cov_type_used": effective_cov_type,
    }

    if use_bootstrap and term in fit.params:
        boot_res = wild_cluster_bootstrap_t(
            fit,
            X=fit.model.exog,
            clusters=np.asarray(tmp[unit].to_numpy()),
            term_name=term,
            B=n_boot,
            seed=seed,
            cov_type=effective_cov_type,
        )
        res["p_DiD_boot"] = boot_res.p_boot
        res["se_DiD_boot"] = boot_res.se_boot
        res["ci_lo_boot"] = boot_res.ci_lo
        res["ci_hi_boot"] = boot_res.ci_hi
        # Use bootstrap p-value as primary if requested
        res["p_DiD"] = boot_res.p_boot

    return cast(DidFitResult, res)

def did_table(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    exclude_crossovers: bool = True,
    celltype: str | None = None,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    standardize: bool = True,
    agg: AggregateFunc = "mean",
    covariates: list[str] | None = None,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
    config: DiDConfig | None = None,
) -> pd.DataFrame:
    """Run Difference-in-Differences (DiD) for a list of features.

    This function implements a fixed-effects DiD model to test for treatment-induced
    longitudinal changes. It is optimized for 'panels' of features (tens to hundreds),
    such as module scores or selected gene sets.

    Statistical Model:
        y ~ visit + visit:arm + covariates + C(participant)
        The 'visit:arm' interaction term coefficient (beta_DiD) is the primary estimand.

    Parameters
    ----------
    adata
        AnnData object containing expression data and metadata.
    features
        List of features to test. Can be gene names in `adata.var_names` or
        observation-level scores in `adata.obs.columns`.
    design
        A `TrialDesign` object specifying the metadata columns.
    visits
        A tuple of (baseline, followup) visit labels.
    exclude_crossovers
        If True, excludes observations where `design.crossover_col` is True.
        Recommended for primary randomized analysis.
    config
        Optional DiDConfig object. If provided, its values override the corresponding
        keyword arguments (aggregate, layer, standardize, agg, covariates, bootstrap,
        seed, and exclude_crossovers).
    celltype
        If provided, subsets the analysis to a specific cell type.
    aggregate
        Aggregation mode:

        - 'cell': Fit model on individual cells (not recommended for p-values,
          as it treats cells as independent).
        - 'participant_visit': Average features per participant-visit before fitting.
          This is the recommended approach for clinical inference.
        - 'participant_visit_celltype': Average per participant-visit-celltype.
    layer
        Layer to extract gene expression from. If None, uses `adata.X`.
    standardize
        If True, z-scores the outcome variable before fitting to provide
        standardized effect sizes.
    agg
        Aggregation function: 'mean', 'median', or 'pct_pos'.
    covariates
        List of additional columns in `adata.obs` to include as fixed effects
        in the model (e.g., ['age', 'sex', 'batch']).
        Covariates must be **numeric** or **constant within each participant-visit**
        group. Non-numeric covariates are aggregated with "first" and will raise
        an error if they vary within a participant-visit (or participant-visit-celltype).
    use_bootstrap
        If True, uses Wild Cluster Bootstrap to calculate p-values. Recommended
        for small sample sizes (e.g. < 15 participants per group).
    n_boot
        Number of bootstrap permutations.
    seed
        Random seed for bootstrap.

    Returns
    -------
    pd.DataFrame
        Table with one row per feature containing beta_DiD, p_DiD, and
        FDR-corrected significance.

    Examples
    --------
    >>> res = did_table(adata, features=["ms_OXPHOS"], design=design, visits=("V1", "V2"))
    >>> print(res[["feature", "beta_DiD", "p_DiD"]])
    """
    # subset and prepare
    if config is not None:
        aggregate = config.aggregate
        layer = config.layer
        standardize = config.standardize
        agg = config.agg
        covariates = config.covariates
        use_bootstrap = config.use_bootstrap
        n_boot = config.n_boot
        seed = config.seed
        exclude_crossovers = config.exclude_crossovers

    ad, obs = _prepare_did_obs(adata, design, visits, celltype, exclude_crossovers)

    # build dataframe with features and all possible grouping columns
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

    rows = []
    for feat in final_features:
        out = did_fit(
            df_use,
            y=feat,
            unit=unit,
            time=time,
            arm_bin=arm_bin,
            covariates=covariates,
            standardize=standardize,
            use_bootstrap=use_bootstrap,
            n_boot=n_boot,
            seed=seed
        )
        row = dict(out)
        row["feature"] = feat
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("p_DiD")
    res = apply_fdr(res, p_col="p_DiD", fdr_col="FDR_DiD")
    return res.reset_index(drop=True)


def did_table_parallel(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    exclude_crossovers: bool = True,
    celltype: str | None = None,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    standardize: bool = True,
    agg: AggregateFunc = "mean",
    covariates: list[str] | None = None,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
    n_jobs: int = -1,
    backend: str = "loky",
    batch_size: int | None = None,
    config: DiDConfig | None = None,
) -> pd.DataFrame:
    """Parallelized DiD across features using joblib.

    This mirrors `did_table` but parallelizes feature-level model fits. It is
    most useful for large feature panels (hundreds to thousands of genes).

    Parameters
    ----------
    n_jobs
        Number of parallel jobs (joblib convention). Use -1 for all cores.
    backend
        Joblib backend ("loky", "multiprocessing", or "threading").
    batch_size
        Optional joblib batch size. If None, joblib chooses.
    """
    if n_jobs == 1:
        return did_table(
            adata=adata,
            features=features,
            design=design,
            visits=visits,
            exclude_crossovers=exclude_crossovers,
            celltype=celltype,
            aggregate=aggregate,
            layer=layer,
            standardize=standardize,
            agg=agg,
            covariates=covariates,
            use_bootstrap=use_bootstrap,
            n_boot=n_boot,
            seed=seed,
            config=config,
        )

    try:
        from joblib import Parallel, delayed
    except ImportError as exc:  # pragma: no cover
        raise ImportError("joblib is required for did_table_parallel") from exc

    if config is not None:
        aggregate = config.aggregate
        layer = config.layer
        standardize = config.standardize
        agg = config.agg
        covariates = config.covariates
        use_bootstrap = config.use_bootstrap
        n_boot = config.n_boot
        seed = config.seed
        exclude_crossovers = config.exclude_crossovers

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

    # Generate independent seeds for each parallel feature using SeedSequence
    ss = np.random.SeedSequence(seed)
    feature_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(final_features))]

    def _fit_feature(idx: int, feat: str) -> dict:
        out = did_fit(
            df_use,
            y=feat,
            unit=unit,
            time=time,
            arm_bin=arm_bin,
            covariates=covariates,
            standardize=standardize,
            use_bootstrap=use_bootstrap,
            n_boot=n_boot,
            seed=feature_seeds[idx],
        )
        row = dict(out)
        row["feature"] = feat
        return row

    job_batch = "auto" if batch_size is None else batch_size
    rows = Parallel(n_jobs=n_jobs, backend=backend, batch_size=job_batch)(
        delayed(_fit_feature)(i, feat) for i, feat in enumerate(final_features)
    )

    res = pd.DataFrame(rows).sort_values("p_DiD")
    res = apply_fdr(res, p_col="p_DiD", fdr_col="FDR_DiD")
    return res.reset_index(drop=True)


def did_table_by_celltype(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    celltypes: Sequence[str] | None = None,
    exclude_crossovers: bool = True,
    aggregate: AggregateMode = "participant_visit",
    layer: str | None = None,
    standardize: bool = True,
    agg: AggregateFunc = "mean",
    covariates: list[str] | None = None,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """Run `did_table` stratified by cell type.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Genes or module scores to test.
    design
        A `TrialDesign` object.
    visits
        Two visit labels for comparison.
    celltypes
        Subset of cell types to analyze. If None, uses all in `design.celltype_col`.
    exclude_crossovers
        Exclude cells marked as crossovers.
    aggregate
        Level of aggregation.
    layer
        Expression layer.
    standardize
        Standardize expression to unit variance.
    agg
        Aggregation function.
    covariates
        Obs columns to include as covariates.
    use_bootstrap
        Use Wild Cluster Bootstrap for p-values.
    n_boot
        Number of bootstrap iterations.
    seed
        Random seed.

    Returns
    -------
    pd.DataFrame
        Table with DiD results for each gene and cell type.
    """
    if design.celltype_col is None:
        raise ValueError("design.celltype_col must be set for stratified analysis.")

    if celltypes is None:
        celltypes = sorted(adata.obs[design.celltype_col].dropna().unique())

    all_res = []
    for ct in celltypes:
        try:
            res_ct = did_table(
                adata,
                features=features,
                design=design,
                visits=visits,
                celltype=ct,
                exclude_crossovers=exclude_crossovers,
                aggregate=aggregate,
                layer=layer,
                standardize=standardize,
                agg=agg,
                covariates=covariates,
                use_bootstrap=use_bootstrap,
                n_boot=n_boot,
                seed=seed,
            )
            res_ct["celltype"] = ct
            all_res.append(res_ct)
        except (ValueError, np.linalg.LinAlgError, KeyError) as e:
            # Common to fail if celltype has too few cells/participants
            warnings.warn(f"Failed DiD for celltype '{ct}': {e}")
            continue

    if not all_res:
        return pd.DataFrame()

    full_res = pd.concat(all_res, ignore_index=True)

    # Recalculate FDR across all tests
    full_res = apply_fdr(full_res, p_col="p_DiD", fdr_col="FDR_DiD_stratified")

    return full_res
