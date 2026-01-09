from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from ..adata_tools import subset_primary
from ..design import TrialDesign
from ..utils import wild_cluster_bootstrap_t
from ._utils import encode_visit, standardize_series

AggregateMode = Literal["cell", "participant_visit", "participant_visit_celltype"]
AggregateFunc = Literal["mean", "median", "pct_pos"]

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
) -> dict:
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
    dict
        Dictionary with keys:
        - beta_DiD: DiD coefficient (β₂)
        - se_DiD: Standard error of β₂
        - p_DiD: P-value for H₀: β₂ = 0
        - beta_time: Main time effect (β₁)
        - p_time: P-value for time effect
        - n_units: Number of participants used
    """
    cols = [unit, time, arm_bin, y]
    if covariates:
        cols.extend(covariates)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for did_fit: {missing}")
    # Include n_cells for WLS weighting if available
    if "n_cells" in df.columns:
        cols.append("n_cells")

    tmp = df[cols].dropna().copy()
    if tmp[unit].nunique() < 4:
        return {"beta_DiD": np.nan, "se_DiD": np.nan, "p_DiD": np.nan, "n_units": tmp[unit].nunique()}

    # time is assumed numeric 0/1 already
    if standardize:
        y_std, ok = standardize_series(tmp, y, min_std=1e-8)
        # Skip features with near-zero variance to avoid misleading standardized estimates
        if not ok:
            return {
                "beta_DiD": np.nan,
                "se_DiD": np.nan,
                "p_DiD": np.nan,
                "beta_time": np.nan,
                "p_time": np.nan,
                "n_units": tmp[unit].nunique(),
            }
        tmp["_y"] = y_std
    else:
        tmp["_y"] = tmp[y].astype(float)

    formula = f"_y ~ {time} + {time}:{arm_bin} + C({unit})"
    if covariates:
        formula += " + " + " + ".join(covariates)

    # detection of aggregation for weighting
    weights = None
    if "n_cells" in tmp.columns:
        weights = np.sqrt(tmp["n_cells"])

    if weights is not None:
        model = smf.wls(formula, data=tmp, weights=weights)
    else:
        model = smf.ols(formula, data=tmp)

    fit = model.fit(cov_type=cov_type, cov_kwds={"groups": tmp[unit]} if cov_type=="cluster" else None)
    term = f"{time}:{arm_bin}"

    res = {
        "beta_DiD": float(fit.params.get(term, np.nan)),
        "se_DiD": float(fit.bse.get(term, np.nan)),
        "p_DiD": float(fit.pvalues.get(term, np.nan)),
        "beta_time": float(fit.params.get(time, np.nan)),
        "p_time": float(fit.pvalues.get(time, np.nan)),
        "n_units": int(tmp[unit].nunique()),
    }

    if use_bootstrap and term in fit.params:
        p_boot = wild_cluster_bootstrap_t(
            fit,
            X=fit.model.exog,
            clusters=np.asarray(tmp[unit].to_numpy()),
            term_name=term,
            B=n_boot,
            seed=seed
        )
        res["p_DiD_boot"] = p_boot
        # Use bootstrap p-value as primary if requested
        res["p_DiD"] = p_boot

    return res

def did_table(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str,str],
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
    # subset
    ad = subset_primary(adata, design, visits=visits, exclude_crossovers=exclude_crossovers)
    if celltype is not None and design.celltype_col:
        ad = ad[ad.obs[design.celltype_col] == celltype].copy()

    obs = encode_visit(ad.obs.copy(), design.visit_col, visits)
    obs["arm_bin"] = (obs[design.arm_col] == design.arm_treated).astype(int)

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

    # features can be genes (in var_names) or obs columns
    # Optimization: Extract all genes from X/layer at once if they are in var_names
    genes_to_extract = [f for f in features if f in ad.var_names and f not in ad.obs.columns]

    # Build feature columns to add (avoid DataFrame fragmentation)
    feature_data = {}

    if genes_to_extract:
        # subset var to requested genes
        ad_sub = ad[:, genes_to_extract]
        X = ad_sub.layers[layer] if layer is not None else ad_sub.X
        if sp.issparse(X):
            if isinstance(X, sp.coo_matrix):
                X = X.tocsr()
            X = X.toarray()
        else:
            X = np.asarray(X)

        for i, gene in enumerate(genes_to_extract):
            feature_data[gene] = X[:, i]

    missing = []
    final_features = []
    for feat in features:
        if feat in ad.obs.columns:
            val = ad.obs[feat]
            if not pd.api.types.is_numeric_dtype(val):
                continue # skip non-numeric obs columns
            feature_data[feat] = val.values
            final_features.append(feat)
        elif feat in genes_to_extract:
            final_features.append(feat)
        else:
            missing.append(feat)

    # Add all feature columns at once to avoid DataFrame fragmentation
    if feature_data:
        df = pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)

    if missing:
        raise KeyError(f"Features not found in obs or var_names: {missing[:5]}")

    if not final_features:
        raise ValueError("No numeric features found to analyze.")

    # aggregate if requested
    if aggregate == "participant_visit":
        grp_cols = [design.participant_col, design.visit_col, design.arm_col]

        df["n_cells"] = 1
        agg_features = list(final_features)

        cov_agg: dict[str, str] = {}
        if covariates:
            for c in covariates:
                if pd.api.types.is_numeric_dtype(df[c]):
                    cov_agg[c] = str(agg)
                else:
                    # Non-numeric covariates must be constant within participant-visit
                    nunique = df.groupby(grp_cols, observed=True)[c].nunique()
                    if nunique.max() > 1:
                        raise ValueError(
                            f"Covariate '{c}' varies within participant-visit; "
                            "use numeric or constant covariates only."
                        )
                    cov_agg[c] = "first"

        df_use = df.groupby(grp_cols, observed=True).agg({
            **{f: agg for f in agg_features},
            **cov_agg,
            "n_cells": "sum",
            "arm_bin": "first"
        }).reset_index()

        unit = design.participant_col
        time = "visit_num"
        arm_bin = "arm_bin"
        df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
        df_use = encode_visit(df_use, design.visit_col, visits)
    elif aggregate == "participant_visit_celltype":
        if design.celltype_col is None:
            raise ValueError("celltype_col is None; cannot use participant_visit_celltype")
        grp_cols = [design.participant_col, design.visit_col, design.arm_col, design.celltype_col]

        df["n_cells"] = 1
        agg_features = list(final_features)

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
            **{f: agg for f in agg_features},
            **cov_agg_ct,
            "n_cells": "sum",
            "arm_bin": "first"
        }).reset_index()

        unit = design.participant_col
        time = "visit_num"
        arm_bin = "arm_bin"
        df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
        df_use = encode_visit(df_use, design.visit_col, visits)
    else:
        # cell-level
        df_use = df.copy()
        unit = design.participant_col
        time = "visit_num"
        arm_bin = "arm_bin"
        df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
        df_use = encode_visit(df_use, design.visit_col, visits)

    rows=[]
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
        out["feature"]=feat
        rows.append(out)
    res=pd.DataFrame(rows).sort_values("p_DiD")
    # FDR
    mask=res["p_DiD"].notna()
    res["FDR_DiD"]=np.nan
    if mask.sum()>0:
        res.loc[mask,"FDR_DiD"]=multipletests(res.loc[mask,"p_DiD"], method="fdr_bh")[1]
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
            import warnings
            warnings.warn(f"Failed DiD for celltype '{ct}': {e}")
            continue

    if not all_res:
        return pd.DataFrame()

    full_res = pd.concat(all_res, ignore_index=True)

    # Recalculate FDR across all tests if possible
    mask = full_res["p_DiD"].notna()
    if mask.sum() > 0:
        full_res["FDR_DiD_stratified"] = np.nan
        full_res.loc[mask, "FDR_DiD_stratified"] = multipletests(
            full_res.loc[mask, "p_DiD"], method="fdr_bh"
        )[1]

    return full_res
