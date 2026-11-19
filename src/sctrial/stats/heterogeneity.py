from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData

from ..design import TrialDesign
from ._utils import apply_fdr, drop_unused_levels, encode_visit, standardize_series
from .did import (
    MIN_CLUSTERS_FOR_ROBUST_SE,
    AggregateFunc,
    AggregateMode,
    _add_feature_columns,
    _ensure_paired,
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

    The model uses the participants observed at both visits. The biomarker is a
    participant attribute: a participant's value is the mean of ``biomarker_col``
    over its cells, and participants without a value are excluded. For each
    feature the interaction needs at least two participants with a value at both
    visits in each arm by biomarker group; when a group is smaller, the
    coefficient and p-value of that feature are NaN and a warning is raised.

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
        If None, numeric biomarkers are split at the median over the participants
        in the model. For binary biomarkers (0/1 or True/False), values are used
        directly.
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

    Examples
    --------
    >>> res = test_treatment_heterogeneity(
    ...     adata,
    ...     features=["sig_IFN_Response", "sig_Cytotoxicity"],
    ...     design=design,
    ...     visits=("Pre", "Post"),
    ...     biomarker_col="baseline_crp",
    ...     threshold=5.0,
    ... )
    >>> res[["feature", "beta_heterogeneity", "p_heterogeneity"]].head()
    """
    if biomarker_col not in adata.obs.columns:
        raise KeyError(f"biomarker_col '{biomarker_col}' not found in adata.obs")

    ad, obs = _prepare_did_obs(adata, design, visits, celltype=None, exclude_crossovers=True)
    unit = design.participant_col
    time = "visit_num"
    arm_bin = "arm_bin"

    raw = adata.obs.loc[obs.index, biomarker_col]
    if pd.api.types.is_bool_dtype(raw):
        values = raw.astype(float)
    else:
        values = pd.to_numeric(raw.astype(object), errors="coerce")
        n_coerced = int((values.isna() & raw.notna()).sum())
        if n_coerced:
            warnings.warn(
                f"biomarker_col '{biomarker_col}' has {n_coerced} non-numeric values, "
                "which are treated as missing.",
                UserWarning,
                stacklevel=2,
            )
    obs["_biomarker"] = values.to_numpy(dtype=float)

    # Add feature columns (genes or obs)
    obs_feat, final_features = _add_feature_columns(obs.copy(), ad, list(features), layer)

    if aggregate != "participant_visit":
        raise ValueError("test_treatment_heterogeneity supports participant_visit only.")

    grp_cols = [unit, design.visit_col, design.arm_col]
    use_cols = list(final_features) + [arm_bin]
    df_use = (
        obs_feat.groupby(grp_cols, observed=True)[use_cols].mean(numeric_only=True).reset_index()
    )
    df_use = _ensure_paired(df_use, unit=unit, time=design.visit_col, visits=visits)
    df_use[unit] = df_use[unit].astype(object)

    # The biomarker is a participant attribute: one value per participant, and the
    # median split is taken over the participants that enter the model.
    per_participant = obs.groupby(obs[unit].astype(object))["_biomarker"].mean()
    per_participant = per_participant.reindex(pd.unique(df_use[unit]))
    n_missing = int(per_participant.isna().sum())
    if n_missing:
        warnings.warn(
            f"{n_missing} of {len(per_participant)} paired participants have no value of "
            f"'{biomarker_col}' and are excluded.",
            UserWarning,
            stacklevel=2,
        )
    per_participant = per_participant.dropna()
    if set(per_participant.unique()) <= {0.0, 1.0}:
        high = per_participant.astype(int)
    else:
        if threshold is None:
            threshold = float(per_participant.median())
        high = (per_participant > threshold).astype(int)

    df_use = df_use[df_use[unit].isin(high.index)].copy()
    df_use["biomarker_high"] = df_use[unit].map(high).astype(int)
    df_use[arm_bin] = df_use[arm_bin].round().astype(int)
    df_use = encode_visit(df_use, design.visit_col, visits)
    obs_feat = df_use

    rows = []
    n_examined = 0
    not_estimable: dict[str, dict[str, int]] = {}
    for feat in final_features:
        tmp = obs_feat.copy()
        if tmp.columns.duplicated().any():
            tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()
        tmp = tmp.dropna(subset=[feat, "biomarker_high"])
        # A participant with a missing value at one visit has no change for this feature.
        visits_seen = tmp.groupby(unit, observed=True)[time].nunique()
        tmp = tmp[tmp[unit].isin(visits_seen.index[visits_seen >= 2])].copy()
        n_clusters = tmp[unit].nunique()
        if n_clusters < MIN_CLUSTERS_FOR_ROBUST_SE:
            warnings.warn(
                f"Feature '{feat}' skipped: only {n_clusters} clusters "
                f"(need >= {MIN_CLUSTERS_FOR_ROBUST_SE} for robust SE).",
                stacklevel=2,
            )
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

        df = drop_unused_levels(tmp.dropna(subset=["outcome_std"]).reset_index(drop=True))

        # The interaction contrasts the change over time across the four arm by biomarker
        # groups. A group with one participant fits that participant exactly and adds
        # nothing to the cluster-robust variance, so each group needs at least two of the
        # participants in this feature's model.
        group_sizes = df.drop_duplicates(unit).groupby([arm_bin, "biomarker_high"]).size()
        counts = {
            f"{arm_name}/{level_name}": int(group_sizes.get((a, b), 0))
            for a, arm_name in ((0, "control"), (1, "treated"))
            for b, level_name in ((0, "low"), (1, "high"))
        }
        n_examined += 1
        if min(counts.values()) < 2:
            not_estimable[feat] = counts
            rows.append(
                {
                    "feature": feat,
                    "beta_heterogeneity": np.nan,
                    "p_heterogeneity": np.nan,
                    "n_units": int(df[unit].nunique()),
                    "threshold": threshold,
                }
            )
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
        # Report effective participant count after model row drops
        model_row_idx = fit.model.data.row_labels
        n_units_eff = int(df[unit].loc[model_row_idx].nunique())
        rows.append(
            {
                "feature": feat,
                "beta_heterogeneity": float(fit.params.get(term, np.nan)),
                "p_heterogeneity": float(fit.pvalues.get(term, np.nan)),
                "n_units": n_units_eff,
                "threshold": threshold,
            }
        )

    if not_estimable:
        first, counts = next(iter(not_estimable.items()))
        warnings.warn(
            f"The arm by visit by biomarker interaction is not estimable for "
            f"{len(not_estimable)} of {n_examined} features: it needs at least "
            "2 paired participants in each arm by biomarker group, found "
            + ", ".join(f"{name}={n}" for name, n in counts.items())
            + f" for '{first}'.",
            UserWarning,
            stacklevel=2,
        )

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    if fdr in {"all", "feature"}:
        res = apply_fdr(res, p_col="p_heterogeneity", fdr_col="FDR_heterogeneity")
    return res
