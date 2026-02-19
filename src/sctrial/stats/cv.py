"""Cross-validation utilities for effect stability assessment.

This module provides leave-one-out (LOO) and k-fold cross-validation
for assessing the robustness of DiD estimates.

Why Cross-Validation for DiD?
-----------------------------
Cross-validation in the DiD context serves different purposes than in
predictive modeling:

1. **Influence diagnostics**: Identify participants with outsized influence
   on effect estimates (potential outliers or data quality issues).

2. **Effect stability**: Assess how robust the DiD estimate is to the
   exclusion of individual participants.

3. **Generalizability**: Estimate how well the effect might replicate
   in new samples (though true generalization requires new data).

Leave-One-Out (LOO)
-------------------
For each participant i:
    1. Fit DiD model excluding participant i
    2. Record beta_DiD^(-i)
    3. Compare to full-sample beta_DiD

Metrics:
- Influence: ``|beta_DiD - beta_DiD^(-i)| / SE(beta_DiD)``
- Cook's D analog for DiD

K-Fold Cross-Validation
-----------------------
1. Randomly partition participants into K folds
2. For each fold k:
   - Fit DiD on participants NOT in fold k
   - Record estimate
3. Report mean, SD, and CI of estimates

Interpretation Guidelines
-------------------------
- High LOO variance: Effect driven by few participants
- Consistently signed CV estimates: Robust effect
- Estimate changes sign across folds: Unreliable effect
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ..adata_tools import subset_primary
from ..design import TrialDesign
from .did import AggregateMode, did_table

__all__ = [
    "loo_cv_did",
    "kfold_cv_did",
    "influence_diagnostics",
    "cv_summary",
]


def loo_cv_did(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    layer: str | None = None,
    exclude_crossovers: bool = True,
    aggregate: AggregateMode = "participant_visit",
    standardize: bool = True,
) -> pd.DataFrame:
    """Leave-one-out cross-validation for DiD analysis.

    For each participant, fits the DiD model without that participant
    and records the effect estimate. This reveals which participants
    have the largest influence on the results.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to analyze.
    design
        TrialDesign object.
    visits
        (baseline, followup) visits.
    layer
        Expression layer.
    exclude_crossovers
        Exclude crossover participants.
    aggregate
        Aggregation mode passed to did_table (default "participant_visit").
    standardize
        Whether to z-score the outcome variable before fitting (default True).

    Returns
    -------
    pd.DataFrame
        Long-format results with columns:
        - feature: Feature name
        - excluded: ID of excluded participant
        - beta_DiD: DiD estimate without this participant
        - se_DiD: Standard error
        - influence: ``|beta_full - beta_loo| / SE``

    Examples
    --------
    >>> loo = loo_cv_did(adata, features=["sig_IFN"], design=design, visits=visits)
    >>> # Find influential participants
    >>> influential = loo[loo["influence"] > 1.0]
    >>> print(f"Influential participants: {influential['excluded'].unique()}")
    """
    # Get paired participants
    ad = subset_primary(adata, design, visits, exclude_crossovers=exclude_crossovers)
    participants = ad.obs[design.participant_col].unique().tolist()

    # Full-sample estimates for comparison
    full_res = did_table(
        adata, features, design, visits,
        exclude_crossovers=exclude_crossovers, layer=layer, aggregate=aggregate,
        standardize=standardize,
    )
    full_betas = full_res.set_index("feature")["beta_DiD"].to_dict()
    full_ses = full_res.set_index("feature")["se_DiD"].to_dict()

    # LOO iterations
    rows = []
    for pid in participants:
        # Exclude this participant
        mask = adata.obs[design.participant_col] != pid
        ad_loo = adata[mask].copy()

        try:
            res_loo = did_table(
                ad_loo, features, design, visits,
                exclude_crossovers=exclude_crossovers, layer=layer, aggregate=aggregate,
                standardize=standardize,
            )

            for _, row in res_loo.iterrows():
                feat = row["feature"]
                beta_loo = row["beta_DiD"]
                se_loo = row["se_DiD"]

                # Influence measure
                beta_full = full_betas.get(feat, np.nan)
                se_full = full_ses.get(feat, np.nan)
                if not np.isnan(beta_full) and not np.isnan(beta_loo) and se_full > 0:
                    influence = abs(beta_full - beta_loo) / se_full
                else:
                    influence = np.nan

                rows.append({
                    "feature": feat,
                    "excluded": pid,
                    "beta_DiD": beta_loo,
                    "se_DiD": se_loo,
                    "beta_full": beta_full,
                    "influence": influence,
                })

        except (ValueError, np.linalg.LinAlgError, KeyError) as exc:
            warnings.warn(
                f"LOO CV: model failed when excluding participant {pid}: {exc}",
                stacklevel=2,
            )
            for feat in features:
                rows.append({
                    "feature": feat,
                    "excluded": pid,
                    "beta_DiD": np.nan,
                    "se_DiD": np.nan,
                    "beta_full": full_betas.get(feat, np.nan),
                    "influence": np.nan,
                })

    return pd.DataFrame(rows)


def kfold_cv_did(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    k: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
    layer: str | None = None,
    exclude_crossovers: bool = True,
) -> pd.DataFrame:
    """K-fold cross-validation for DiD effect stability.

    Randomly partitions participants into K folds and estimates DiD
    using K-1 folds. Repeating this process gives a distribution of
    estimates that reflects sampling variability.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Features to analyze.
    design
        TrialDesign object.
    visits
        (baseline, followup) visits.
    k
        Number of folds (default 5).
    n_repeats
        Number of times to repeat CV (default 10).
    seed
        Random seed.
    layer
        Expression layer.
    exclude_crossovers
        Exclude crossover participants.

    Returns
    -------
    pd.DataFrame
        Results with one row per feature:
        - feature: Feature name
        - beta_full: Full-sample estimate
        - beta_cv_mean: Mean CV estimate
        - beta_cv_sd: SD of CV estimates
        - beta_cv_lower, beta_cv_upper: 2.5-97.5 percentiles
        - cv_stability: beta_cv_mean / beta_cv_sd (higher = more stable)
        - sign_consistency: Proportion of CV estimates with same sign as full

    Examples
    --------
    >>> cv = kfold_cv_did(adata, features=genes, design=design, visits=visits, k=5)
    >>> # Check stability
    >>> stable = cv[cv["sign_consistency"] > 0.9]
    >>> print(f"Stable effects: {len(stable)} / {len(cv)}")
    """
    rng = np.random.default_rng(seed)

    # Get paired participants
    ad = subset_primary(adata, design, visits, exclude_crossovers=exclude_crossovers)
    participants = ad.obs[design.participant_col].unique().tolist()
    n_participants = len(participants)

    if n_participants < k:
        k = max(2, n_participants // 2)

    # Full-sample estimates
    full_res = did_table(
        adata, features, design, visits,
        exclude_crossovers=exclude_crossovers, layer=layer
    )
    full_betas = full_res.set_index("feature")["beta_DiD"].to_dict()

    # Collect CV estimates
    cv_estimates: dict[str, list[float]] = {feat: [] for feat in features}
    n_failed_folds = 0

    for _ in range(n_repeats):
        # Shuffle participants
        shuffled = rng.permutation(participants).tolist()

        # K-fold splits
        fold_size = n_participants // k
        for fold_idx in range(k):
            # Exclude this fold
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < k - 1 else n_participants
            excluded = set(shuffled[start:end])
            included = [p for p in participants if p not in excluded]

            if len(included) < 4:  # Need minimum participants
                continue

            # Subset data
            mask = adata.obs[design.participant_col].isin(included)
            ad_fold = adata[mask].copy()

            try:
                res_fold = did_table(
                    ad_fold, features, design, visits,
                    exclude_crossovers=exclude_crossovers, layer=layer
                )

                for _, row in res_fold.iterrows():
                    feat = row["feature"]
                    if feat in cv_estimates:
                        cv_estimates[feat].append(row["beta_DiD"])

            except (ValueError, np.linalg.LinAlgError, KeyError):
                n_failed_folds += 1

    if n_failed_folds > 0:
        total_folds = n_repeats * k
        warnings.warn(
            f"K-fold CV: {n_failed_folds}/{total_folds} folds failed to fit.",
            stacklevel=2,
        )

    # Summarize CV estimates
    rows = []
    for feat in features:
        estimates = np.array(cv_estimates[feat])
        valid = estimates[~np.isnan(estimates)]
        beta_full = full_betas.get(feat, np.nan)

        if len(valid) >= 3:
            cv_mean = np.mean(valid)
            cv_sd = np.std(valid, ddof=1)
            cv_lower = np.percentile(valid, 2.5)
            cv_upper = np.percentile(valid, 97.5)
            stability = cv_mean / cv_sd if cv_sd > 0 else np.nan

            # Sign consistency
            if not np.isnan(beta_full):
                same_sign = np.sum(np.sign(valid) == np.sign(beta_full))
                sign_consistency = same_sign / len(valid)
            else:
                sign_consistency = np.nan
        else:
            cv_mean = cv_sd = cv_lower = cv_upper = stability = sign_consistency = np.nan

        rows.append({
            "feature": feat,
            "beta_full": beta_full,
            "beta_cv_mean": cv_mean,
            "beta_cv_sd": cv_sd,
            "beta_cv_lower": cv_lower,
            "beta_cv_upper": cv_upper,
            "cv_stability": stability,
            "sign_consistency": sign_consistency,
            "n_cv_samples": len(valid),
        })

    return pd.DataFrame(rows)


def influence_diagnostics(
    loo_results: pd.DataFrame,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Summarize influence diagnostics from LOO results.

    Parameters
    ----------
    loo_results
        Output from loo_cv_did.
    threshold
        Influence threshold for flagging (default 1.0 = 1 SE shift).

    Returns
    -------
    pd.DataFrame
        Per-participant influence with columns:
        - feature: Feature name
        - excluded: ID of excluded participant
        - influence: Influence score for this participant
        - beta_DiD: DiD estimate without this participant
        - is_influential: Whether influence > threshold
    """
    if loo_results.empty:
        return pd.DataFrame()

    # Return per-participant results with influence flag
    result = loo_results.copy()
    result["is_influential"] = result["influence"].abs() > threshold
    return result[["feature", "excluded", "influence", "beta_DiD", "beta_full", "is_influential"]]


def cv_summary(
    cv_results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Generate a summary of CV results (works with both LOO and k-fold).

    Parameters
    ----------
    cv_results
        Output from loo_cv_did or kfold_cv_did.
    alpha
        Significance level for classification.

    Returns
    -------
    pd.DataFrame
        Summary per feature with:
        - feature: Feature name
        - mean_estimate: Full-sample estimate (or mean LOO estimate)
        - mean_loo: Mean of LOO estimates
        - std_loo: Standard deviation of LOO estimates
        - cv: Coefficient of variation (std/mean)
    """
    df = cv_results.copy()

    # Detect LOO vs kfold format
    is_loo = "excluded" in df.columns

    if is_loo:
        # LOO format: one row per feature-participant
        rows = []
        for feat, group in df.groupby("feature"):
            betas = group["beta_DiD"].dropna()
            beta_full = group["beta_full"].iloc[0] if "beta_full" in group.columns else np.nan

            if len(betas) >= 2:
                mean_loo = betas.mean()
                std_loo = betas.std(ddof=1)
                cv = abs(std_loo / mean_loo) if abs(mean_loo) > 1e-12 else np.nan
            else:
                mean_loo = std_loo = cv = np.nan

            rows.append({
                "feature": feat,
                "mean_estimate": beta_full,
                "mean_loo": mean_loo,
                "std_loo": std_loo,
                "cv": cv,
                "n_loo": len(betas),
            })
        return pd.DataFrame(rows)

    else:
        # kfold format: one row per feature
        result = df[["feature"]].copy()
        result["mean_estimate"] = df.get("beta_full", np.nan)
        result["mean_loo"] = df.get("beta_cv_mean", np.nan)
        result["std_loo"] = df.get("beta_cv_sd", np.nan)

        # Calculate CV
        result["cv"] = np.where(
            result["mean_loo"].abs() > 1e-12,
            result["std_loo"].abs() / result["mean_loo"].abs(),
            np.nan
        )
        return result
