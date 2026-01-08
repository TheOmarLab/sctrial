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

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from ..adata_tools import subset_primary
from ..design import TrialDesign
from .did import did_table

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
    aggregate: str = "participant_visit",
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

    Returns
    -------
    pd.DataFrame
        Long-format results with columns:
        - feature: Feature name
        - excluded_participant: ID of excluded participant
        - beta_DiD: DiD estimate without this participant
        - se_DiD: Standard error
        - influence: ``|beta_full - beta_loo| / SE``

    Examples
    --------
    >>> loo = loo_cv_did(adata, features=["sig_IFN"], design=design, visits=visits)
    >>> # Find influential participants
    >>> influential = loo[loo["influence"] > 1.0]
    >>> print(f"Influential participants: {influential['excluded_participant'].unique()}")
    """
    # Get paired participants
    ad = subset_primary(adata, design, visits, exclude_crossovers=exclude_crossovers)
    participants = ad.obs[design.participant_col].unique().tolist()

    # Full-sample estimates for comparison
    full_res = did_table(
        adata, features, design, visits,
        exclude_crossovers=exclude_crossovers, layer=layer, aggregate=aggregate
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
                exclude_crossovers=exclude_crossovers, layer=layer, aggregate=aggregate
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
                    "excluded_participant": pid,
                    "beta_DiD": beta_loo,
                    "se_DiD": se_loo,
                    "beta_full": beta_full,
                    "influence": influence,
                })

        except Exception:
            # If model fails, record NaN
            for feat in features:
                rows.append({
                    "feature": feat,
                    "excluded_participant": pid,
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

            except Exception:
                pass

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
        Summary with:
        - feature: Feature name
        - max_influence: Maximum influence score
        - influential_participant: Participant with max influence
        - n_influential: Count of participants with influence > threshold
        - stability: 1 - (max_influence / median_influence)
    """
    rows = []
    for feat, group in loo_results.groupby("feature"):
        influences = group["influence"].dropna()

        if len(influences) == 0:
            rows.append({"feature": feat})
            continue

        max_inf = influences.max()
        max_idx = influences.idxmax()
        max_participant = group.loc[max_idx, "excluded_participant"]
        n_influential = (influences > threshold).sum()
        median_inf = influences.median()

        stability = 1 - (max_inf / median_inf) if median_inf > 0 else np.nan

        rows.append({
            "feature": feat,
            "max_influence": max_inf,
            "influential_participant": max_participant,
            "n_influential": int(n_influential),
            "mean_influence": influences.mean(),
            "stability_score": stability,
        })

    return pd.DataFrame(rows)


def cv_summary(
    kfold_results: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Generate a summary of k-fold CV results.

    Parameters
    ----------
    kfold_results
        Output from kfold_cv_did.
    alpha
        Significance level for classification.

    Returns
    -------
    dict
        Summary statistics:
        - n_features: Total features tested
        - n_stable: Features with sign_consistency > 0.9
        - n_significant_stable: Stable features with CI excluding 0
        - mean_cv_sd: Mean CV standard deviation
        - recommendations: List of recommendations
    """
    df = kfold_results.copy()

    n_features = len(df)
    n_stable = (df["sign_consistency"] > 0.9).sum()

    # Check if CV CI excludes 0
    excludes_zero = (df["beta_cv_lower"] > 0) | (df["beta_cv_upper"] < 0)
    n_significant_stable = (excludes_zero & (df["sign_consistency"] > 0.9)).sum()

    mean_cv_sd = df["beta_cv_sd"].mean()

    recommendations = []
    if n_stable < n_features * 0.5:
        recommendations.append(
            "Less than 50% of effects are stable. Consider larger sample size."
        )
    if mean_cv_sd > 0.5:
        recommendations.append(
            "High CV variability. Effects may not replicate reliably."
        )
    if n_significant_stable > 0:
        recommendations.append(
            f"{n_significant_stable} features show stable, significant effects."
        )

    return {
        "n_features": n_features,
        "n_stable": int(n_stable),
        "n_significant_stable": int(n_significant_stable),
        "mean_cv_sd": mean_cv_sd,
        "prop_stable": n_stable / n_features if n_features > 0 else np.nan,
        "recommendations": recommendations,
    }
