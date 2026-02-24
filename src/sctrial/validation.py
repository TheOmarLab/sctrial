"""Data validation utilities for trial analysis."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from .datasets import count_paired
from .design import TrialDesign
from .utils import get_counts_matrix

logger = logging.getLogger(__name__)

__all__ = [
    "TrialDataValidator",
    "validate_adata",
    "validate_features",
    "diagnose_trial_data",
    "check_covariate_balance",
]


class TrialDataValidator:
    """Comprehensive validation for trial analysis data."""

    @staticmethod
    def validate_adata(
        adata: AnnData,
        design: TrialDesign,
        strict: bool = False,
    ) -> list[str]:
        """Validate AnnData object for trial analysis.

        Parameters
        ----------
        adata
            AnnData object to validate.
        design
            TrialDesign specifying column names.
        strict
            If True, raises exceptions. If False, returns warnings.

        Returns
        -------
        List of warning/error messages.

        Examples
        --------
        >>> validator = TrialDataValidator()
        >>> issues = validator.validate_adata(adata, design, strict=False)
        >>> if issues:
        ...     print(f"Found {len(issues)} issues:")
        ...     for issue in issues:
        ...         print(f"  - {issue}")
        """
        issues = []

        # Check required columns
        required_cols = [design.participant_col, design.visit_col, design.arm_col]
        for col in required_cols:
            if col not in adata.obs.columns:
                msg = f"Required column '{col}' not found in adata.obs"
                if strict:
                    raise KeyError(msg)
                issues.append(msg)

        # Check for missing data
        for col in required_cols:
            if col in adata.obs.columns:
                n_missing = adata.obs[col].isna().sum()
                if n_missing > 0:
                    pct = 100 * n_missing / len(adata)
                    msg = f"{n_missing} ({pct:.1f}%) missing values in '{col}'"
                    issues.append(msg)

        # Check sample size
        if design.participant_col in adata.obs.columns:
            n_participants = adata.obs[design.participant_col].nunique()
            if n_participants < 4:
                msg = f"Only {n_participants} participants (minimum 4 required for DiD)"
                if strict:
                    raise ValueError(msg)
                issues.append(msg)
            elif n_participants < 10:
                issues.append(
                    f"Only {n_participants} participants; consider using "
                    "bootstrap (use_bootstrap=True) for robust inference"
                )

        # Check paired data
        if design.visit_col in adata.obs.columns:
            visits = adata.obs[design.visit_col].unique()
            if len(visits) < 2:
                msg = f"Only {len(visits)} visit(s) found (need >= 2 for longitudinal analysis)"
                if strict:
                    raise ValueError(msg)
                issues.append(msg)

        # Check for counts layer
        counts, _source = get_counts_matrix(adata)
        if counts is None:
            issues.append(
                "No raw counts found in adata.layers['counts'], adata.raw.X, adata.layers['raw'], or adata.X. "
                "For best results, provide raw counts in adata.layers['counts']"
            )

        # Check cell type column if specified
        if design.celltype_col:
            if design.celltype_col not in adata.obs.columns:
                msg = f"Cell type column '{design.celltype_col}' not found in adata.obs"
                if strict:
                    raise KeyError(msg)
                issues.append(msg)

        return issues

    @staticmethod
    def validate_features(
        adata: AnnData,
        features: Sequence[str],
        allow_missing: bool = False,
    ) -> tuple[list[str], list[str]]:
        """Validate feature names.

        Parameters
        ----------
        adata
            AnnData object.
        features
            List of feature names to validate.
        allow_missing
            If False, raises error for missing features.

        Returns
        -------
        Tuple of (valid_features, missing_features).

        Examples
        --------
        >>> valid, missing = TrialDataValidator.validate_features(
        ...     adata, ["Gene1", "Gene2", "NonExistent"]
        ... )
        >>> print(f"Valid: {valid}, Missing: {missing}")
        """
        valid = []
        missing = []

        for feat in features:
            if feat in adata.obs.columns or feat in adata.var_names:
                valid.append(feat)
            else:
                missing.append(feat)

        if missing and not allow_missing:
            raise KeyError(
                f"Features not found: {missing[:10]}\n"
                f"Available obs columns: {list(adata.obs.columns)[:10]}\n"
                f"Available var names: {list(adata.var_names)[:10]}\n"
                f"Hint: Check spelling and case sensitivity"
            )

        return valid, missing


def validate_adata(
    adata: AnnData,
    design: TrialDesign,
    strict: bool = False,
) -> list[str]:
    """Validate AnnData object for trial analysis.

    Convenience wrapper around TrialDataValidator.validate_adata().

    Parameters
    ----------
    adata
        AnnData object to validate.
    design
        TrialDesign specifying column names.
    strict
        If True, raises exceptions. If False, returns warnings.

    Returns
    -------
    List of warning/error messages.
    """
    return TrialDataValidator.validate_adata(adata, design, strict=strict)


def validate_features(
    adata: AnnData,
    features: Sequence[str],
    allow_missing: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate feature names.

    Convenience wrapper around TrialDataValidator.validate_features().

    Parameters
    ----------
    adata
        AnnData object.
    features
        List of feature names to validate.
    allow_missing
        If False, raises error for missing features.

    Returns
    -------
    Tuple of (valid_features, missing_features).
    """
    return TrialDataValidator.validate_features(adata, features, allow_missing)


def diagnose_trial_data(
    adata: AnnData,
    design: TrialDesign,
    verbose: bool = True,
) -> dict[str, Any]:
    """Comprehensive diagnostic report for trial data.

    Parameters
    ----------
    adata
        AnnData object to diagnose.
    design
        TrialDesign object.
    verbose
        If True, prints diagnostic report.

    Returns
    -------
    dict
        Diagnostic summary with keys including:
        - n_cells, n_genes, n_participants, n_visits, n_arms
        - paired_participants (dict of visit pairs -> counts)
        - cells_per_participant (pd.Series)
        - warnings (list of strings)
        - recommendations (list of strings)

    Examples
    --------
    >>> diagnostics = diagnose_trial_data(adata, design, verbose=True)
    >>> if diagnostics['warnings']:
    ...     print("Warnings found:")
    ...     for w in diagnostics['warnings']:
    ...         print(f"  - {w}")
    """
    report: dict[str, Any] = {}
    warnings_list: list[str] = []
    recommendations: list[str] = []

    # Basic counts
    report["n_cells"] = adata.n_obs
    report["n_genes"] = adata.n_vars

    if design.participant_col in adata.obs.columns:
        report["n_participants"] = adata.obs[design.participant_col].nunique()

        if report["n_participants"] < 4:
            warnings_list.append(
                f"Only {report['n_participants']} participants (minimum 4 required)"
            )
            recommendations.append("Consider pooling data from multiple cohorts")
        elif report["n_participants"] < 10:
            recommendations.append(
                "Sample size is small; use bootstrap inference (use_bootstrap=True)"
            )

    if design.visit_col in adata.obs.columns:
        visits = sorted(adata.obs[design.visit_col].unique())
        report["n_visits"] = len(visits)
        report["visits"] = visits

        # Check paired participants for all visit pairs
        if len(visits) >= 2:
            paired_counts = {}
            for i, v1 in enumerate(visits):
                for v2 in visits[i + 1 :]:
                    n_paired = count_paired(adata.obs, design.visit_col, [v1, v2], design.participant_col)
                    paired_counts[(v1, v2)] = n_paired

                    if n_paired < 4:
                        warnings_list.append(
                            f"Only {n_paired} paired participants for {v1} vs {v2}"
                        )

            report["paired_participants"] = paired_counts

    if design.arm_col in adata.obs.columns:
        arms = sorted(adata.obs[design.arm_col].unique())
        report["n_arms"] = len(arms)
        report["arms"] = arms

    # Check cell counts per participant
    if design.participant_col in adata.obs.columns:
        if design.visit_col in adata.obs.columns:
            cells_per_pv = adata.obs.groupby(
                [design.participant_col, design.visit_col], observed=True
            ).size()
            report["cells_per_participant_visit_mean"] = cells_per_pv.mean()
            report["cells_per_participant_visit_median"] = cells_per_pv.median()
            report["cells_per_participant_visit_min"] = cells_per_pv.min()

            if cells_per_pv.min() < 10:
                warnings_list.append(
                    f"Some participant-visits have < 10 cells (min: {cells_per_pv.min()})"
                )
                recommendations.append("Consider QC filtering to remove low-quality samples")
        else:
            cells_per_p = adata.obs.groupby(design.participant_col, observed=True).size()
            report["cells_per_participant_mean"] = cells_per_p.mean()

    # Check cell type distribution if available
    if design.celltype_col and design.celltype_col in adata.obs.columns:
        celltype_counts = adata.obs[design.celltype_col].value_counts()
        report["n_celltypes"] = len(celltype_counts)
        report["celltype_distribution"] = celltype_counts.to_dict()

    report["warnings"] = warnings_list
    report["recommendations"] = recommendations

    if verbose:
        _print_diagnostic_report(report)

    return report


def check_covariate_balance(
    adata: AnnData,
    design: TrialDesign,
    covariates: Sequence[str],
    *,
    visit: str | None = None,
    dropna: bool = True,
    smd_threshold: float = 0.1,
) -> pd.DataFrame:
    """Compute standardized mean differences (SMD) for baseline covariates.

    This compares treated vs control arms at a single visit (usually baseline)
    and reports SMD values that quantify imbalance.

    Parameters
    ----------
    adata
        AnnData object with trial metadata in ``adata.obs``.
    design
        TrialDesign describing participant, visit, and arm columns.
    covariates
        List of covariate column names in ``adata.obs``.
    visit
        Visit label to use for balance checks. If None, uses
        ``design.baseline_visit``; otherwise raises if not available.
    dropna
        If True, drop rows with missing covariate values for each covariate.
    smd_threshold
        Absolute SMD threshold for "balanced" flag (default 0.1).

    Returns
    -------
    pd.DataFrame
        Table with SMD values. Numeric covariates produce one row per covariate.
        Categorical covariates produce one row per level with proportions.
    """
    if design.visit_col not in adata.obs.columns:
        raise KeyError(f"visit_col '{design.visit_col}' not in adata.obs")
    if design.arm_col not in adata.obs.columns:
        raise KeyError(f"arm_col '{design.arm_col}' not in adata.obs")

    if visit is None:
        if design.baseline_visit is None:
            raise ValueError("visit must be provided or design.baseline_visit must be set.")
        visit = design.baseline_visit

    obs = adata.obs.copy()
    obs = obs[obs[design.visit_col] == visit].copy()
    if obs.empty:
        raise ValueError(f"No observations found for visit '{visit}'.")

    # collapse to participant-level to avoid pseudoreplication
    group_cols = [design.participant_col, design.arm_col]
    base = obs[group_cols + list(covariates)].copy()

    rows: list[dict[str, object]] = []
    for cov in covariates:
        if cov not in base.columns:
            raise KeyError(f"Covariate '{cov}' not found in adata.obs")
        sub = base[group_cols + [cov]].copy()
        if dropna:
            sub = sub.dropna(subset=[cov])
        if sub.empty:
            continue

        # reduce to participant-level
        sub = sub.groupby(group_cols, observed=True)[cov].first().reset_index()

        treated = sub[sub[design.arm_col] == design.arm_treated][cov]
        control = sub[sub[design.arm_col] == design.arm_control][cov]

        if treated.empty or control.empty:
            continue

        if pd.api.types.is_numeric_dtype(sub[cov]):
            mean_t = float(treated.mean())
            mean_c = float(control.mean())
            sd_t = float(treated.std(ddof=1))
            sd_c = float(control.std(ddof=1))
            pooled = np.sqrt((sd_t**2 + sd_c**2) / 2) if (sd_t > 0 and sd_c > 0) else np.nan
            smd = (mean_t - mean_c) / pooled if pooled and np.isfinite(pooled) else np.nan
            rows.append({
                "covariate": cov,
                "level": None,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "smd": float(smd) if np.isfinite(smd) else np.nan,
                "n_treated": int(treated.shape[0]),
                "n_control": int(control.shape[0]),
                "balanced": bool(np.isfinite(smd) and abs(smd) < smd_threshold),
            })
        else:
            # categorical: compute SMD for each level via proportions
            levels = pd.Series(sub[cov].astype(str)).unique()
            for lvl in levels:
                p_t = float((treated.astype(str) == lvl).mean())
                p_c = float((control.astype(str) == lvl).mean())
                p_pool = (p_t + p_c) / 2
                denom = np.sqrt(p_pool * (1 - p_pool)) if 0 < p_pool < 1 else np.nan
                smd = (p_t - p_c) / denom if denom and np.isfinite(denom) else np.nan
                rows.append({
                    "covariate": cov,
                    "level": str(lvl),
                    "mean_treated": p_t,
                    "mean_control": p_c,
                    "smd": float(smd) if np.isfinite(smd) else np.nan,
                    "n_treated": int(treated.shape[0]),
                    "n_control": int(control.shape[0]),
                    "balanced": bool(np.isfinite(smd) and abs(smd) < smd_threshold),
                })

    return pd.DataFrame(rows)


def _print_diagnostic_report(report: dict[str, Any]) -> None:
    """Print formatted diagnostic report."""
    logger.info("=" * 60)
    logger.info("TRIAL DATA DIAGNOSTIC REPORT")
    logger.info("=" * 60)

    logger.info("DATA SUMMARY")
    logger.info(f"  Cells:        {report.get('n_cells', 'N/A'):,}")
    logger.info(f"  Genes:        {report.get('n_genes', 'N/A'):,}")
    logger.info(f"  Participants: {report.get('n_participants', 'N/A')}")
    logger.info(f"  Visits:       {report.get('n_visits', 'N/A')}")
    logger.info(f"  Arms:         {report.get('n_arms', 'N/A')}")

    if "visits" in report:
        logger.info(f"    Visit labels: {', '.join(map(str, report['visits']))}")

    if "arms" in report:
        logger.info(f"    Arm labels: {', '.join(map(str, report['arms']))}")

    if "paired_participants" in report:
        logger.info("PAIRED PARTICIPANTS")
        for (v1, v2), count in report["paired_participants"].items():
            status = "OK" if count >= 4 else "LOW"
            logger.info(f"  [{status}] {v1} <-> {v2}: {count} paired")

    if "cells_per_participant_visit_mean" in report:
        logger.info("CELLS PER PARTICIPANT-VISIT")
        logger.info(f"  Mean:   {report['cells_per_participant_visit_mean']:.1f}")
        logger.info(f"  Median: {report['cells_per_participant_visit_median']:.1f}")
        logger.info(f"  Min:    {report['cells_per_participant_visit_min']}")

    if "celltype_distribution" in report:
        logger.debug("CELL TYPE DISTRIBUTION")
        for ct, count in list(report["celltype_distribution"].items())[:10]:
            logger.debug(f"  {ct}: {count:,}")

    if report.get("warnings"):
        logger.warning(f"WARNINGS ({len(report['warnings'])})")
        for w in report["warnings"]:
            logger.warning(f"  - {w}")

    if report.get("recommendations"):
        logger.info("RECOMMENDATIONS")
        for r in report["recommendations"]:
            logger.info(f"  - {r}")

    logger.info("=" * 60)
