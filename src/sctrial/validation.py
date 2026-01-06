"""Data validation utilities for trial analysis."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from anndata import AnnData

from .datasets import count_paired
from .design import TrialDesign

__all__ = [
    "TrialDataValidator",
    "validate_adata",
    "validate_features",
    "diagnose_trial_data",
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
        if "counts" not in adata.layers and not _looks_like_counts(adata.X):
            issues.append(
                "No 'counts' layer found and adata.X doesn't appear to contain raw counts. "
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
    dict with keys:
        - n_participants: int
        - n_visits: int
        - n_arms: int
        - paired_participants: Dict[Tuple[str,str], int]
        - cells_per_participant: pd.Series
        - warnings: List[str]
        - recommendations: List[str]

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


def _looks_like_counts(X: Any, sample_size: int = 1000) -> bool:
    """Check if matrix looks like raw counts."""
    if X is None:
        return False

    import scipy.sparse as sp

    if sp.issparse(X):
        data = X.data
    else:
        data = np.asarray(X).ravel()

    if data.size == 0:
        return False

    # Sample for efficiency
    if data.size > sample_size:
        data = np.random.choice(data[np.isfinite(data)], size=sample_size, replace=False)

    # Check if values are non-negative integers
    return (data >= 0).all() and np.allclose(data, np.round(data), atol=1e-3)


def _print_diagnostic_report(report: dict[str, Any]) -> None:
    """Print formatted diagnostic report."""
    print("=" * 60)
    print("TRIAL DATA DIAGNOSTIC REPORT")
    print("=" * 60)

    print("\n📊 DATA SUMMARY")
    print(f"  Cells:        {report.get('n_cells', 'N/A'):,}")
    print(f"  Genes:        {report.get('n_genes', 'N/A'):,}")
    print(f"  Participants: {report.get('n_participants', 'N/A')}")
    print(f"  Visits:       {report.get('n_visits', 'N/A')}")
    print(f"  Arms:         {report.get('n_arms', 'N/A')}")

    if "visits" in report:
        print(f"    Visit labels: {', '.join(map(str, report['visits']))}")

    if "arms" in report:
        print(f"    Arm labels: {', '.join(map(str, report['arms']))}")

    if "paired_participants" in report:
        print("\n🔗 PAIRED PARTICIPANTS")
        for (v1, v2), count in report["paired_participants"].items():
            status = "✓" if count >= 4 else "⚠️"
            print(f"  {status} {v1} <-> {v2}: {count} paired")

    if "cells_per_participant_visit_mean" in report:
        print("\n📈 CELLS PER PARTICIPANT-VISIT")
        print(f"  Mean:   {report['cells_per_participant_visit_mean']:.1f}")
        print(f"  Median: {report['cells_per_participant_visit_median']:.1f}")
        print(f"  Min:    {report['cells_per_participant_visit_min']}")

    if "celltype_distribution" in report:
        print("\n🧬 CELL TYPE DISTRIBUTION")
        for ct, count in list(report["celltype_distribution"].items())[:10]:
            print(f"  {ct}: {count:,}")

    if report.get("warnings"):
        print(f"\n⚠️  WARNINGS ({len(report['warnings'])})")
        for w in report["warnings"]:
            print(f"  • {w}")

    if report.get("recommendations"):
        print("\n💡 RECOMMENDATIONS")
        for r in report["recommendations"]:
            print(f"  • {r}")

    print("=" * 60)
