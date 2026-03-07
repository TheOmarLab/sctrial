"""Convenience functions for quick trial analysis workflows."""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd
from anndata import AnnData

from .design import TrialDesign
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
from .stats.did import did_table

logger = logging.getLogger(__name__)

__all__ = [
    "quick_did",
    "auto_detect_design",
]


def quick_did(
    adata: AnnData,
    module_scores: dict[str, list[str]],
    visits: tuple[str, str],
    participant_col: str = "participant_id",
    visit_col: str = "visit",
    arm_col: str = "arm",
    arm_treated: str = "Treated",
    arm_control: str = "Control",
    celltype_col: str | None = None,
    layer: str = "log1p_cpm",
    counts_layer: str = "counts",
    score_method: Literal["zmean", "mean"] = "zmean",
    min_genes: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """One-line wrapper for the most common DiD workflow.

    This function combines preprocessing, scoring, and DiD analysis
    in a single call for quick exploration.

    Parameters
    ----------
    adata
        AnnData object with trial data.
    module_scores
        Dictionary mapping module names to gene lists.
    visits
        Tuple of (baseline, followup) visit labels.  The order matters:
        the first element is the baseline visit and the second is the
        follow-up, which determines the sign of the DiD estimate.
    participant_col
        Column name for participant identifiers.
    visit_col
        Column name for visit labels.
    arm_col
        Column name for treatment arm.
    arm_treated
        Label for treated arm.
    arm_control
        Label for control arm.
    celltype_col
        Optional column name for cell types.
    layer
        Layer name for normalized expression. If not present, will be created.
    counts_layer
        Layer name for raw counts (used if layer needs to be created).
    score_method
        Method for gene set scoring ('mean' or 'zmean').
    min_genes
        Minimum number of overlapping genes for a module score to be
        computed.  Modules with fewer overlapping genes get NaN scores.
        Default is 3.
    **kwargs
        Additional arguments passed to did_table().

    Returns
    -------
    pd.DataFrame
        DiD results table.

    Examples
    --------
    >>> gene_sets = {
    ...     "OXPHOS": ["COX7A1", "ATP5F1A", "NDUFA1"],
    ...     "Glycolysis": ["PKM", "LDHA", "HK2"]
    ... }
    >>> res = quick_did(
    ...     adata,
    ...     module_scores=gene_sets,
    ...     visits=("V1", "V2")
    ... )
    >>> print(res[["feature", "beta_DiD", "p_DiD", "FDR_DiD"]])
    """
    # Validate that visit_col exists before doing anything else
    if visit_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{visit_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns[:20])}"
        )

    # Validate arm labels exist in the data
    actual_arms = set(adata.obs[arm_col].unique()) if arm_col in adata.obs.columns else set()
    if arm_col not in adata.obs.columns:
        raise ValueError(
            f"Column '{arm_col}' not found in adata.obs."
        )
    if arm_treated not in actual_arms:
        raise ValueError(
            f"arm_treated='{arm_treated}' not found in column '{arm_col}'. "
            f"Available values: {sorted(actual_arms)}"
        )
    if arm_control not in actual_arms:
        raise ValueError(
            f"arm_control='{arm_control}' not found in column '{arm_col}'. "
            f"Available values: {sorted(actual_arms)}"
        )

    # 1. Create design
    design = TrialDesign(
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        arm_treated=arm_treated,
        arm_control=arm_control,
        celltype_col=celltype_col,
    )

    # 2. Ensure preprocessing
    if layer not in adata.layers:
        logger.info("Creating '%s' layer from '%s'...", layer, counts_layer)
        adata = add_log1p_cpm_layer(
            adata, counts_layer=counts_layer, out_layer=layer
        )

    # 3. Score gene sets
    logger.info("Scoring %d gene sets...", len(module_scores))
    adata = score_gene_sets(
        adata, module_scores, layer=layer, method=score_method,
        prefix="ms_", min_genes=min_genes,
    )

    # 4. Run DiD
    features = [f"ms_{k}" for k in module_scores.keys()]
    logger.info("Running DiD for %d features...", len(features))
    return did_table(adata, features=features, design=design, visits=visits, **kwargs)


def _detect_column_patterns() -> dict[str, list[str]]:
    """Return column name patterns for auto-detection.

    Patterns are tried in order: exact match first, then word-boundary
    partial match (see :func:`_find_column`).
    """
    return {
        "participant": [
            "participant_id",
            "participant",
            "patient_id",
            "patient",
            "donor_id",
            "donor",
            "subject_id",
            "subject",
            "sample_id",
        ],
        "visit": [
            "visit",
            "timepoint",
            "time_point",
            "time",
            "day",
            "week",
            "collection_day",
        ],
        "arm": [
            "arm",
            "treatment_arm",
            "arm_id",
            "treatment",
            "group",
            "condition",
        ],
        "celltype": [
            "celltype",
            "cell_type",
            "cluster",
            "annotation",
            "cell_annotation",
            "celltype_major",
        ],
    }


def _auto_detect_arm_labels(
    arms: list[str],
    arm_treated: str | None,
    arm_control: str | None,
) -> tuple[str | None, str | None]:
    """Auto-detect arm labels based on common keywords."""
    treated_keywords = ["treat", "drug", "active", "intervention"]
    control_keywords = ["control", "placebo", "sham", "vehicle"]

    for arm in arms:
        arm_lower = str(arm).lower()
        if arm_treated is None and any(kw in arm_lower for kw in treated_keywords):
            arm_treated = str(arm)
        if arm_control is None and any(kw in arm_lower for kw in control_keywords):
            arm_control = str(arm)
    return arm_treated, arm_control


def _print_design_summary(
    participant_col: str,
    visit_col: str,
    arm_col: str,
    arm_treated: str,
    arm_control: str,
    celltype_col: str | None,
) -> None:
    """Print a summary of the detected trial design."""
    logger.info("\n" + "=" * 60)
    logger.info("AUTO-DETECTED TRIAL DESIGN")
    logger.info("=" * 60)
    logger.info("Participant column: %s", participant_col)
    logger.info("Visit column:       %s", visit_col)
    logger.info("Arm column:         %s", arm_col)
    logger.info("  Treated arm:      %s", arm_treated)
    logger.info("  Control arm:      %s", arm_control)
    if celltype_col:
        logger.info("Cell type column:   %s", celltype_col)
    else:
        logger.info("Cell type column:   (not detected)")
    logger.info("=" * 60)
    logger.warning("Please verify this design is correct before using!")
    logger.info("=" * 60 + "\n")


def auto_detect_design(
    adata: AnnData,
    arm_treated: str | None = None,
    arm_control: str | None = None,
) -> TrialDesign:
    """Auto-detect trial design from common column naming patterns.

    Looks for common patterns in column names:

    - participant: ``participant_id``, ``patient_id``, ``donor_id``,
      ``subject_id``, ``sample_id``
    - visit: ``visit``, ``timepoint``, ``time``, ``day``, ``week``,
      ``time_point``
    - arm: ``arm``, ``treatment_arm``, ``arm_id``, ``treatment``,
      ``group``, ``condition``
    - celltype: ``celltype``, ``cell_type``, ``cluster``,
      ``annotation``, ``cell_annotation``

    Column matching uses exact (case-insensitive) match first, then
    word-boundary partial match (pattern must appear at a word boundary
    in the column name, so ``"arm"`` matches ``"arm_id"`` but not
    ``"farm_id"``).

    Parameters
    ----------
    adata
        AnnData object to analyze.
    arm_treated
        Optional: specify the label for treated arm.  If not provided,
        the function tries keyword-based detection (e.g. "Treated",
        "Drug", "Active").  Raises if detection fails.
    arm_control
        Optional: specify the label for control arm.  If not provided,
        the function tries keyword-based detection (e.g. "Control",
        "Placebo").  Raises if detection fails.

    Returns
    -------
    TrialDesign
        Detected design (may need manual adjustment).

    Examples
    --------
    >>> design = auto_detect_design(adata)
    >>> print(f"Detected design: {design}")
    >>> # Verify the detected design. If arm labels need adjustment,
    >>> # create a new TrialDesign with corrected values:
    >>> from dataclasses import replace
    >>> design = replace(design, arm_treated="Drug_A", arm_control="Placebo")

    Raises
    ------
    ValueError
        If required columns cannot be detected, or if arm labels
        cannot be determined (more than 2 arms, only 1 arm, or no
        keyword match for ambiguous labels).
    """
    obs_cols = adata.obs.columns.tolist()
    obs_cols_lower = [c.lower() for c in obs_cols]

    patterns = _detect_column_patterns()
    participant_col = _find_column(obs_cols, obs_cols_lower, patterns["participant"])
    visit_col = _find_column(obs_cols, obs_cols_lower, patterns["visit"])
    arm_col = _find_column(obs_cols, obs_cols_lower, patterns["arm"])
    celltype_col = _find_column(obs_cols, obs_cols_lower, patterns["celltype"], required=False)

    # Validate required columns were detected (check early)
    if participant_col is None:
        raise ValueError(
            "Could not detect participant column. Please specify manually "
            "or rename column."
        )
    if visit_col is None:
        raise ValueError(
            "Could not detect visit column. Please specify manually "
            "or rename column."
        )
    if arm_col is None:
        raise ValueError(
            "Could not detect arm column. Please specify manually "
            "or rename column."
        )

    # Auto-detect arm labels if not provided
    if arm_treated is None or arm_control is None:
        unique_arms = adata.obs[arm_col].unique()
        if len(unique_arms) == 2:
            # Try keyword-based detection
            arm_treated, arm_control = _auto_detect_arm_labels(
                [str(a) for a in unique_arms], arm_treated, arm_control
            )

            # If keywords didn't resolve both labels, raise instead
            # of silently assigning by encounter order.
            if arm_treated is None or arm_control is None:
                raise ValueError(
                    f"Found 2 arms {sorted(str(a) for a in unique_arms)} in column '{arm_col}' but could not "
                    "determine which is treated vs control. "
                    "Please specify arm_treated and arm_control."
                )

            logger.info(
                "Auto-detected arms: treated='%s', control='%s'",
                arm_treated, arm_control,
            )
            logger.warning("Please verify these are correct!")
        elif len(unique_arms) > 2:
            raise ValueError(
                f"Found {len(unique_arms)} arms in column '{arm_col}': "
                f"{sorted(str(a) for a in unique_arms)}. "
                "Please specify arm_treated and arm_control manually."
            )
        else:
            raise ValueError(
                f"Found only 1 arm in column '{arm_col}': '{unique_arms[0]}'. "
                "DiD requires at least 2 arms."
            )

    # Create design
    design = TrialDesign(
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        arm_treated=arm_treated,
        arm_control=arm_control,
        celltype_col=celltype_col,
    )

    # Print summary
    _print_design_summary(
        participant_col,
        visit_col,
        arm_col,
        arm_treated,
        arm_control,
        celltype_col,
    )

    return design


def _find_column(
    columns: list[str],
    columns_lower: list[str],
    patterns: list[str],
    required: bool = True,
) -> str | None:
    """Find column matching one of the patterns.

    Matching strategy (tried in order for each pattern):

    1. **Exact match** (case-insensitive): ``col.lower() == pattern``
    2. **Word-boundary partial match**: pattern must appear at a word
       boundary (start of string, after ``_``, or after ``-``).
       This prevents ``"arm"`` from matching ``"farm_id"`` while still
       matching ``"arm_treatment"`` or ``"treatment_arm"``.
    """
    import re

    for pattern in patterns:
        # First try exact match (case-insensitive)
        for i, col_lower in enumerate(columns_lower):
            if col_lower == pattern:
                return columns[i]

        # Then try word-boundary partial match
        # Pattern must appear at start of string or after _ or -
        boundary_re = re.compile(r"(?:^|[_\-])" + re.escape(pattern))
        for i, col_lower in enumerate(columns_lower):
            if boundary_re.search(col_lower):
                return columns[i]

    if required:
        raise ValueError(
            f"Could not auto-detect column. Tried patterns: {patterns}\n"
            f"Available columns: {columns[:20]}\n"
            "Please specify the column manually when creating TrialDesign."
        )

    return None
