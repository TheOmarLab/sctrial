"""Convenience functions for quick trial analysis workflows."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from anndata import AnnData

from .design import TrialDesign
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
from .stats.did import did_table


__all__ = [
    "quick_did",
    "auto_detect_design",
]


def quick_did(
    adata: AnnData,
    module_scores: Dict[str, List[str]],
    participant_col: str = "participant_id",
    visit_col: str = "visit",
    arm_col: str = "arm",
    arm_treated: str = "Treated",
    arm_control: str = "Control",
    visits: Optional[Tuple[str, str]] = None,
    celltype_col: Optional[str] = None,
    layer: str = "log1p_cpm",
    counts_layer: str = "counts",
    score_method: str = "zmean",
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
    visits
        Tuple of (baseline, followup) visit labels. If None, uses first two
        unique visits in sorted order.
    celltype_col
        Optional column name for cell types.
    layer
        Layer name for normalized expression. If not present, will be created.
    counts_layer
        Layer name for raw counts (used if layer needs to be created).
    score_method
        Method for gene set scoring ('mean' or 'zmean').
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
    # 1. Create design
    design = TrialDesign(
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        arm_treated=arm_treated,
        arm_control=arm_control,
        celltype_col=celltype_col,
    )

    # 2. Auto-detect visits if not provided
    if visits is None:
        all_visits = sorted(adata.obs[visit_col].unique())
        if len(all_visits) < 2:
            raise ValueError(
                f"Need at least 2 visits for DiD analysis, found {len(all_visits)}"
            )
        visits = (all_visits[0], all_visits[1])
        print(f"Auto-detected visits: {visits}")

    # 3. Ensure preprocessing
    if layer not in adata.layers:
        print(f"Creating '{layer}' layer from '{counts_layer}'...")
        adata = add_log1p_cpm_layer(
            adata, counts_layer=counts_layer, out_layer=layer
        )

    # 4. Score gene sets
    print(f"Scoring {len(module_scores)} gene sets...")
    adata = score_gene_sets(
        adata, module_scores, layer=layer, method=score_method, prefix="ms_"
    )

    # 5. Run DiD
    features = [f"ms_{k}" for k in module_scores.keys()]
    print(f"Running DiD for {len(features)} features...")
    return did_table(adata, features=features, design=design, visits=visits, **kwargs)


def auto_detect_design(
    adata: AnnData,
    arm_treated: Optional[str] = None,
    arm_control: Optional[str] = None,
) -> TrialDesign:
    """Auto-detect trial design from common column naming patterns.

    Looks for common patterns in column names:
    - participant: 'participant_id', 'patient_id', 'donor_id', 'subject_id', 'sample_id'
    - visit: 'visit', 'timepoint', 'time', 'day', 'week', 'time_point'
    - arm: 'arm', 'treatment', 'group', 'condition', 'arm_id'
    - celltype: 'celltype', 'cell_type', 'cluster', 'annotation', 'cell_annotation'

    Parameters
    ----------
    adata
        AnnData object to analyze.
    arm_treated
        Optional: specify the label for treated arm.
    arm_control
        Optional: specify the label for control arm.

    Returns
    -------
    TrialDesign
        Detected design (may need manual adjustment).

    Examples
    --------
    >>> design = auto_detect_design(adata)
    >>> print(f"Detected design: {design}")
    >>> # Verify and adjust if needed
    >>> design.arm_treated = "Drug_A"
    >>> design.arm_control = "Placebo"

    Raises
    ------
    ValueError
        If required columns cannot be detected.
    """
    obs_cols = adata.obs.columns.tolist()
    obs_cols_lower = [c.lower() for c in obs_cols]

    # Detect participant column
    participant_patterns = [
        "participant_id",
        "participant",
        "patient_id",
        "patient",
        "donor_id",
        "donor",
        "subject_id",
        "subject",
        "sample_id",
    ]
    participant_col = _find_column(obs_cols, obs_cols_lower, participant_patterns)

    # Detect visit column
    visit_patterns = [
        "visit",
        "timepoint",
        "time_point",
        "time",
        "day",
        "week",
        "collection_day",
    ]
    visit_col = _find_column(obs_cols, obs_cols_lower, visit_patterns)

    # Detect arm column
    arm_patterns = [
        "arm",
        "treatment",
        "group",
        "condition",
        "arm_id",
        "treatment_arm",
    ]
    arm_col = _find_column(obs_cols, obs_cols_lower, arm_patterns)

    # Detect celltype column (optional)
    celltype_patterns = [
        "celltype",
        "cell_type",
        "cluster",
        "annotation",
        "cell_annotation",
        "celltype_major",
    ]
    celltype_col = _find_column(
        obs_cols, obs_cols_lower, celltype_patterns, required=False
    )

    # Auto-detect arm labels if not provided
    if arm_col and (arm_treated is None or arm_control is None):
        unique_arms = adata.obs[arm_col].unique()
        if len(unique_arms) == 2:
            # Try to guess which is treated/control based on common names
            treated_keywords = ["treat", "drug", "active", "intervention"]
            control_keywords = ["control", "placebo", "sham", "vehicle"]

            for arm in unique_arms:
                arm_lower = str(arm).lower()
                if arm_treated is None and any(kw in arm_lower for kw in treated_keywords):
                    arm_treated = str(arm)
                if arm_control is None and any(kw in arm_lower for kw in control_keywords):
                    arm_control = str(arm)

            # If still not detected, just use the two arms
            if arm_treated is None:
                arm_treated = str(unique_arms[0])
            if arm_control is None:
                arm_control = str(unique_arms[1])

            print(f"Auto-detected arms: treated='{arm_treated}', control='{arm_control}'")
            print("⚠️  Please verify these are correct!")
        elif len(unique_arms) > 2:
            print(
                f"⚠️  Found {len(unique_arms)} arms: {list(unique_arms)}\n"
                "Please specify arm_treated and arm_control manually."
            )
        else:
            print(f"⚠️  Found only 1 arm: {unique_arms[0]}")

    # Create design
    design = TrialDesign(
        participant_col=participant_col,
        visit_col=visit_col,
        arm_col=arm_col,
        arm_treated=arm_treated or "UNKNOWN",
        arm_control=arm_control or "UNKNOWN",
        celltype_col=celltype_col,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("AUTO-DETECTED TRIAL DESIGN")
    print("=" * 60)
    print(f"Participant column: {participant_col}")
    print(f"Visit column:       {visit_col}")
    print(f"Arm column:         {arm_col}")
    print(f"  Treated arm:      {arm_treated}")
    print(f"  Control arm:      {arm_control}")
    if celltype_col:
        print(f"Cell type column:   {celltype_col}")
    else:
        print("Cell type column:   (not detected)")
    print("=" * 60)
    print("⚠️  Please verify this design is correct before using!")
    print("=" * 60 + "\n")

    return design


def _find_column(
    columns: List[str],
    columns_lower: List[str],
    patterns: List[str],
    required: bool = True,
) -> Optional[str]:
    """Find column matching one of the patterns."""
    for pattern in patterns:
        # First try exact match (case-insensitive)
        for i, col_lower in enumerate(columns_lower):
            if col_lower == pattern:
                return columns[i]

        # Then try partial match
        for i, col_lower in enumerate(columns_lower):
            if pattern in col_lower:
                return columns[i]

    if required:
        raise ValueError(
            f"Could not auto-detect column. Tried patterns: {patterns}\n"
            f"Available columns: {columns[:20]}\n"
            f"Please specify the column manually when creating TrialDesign."
        )

    return None
