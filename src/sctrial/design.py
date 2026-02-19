from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData

__all__ = ["TrialDesign"]


@dataclass(frozen=True)
class TrialDesign:
    """Describe the trial-design columns and metadata labels in `adata.obs`.

    The `TrialDesign` object centralizes the mapping of your study design to
    the AnnData object. It is used by almost all statistical and plotting
    functions in `sctrial`.
    """

    participant_col: str = "participant_id"
    """Name of the column containing unique participant identifiers."""

    visit_col: str = "visit"
    """Name of the column containing visit or timepoint labels."""

    arm_col: str = "arm"
    """Name of the column containing treatment arm assignments."""

    arm_treated: str = "Treated"
    """The label in `arm_col` representing the treatment/experimental group."""

    arm_control: str = "Control"
    """The label in `arm_col` representing the control/placebo group."""

    celltype_col: str | None = "celltype"
    """Optional name of the column containing cell-type annotations."""

    crossover_col: str | None = None
    """Optional name of the column containing boolean-like indicators for crossover cells."""

    baseline_visit: str | None = None
    """Optional default baseline visit label (e.g., 'Baseline', 'V1')."""

    followup_visit: str | None = None
    """Optional default follow-up visit label (e.g., 'Follow-up', 'V2')."""

    def primary_visits(
            self,
            baseline: str | None = None,
            followup: str | None = None,
    ) -> tuple[str, str]:
        """Return (baseline, followup) visit labels.

        Parameters
        ----------
        baseline
            Optional explicit baseline visit label. If None, uses
            ``self.baseline_visit``.
        followup
            Optional explicit follow-up visit label. If None, uses
            ``self.followup_visit``.
        """
        b = baseline or self.baseline_visit
        f = followup or self.followup_visit
        if b is None or f is None:
            raise ValueError(
                "Primary visits not specified. Provide baseline/followup or set "
                "TrialDesign(baseline_visit=..., followup_visit=...)."
            )
        return (b, f)

    def required_cols(
            self,
            *,
            include_celltype: bool = False,
            include_crossover: bool = False,
    ) -> Sequence[str]:
        """Return required obs columns for this design.

        Parameters
        ----------
        include_celltype
            If True, include ``celltype_col`` when it is defined.
        include_crossover
            If True, include ``crossover_col`` when it is defined.
        """
        cols = [self.participant_col, self.visit_col, self.arm_col]
        if include_celltype and self.celltype_col is not None:
            cols.append(self.celltype_col)
        if include_crossover and self.crossover_col is not None:
            cols.append(self.crossover_col)
        return cols

    def validate(
            self,
            adata: AnnData,
            *,
            include_celltype: bool = False,
            include_crossover: bool = False,
            check_arm_labels: bool = True,
    ) -> None:
        """Validate that `adata.obs` contains required columns and labels.

        Parameters
        ----------
        adata
            AnnData object to validate.
        include_celltype
            If True, require ``celltype_col`` in ``adata.obs``.
        include_crossover
            If True, require ``crossover_col`` in ``adata.obs``.
        check_arm_labels
            If True, verify that treated/control labels are present in ``arm_col``.
        """
        obs = adata.obs
        missing = [
            c
            for c in self.required_cols(
                include_celltype=include_celltype,
                include_crossover=include_crossover,
            )
            if c not in obs.columns
        ]
        if missing:
            raise KeyError(
                f"Missing required obs columns: {missing}. "
                f"Available: {list(obs.columns)}"
            )

        if check_arm_labels:
            arms = set(pd.Series(obs[self.arm_col]).dropna().unique().tolist())
            if (self.arm_treated not in arms) or (self.arm_control not in arms):
                raise ValueError(
                    f"Arm labels not found in obs['{self.arm_col}']. "
                    f"Expected treated='{self.arm_treated}', control='{self.arm_control}'. "
                    f"Observed arms: {sorted(arms)}"
                )

    def arm_bin(self, obs: pd.DataFrame) -> pd.Series:
        """Return 0/1 treated indicator aligned to obs.index."""
        if self.arm_col not in obs.columns:
            raise KeyError(f"arm_col '{self.arm_col}' not in obs.")
        return (obs[self.arm_col] == self.arm_treated).astype(int)
