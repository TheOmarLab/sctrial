"""Trial design specification: TrialDesign dataclass and design detection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData

logger = logging.getLogger(__name__)

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

    arm_col: str | None = "arm"
    """Name of the column containing treatment arm assignments.

    Set to ``None`` for single-arm studies that lack an arm column.
    """

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

        Returns
        -------
        tuple[str, str]
            Tuple of (baseline, followup) visit labels.
        """
        b = baseline if baseline is not None else self.baseline_visit
        f = followup if followup is not None else self.followup_visit
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

        Returns
        -------
        list[str]
            List of required columns.
        """
        cols = [c for c in [self.participant_col, self.visit_col, self.arm_col] if c is not None]
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

        Returns
        -------
        None
        Raises
        ------
        KeyError
            If required columns are missing.
        ValueError
            If arm labels are not found in ``adata.obs[self.arm_col]``.
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
                f"Missing required obs columns: {missing}. Available: {list(obs.columns)}"
            )

        if check_arm_labels and self.arm_col is not None:
            if self.arm_treated == self.arm_control:
                raise ValueError(
                    f"arm_treated and arm_control must be distinct for "
                    f"between-arm analyses, got {self.arm_treated!r} for both. "
                    f"For single-arm studies, use check_arm_labels=False."
                )
            arms = set(pd.Series(obs[self.arm_col]).dropna().unique().tolist())
            if (self.arm_treated not in arms) or (self.arm_control not in arms):
                raise ValueError(
                    f"Arm labels not found in obs['{self.arm_col}']. "
                    f"Expected treated='{self.arm_treated}', control='{self.arm_control}'. "
                    f"Observed arms: {sorted(arms)}"
                )
            extra = arms - {self.arm_treated, self.arm_control}
            if extra:
                import warnings

                warnings.warn(
                    f"obs['{self.arm_col}'] contains arms beyond treated/control: "
                    f"{sorted(extra)}. These participants will be treated as "
                    f"control in arm_bin(). Consider subsetting your data to "
                    f"only the two arms of interest.",
                    UserWarning,
                    stacklevel=2,
                )

    def arm_bin(self, obs: pd.DataFrame) -> pd.Series:
        """Return 0/1 treated indicator aligned to obs.index.

        Parameters
        ----------
        obs
            DataFrame containing the participant-visit data.

        Returns
        -------
        pd.Series
            A Series with 0/1 indicator of treated status.
        Raises
        ------
        ValueError
            If arm_treated and arm_control are the same.
        KeyError
            If arm_col is not in obs.columns.
        """
        if self.arm_col is None:
            raise ValueError(
                "arm_bin() requires arm_col to be set. "
                "Single-arm designs (arm_col=None) do not have arm indicators."
            )
        if self.arm_treated == self.arm_control:
            raise ValueError(
                f"arm_bin() requires distinct arm labels, but "
                f"arm_treated == arm_control == {self.arm_treated!r}."
            )
        if self.arm_col not in obs.columns:
            raise KeyError(f"arm_col '{self.arm_col}' not in obs.")
        return (obs[self.arm_col] == self.arm_treated).astype(int)
