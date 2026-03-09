"""Dedicated tests for sctrial.design.TrialDesign.

Covers: construction validation, primary_visits, required_cols,
validate (column checks, arm label checks, extra-arm warning), arm_bin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sctrial.design import TrialDesign

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(
    arms: list[str] | None = None,
    visits: list[str] | None = None,
    extra_cols: dict | None = None,
    col_names: dict | None = None,
) -> AnnData:
    """Build a minimal trial AnnData for design tests."""
    arms = arms or ["Treated", "Control"]
    visits = visits or ["V1", "V2"]
    cn = {"participant": "participant_id", "visit": "visit", "arm": "arm"}
    if col_names:
        cn.update(col_names)
    rows = []
    pid = 0
    for arm in arms:
        for _ in range(2):  # 2 participants per arm
            pid += 1
            for v in visits:
                rows.append({cn["participant"]: f"P{pid}", cn["visit"]: v, cn["arm"]: arm})
    obs = pd.DataFrame(rows)
    if extra_cols:
        for k, v in extra_cols.items():
            obs[k] = v
    X = np.zeros((len(obs), 3))
    return AnnData(X=X, obs=obs)


# ===================================================================
# Construction Validation (__post_init__)
# ===================================================================


class TestConstruction:
    """TrialDesign is a frozen dataclass; identical arm labels are allowed at
    construction (needed for single-arm studies) but rejected by validate() and arm_bin()."""

    def test_identical_arm_labels_allowed(self):
        """Single-arm designs legitimately use arm_treated == arm_control."""
        d = TrialDesign(arm_treated="A", arm_control="A")
        assert d.arm_treated == "A"
        assert d.arm_control == "A"

    def test_distinct_arm_labels_ok(self):
        d = TrialDesign(arm_treated="Drug", arm_control="Placebo")
        assert d.arm_treated == "Drug"
        assert d.arm_control == "Placebo"

    def test_defaults_are_distinct(self):
        d = TrialDesign()
        assert d.arm_treated != d.arm_control

    def test_frozen(self):
        d = TrialDesign()
        with pytest.raises(AttributeError):
            d.arm_treated = "X"  # type: ignore[misc]


# ===================================================================
# primary_visits
# ===================================================================


class TestPrimaryVisits:
    """P3 #3: Uses 'is not None' fallback, not truthy 'or'."""

    def test_explicit_overrides_defaults(self):
        d = TrialDesign(baseline_visit="V1", followup_visit="V2")
        assert d.primary_visits(baseline="B", followup="F") == ("B", "F")

    def test_falls_back_to_defaults(self):
        d = TrialDesign(baseline_visit="V1", followup_visit="V2")
        assert d.primary_visits() == ("V1", "V2")

    def test_no_defaults_no_args_raises(self):
        d = TrialDesign()
        with pytest.raises(ValueError, match="Primary visits not specified"):
            d.primary_visits()

    def test_partial_default_raises(self):
        d = TrialDesign(baseline_visit="V1")
        with pytest.raises(ValueError, match="Primary visits not specified"):
            d.primary_visits()

    def test_empty_string_preserved(self):
        """Empty string is a valid (if unusual) visit label — not replaced by default."""
        d = TrialDesign(baseline_visit="V1", followup_visit="V2")
        b, f = d.primary_visits(baseline="", followup="")
        assert b == ""
        assert f == ""


# ===================================================================
# required_cols
# ===================================================================


class TestRequiredCols:
    def test_base_cols(self):
        d = TrialDesign()
        cols = d.required_cols()
        assert list(cols) == ["participant_id", "visit", "arm"]

    def test_include_celltype(self):
        d = TrialDesign(celltype_col="ct")
        cols = d.required_cols(include_celltype=True)
        assert "ct" in cols

    def test_include_celltype_none(self):
        """celltype_col=None is not included even when flag is True."""
        d = TrialDesign(celltype_col=None)
        cols = d.required_cols(include_celltype=True)
        assert len(cols) == 3  # just the base cols

    def test_include_crossover(self):
        d = TrialDesign(crossover_col="xover")
        cols = d.required_cols(include_crossover=True)
        assert "xover" in cols

    def test_include_crossover_none(self):
        d = TrialDesign(crossover_col=None)
        cols = d.required_cols(include_crossover=True)
        assert len(cols) == 3

    def test_include_both(self):
        d = TrialDesign(celltype_col="ct", crossover_col="xo")
        cols = d.required_cols(include_celltype=True, include_crossover=True)
        assert "ct" in cols and "xo" in cols
        assert len(cols) == 5


# ===================================================================
# validate — column checks
# ===================================================================


class TestValidateColumns:
    def test_valid_data_passes(self):
        adata = _make_adata()
        TrialDesign().validate(adata)

    def test_missing_column_raises(self):
        adata = _make_adata()
        d = TrialDesign(arm_col="treatment_group")
        with pytest.raises(KeyError, match="Missing required obs columns"):
            d.validate(adata)

    def test_missing_celltype_when_required(self):
        adata = _make_adata()
        d = TrialDesign(celltype_col="cell_type")
        with pytest.raises(KeyError, match="cell_type"):
            d.validate(adata, include_celltype=True)

    def test_celltype_not_required_by_default(self):
        """Even if celltype_col doesn't exist, validate passes without include_celltype."""
        adata = _make_adata()
        d = TrialDesign(celltype_col="nonexistent")
        d.validate(adata)  # should not raise


# ===================================================================
# validate — arm label checks
# ===================================================================


class TestValidateArmLabels:
    def test_correct_labels_pass(self):
        adata = _make_adata(arms=["Treated", "Control"])
        TrialDesign().validate(adata)

    def test_wrong_treated_label_raises(self):
        adata = _make_adata(arms=["Drug", "Placebo"])
        d = TrialDesign(arm_treated="Treated", arm_control="Placebo")
        with pytest.raises(ValueError, match="Arm labels not found"):
            d.validate(adata)

    def test_skip_arm_label_check(self):
        adata = _make_adata(arms=["Drug", "Placebo"])
        d = TrialDesign(arm_treated="Treated", arm_control="Control")
        d.validate(adata, check_arm_labels=False)  # should not raise

    def test_identical_arm_labels_raises(self):
        """P1 #2: validate rejects arm_treated == arm_control under check_arm_labels=True."""
        adata = _make_adata(arms=["A"])
        d = TrialDesign(arm_treated="A", arm_control="A")
        with pytest.raises(ValueError, match="must be distinct"):
            d.validate(adata, check_arm_labels=True)

    def test_identical_arm_labels_skip_check(self):
        """Single-arm studies pass validate with check_arm_labels=False."""
        adata = _make_adata(arms=["A"])
        d = TrialDesign(arm_treated="A", arm_control="A")
        d.validate(adata, check_arm_labels=False)  # should not raise


# ===================================================================
# validate — extra-arm warning (P1 #1)
# ===================================================================


class TestValidateExtraArms:
    """P1 #1: warn when data has arms beyond treated/control."""

    def test_two_arms_no_warning(self):
        adata = _make_adata(arms=["Treated", "Control"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrialDesign().validate(adata)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_extra_arm_warns(self):
        adata = _make_adata(arms=["Treated", "Control", "Placebo"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrialDesign().validate(adata)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "Placebo" in str(user_warnings[0].message)
        assert "treated as control" in str(user_warnings[0].message)

    def test_multiple_extra_arms_warns(self):
        adata = _make_adata(arms=["Treated", "Control", "LowDose", "HighDose"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TrialDesign().validate(adata)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        msg = str(user_warnings[0].message)
        assert "HighDose" in msg and "LowDose" in msg


# ===================================================================
# arm_bin
# ===================================================================


class TestArmBin:
    def test_binary_encoding(self):
        adata = _make_adata(arms=["Treated", "Control"])
        d = TrialDesign()
        result = d.arm_bin(adata.obs)
        assert set(result.unique()) == {0, 1}
        assert (result[adata.obs["arm"] == "Treated"] == 1).all()
        assert (result[adata.obs["arm"] == "Control"] == 0).all()

    def test_identical_arm_labels_raises(self):
        """P1 #2: arm_bin rejects identical arm labels."""
        adata = _make_adata(arms=["A"])
        d = TrialDesign(arm_treated="A", arm_control="A")
        with pytest.raises(ValueError, match="arm_bin.*requires distinct"):
            d.arm_bin(adata.obs)

    def test_missing_arm_col_raises(self):
        adata = _make_adata()
        d = TrialDesign(arm_col="nonexistent")
        with pytest.raises(KeyError, match="nonexistent"):
            d.arm_bin(adata.obs)

    def test_multi_arm_collapses_to_binary(self):
        """Extra arms beyond treated get 0 — verify this is the behavior."""
        adata = _make_adata(arms=["Treated", "Control", "Placebo"])
        d = TrialDesign()
        result = d.arm_bin(adata.obs)
        # Placebo participants should map to 0
        placebo_mask = adata.obs["arm"] == "Placebo"
        assert (result[placebo_mask] == 0).all()

    def test_index_alignment(self):
        """arm_bin result should be aligned to the input obs index."""
        adata = _make_adata()
        d = TrialDesign()
        result = d.arm_bin(adata.obs)
        assert list(result.index) == list(adata.obs.index)
