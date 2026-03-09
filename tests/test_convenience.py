"""Comprehensive tests for sctrial.convenience module.

Covers: quick_did (visits validation, arm validation, min_genes pass-through,
auto-visit deprecation warning, layer creation) and auto_detect_design
(column detection, arm label detection, word-boundary matching, multi-arm
handling, missing column errors).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st
from sctrial.convenience import _find_column

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trial_adata(
    n_participants: int = 10,
    n_genes: int = 10,
    visits: tuple[str, str] = ("V1", "V2"),
    arms: tuple[str, str] = ("Treated", "Control"),
    col_names: dict[str, str] | None = None,
) -> AnnData:
    """Create a minimal trial AnnData for testing."""
    if col_names is None:
        col_names = {
            "participant": "participant_id",
            "visit": "visit",
            "arm": "arm",
            "celltype": "celltype",
        }
    rng = np.random.default_rng(42)
    obs_rows = []
    n_treated = n_participants // 2
    for i in range(n_participants):
        arm = arms[0] if i < n_treated else arms[1]
        for v in visits:
            obs_rows.append(
                {
                    col_names["participant"]: f"P{i}",
                    col_names["visit"]: v,
                    col_names["arm"]: arm,
                    col_names["celltype"]: "CT1",
                }
            )
    obs = pd.DataFrame(obs_rows)
    X = rng.poisson(2, (len(obs), n_genes)).astype(float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = [f"G{i}" for i in range(n_genes)]
    adata.layers["counts"] = X.copy()
    return adata


# ===================================================================
# quick_did — Input Validation
# ===================================================================


class TestQuickDidValidation:
    """Input validation for quick_did."""

    def test_missing_visit_col_raises(self):
        adata = _make_trial_adata()
        with pytest.raises(ValueError, match="not found"):
            st.quick_did(
                adata,
                module_scores={"S": ["G0", "G1", "G2"]},
                visits=("V1", "V2"),
                visit_col="nonexistent",
            )

    def test_missing_arm_col_raises(self):
        adata = _make_trial_adata()
        with pytest.raises(ValueError, match="not found"):
            st.quick_did(
                adata,
                module_scores={"S": ["G0", "G1", "G2"]},
                visits=("V1", "V2"),
                arm_col="nonexistent",
            )

    def test_wrong_arm_treated_raises(self):
        adata = _make_trial_adata()
        with pytest.raises(ValueError, match="arm_treated='Drug'"):
            st.quick_did(
                adata,
                module_scores={"S": ["G0", "G1", "G2"]},
                visits=("V1", "V2"),
                arm_treated="Drug",
            )

    def test_wrong_arm_control_raises(self):
        adata = _make_trial_adata()
        with pytest.raises(ValueError, match="arm_control='Placebo'"):
            st.quick_did(
                adata,
                module_scores={"S": ["G0", "G1", "G2"]},
                visits=("V1", "V2"),
                arm_control="Placebo",
            )


# ===================================================================
# quick_did — visits is required (no auto-detection)
# ===================================================================


class TestQuickDidVisits:
    """P1 #1: visits must be explicitly specified (no lexicographic guessing)."""

    def test_visits_required(self):
        """Omitting visits raises TypeError (required positional argument)."""
        adata = _make_trial_adata()
        with pytest.raises(TypeError):
            st.quick_did(adata, module_scores={"S": ["G0", "G1", "G2"]})

    def test_explicit_visits_works(self):
        adata = _make_trial_adata()
        res = st.quick_did(
            adata,
            module_scores={"S": ["G0", "G1", "G2"]},
            visits=("V1", "V2"),
            counts_layer="counts",
        )
        assert not res.empty


# ===================================================================
# quick_did — min_genes Pass-Through
# ===================================================================


class TestQuickDidMinGenes:
    """P2 #4: min_genes is now exposed in quick_did."""

    def test_small_module_default_min_genes_nan(self):
        """1-gene module → NaN with default min_genes=3."""
        adata = _make_trial_adata()
        res = st.quick_did(
            adata,
            module_scores={"tiny": ["G0"]},
            visits=("V1", "V2"),
            counts_layer="counts",
        )
        assert res["beta_DiD"].isna().all()

    def test_small_module_custom_min_genes(self):
        """1-gene module → valid result with min_genes=1."""
        adata = _make_trial_adata()
        res = st.quick_did(
            adata,
            module_scores={"tiny": ["G0"]},
            visits=("V1", "V2"),
            counts_layer="counts",
            min_genes=1,
        )
        assert not res["beta_DiD"].isna().all()


# ===================================================================
# quick_did — Layer Creation
# ===================================================================


class TestQuickDidLayer:
    """Layer creation path."""

    def test_creates_layer_if_missing(self):
        adata = _make_trial_adata()
        assert "log1p_cpm" not in adata.layers
        st.quick_did(
            adata,
            module_scores={"S": ["G0", "G1", "G2"]},
            visits=("V1", "V2"),
            counts_layer="counts",
        )
        # Layer should have been created by quick_did
        assert "log1p_cpm" in adata.layers

    def test_uses_existing_layer(self):
        adata = _make_trial_adata()
        adata.layers["log1p_cpm"] = adata.X * 0.5
        res = st.quick_did(
            adata,
            module_scores={"S": ["G0", "G1", "G2"]},
            visits=("V1", "V2"),
        )
        assert not res.empty


# ===================================================================
# quick_did — End-to-End
# ===================================================================


class TestQuickDidEndToEnd:
    """End-to-end quick_did tests."""

    def test_basic_run(self, sample_adata):
        gene_sets = {"Sig": ["G0", "G1", "G2"]}
        res = st.quick_did(
            sample_adata,
            module_scores=gene_sets,
            visits=("V1", "V2"),
            counts_layer="counts",
        )
        assert not res.empty
        assert "feature" in res.columns
        assert "beta_DiD" in res.columns

    def test_multiple_modules(self):
        adata = _make_trial_adata()
        gene_sets = {
            "A": ["G0", "G1", "G2"],
            "B": ["G3", "G4", "G5"],
        }
        res = st.quick_did(
            adata,
            module_scores=gene_sets,
            visits=("V1", "V2"),
            counts_layer="counts",
        )
        assert len(res) == 2
        assert set(res["feature"]) == {"ms_A", "ms_B"}


# ===================================================================
# auto_detect_design — Column Detection
# ===================================================================


class TestAutoDetectColumns:
    """Column name pattern matching."""

    def test_standard_columns(self):
        adata = _make_trial_adata()
        design = st.auto_detect_design(adata)
        assert design.participant_col == "participant_id"
        assert design.visit_col == "visit"
        assert design.arm_col == "arm"

    def test_alternative_column_names(self):
        adata = _make_trial_adata(
            col_names={
                "participant": "patient_id",
                "visit": "timepoint",
                "arm": "treatment",
                "celltype": "cell_type",
            },
        )
        design = st.auto_detect_design(adata)
        assert design.participant_col == "patient_id"
        assert design.visit_col == "timepoint"
        assert design.arm_col == "treatment"
        assert design.celltype_col == "cell_type"

    def test_missing_participant_col_raises(self):
        adata = _make_trial_adata(
            col_names={
                "participant": "my_id",  # not in patterns
                "visit": "visit",
                "arm": "arm",
                "celltype": "celltype",
            },
        )
        with pytest.raises(ValueError, match="participant"):
            st.auto_detect_design(adata)

    def test_missing_visit_col_raises(self):
        adata = _make_trial_adata(
            col_names={
                "participant": "participant_id",
                "visit": "phase",  # not in patterns
                "arm": "arm",
                "celltype": "celltype",
            },
        )
        with pytest.raises(ValueError, match="visit"):
            st.auto_detect_design(adata)

    def test_missing_arm_col_raises(self):
        adata = _make_trial_adata(
            col_names={
                "participant": "participant_id",
                "visit": "visit",
                "arm": "intervention_type",  # not in patterns
                "celltype": "celltype",
            },
        )
        with pytest.raises(ValueError, match="arm"):
            st.auto_detect_design(adata)

    def test_celltype_optional(self):
        """celltype not detected → celltype_col is None (no error)."""
        adata = _make_trial_adata(
            col_names={
                "participant": "participant_id",
                "visit": "visit",
                "arm": "arm",
                "celltype": "my_labels",  # not in patterns
            },
        )
        design = st.auto_detect_design(adata)
        assert design.celltype_col is None


# ===================================================================
# auto_detect_design — Arm Label Detection
# ===================================================================


class TestAutoDetectArmLabels:
    """P1 #2: Arm label auto-detection and failure modes."""

    def test_keyword_detection_treated_control(self):
        """Standard 'Treated'/'Control' labels are detected."""
        adata = _make_trial_adata(arms=("Treated", "Control"))
        design = st.auto_detect_design(adata)
        assert design.arm_treated == "Treated"
        assert design.arm_control == "Control"

    def test_keyword_detection_drug_placebo(self):
        """'Drug_A'/'Placebo' labels are detected via keyword matching."""
        adata = _make_trial_adata(arms=("Drug_A", "Placebo"))
        design = st.auto_detect_design(adata)
        assert design.arm_treated == "Drug_A"
        assert design.arm_control == "Placebo"

    def test_ambiguous_labels_raises(self):
        """P1 #2: Ambiguous labels (no keywords) raise instead of guessing."""
        adata = _make_trial_adata(arms=("GroupA", "GroupB"))
        with pytest.raises(ValueError, match="could not determine"):
            st.auto_detect_design(adata)

    def test_explicit_arm_labels_override(self):
        """User-provided arm labels bypass keyword detection."""
        adata = _make_trial_adata(arms=("GroupA", "GroupB"))
        design = st.auto_detect_design(adata, arm_treated="GroupA", arm_control="GroupB")
        assert design.arm_treated == "GroupA"
        assert design.arm_control == "GroupB"

    def test_multi_arm_raises(self):
        """P3 #5: >2 arms raises ValueError."""
        obs = pd.DataFrame(
            {
                "participant_id": ["P1", "P2", "P3"],
                "visit": ["V1", "V1", "V1"],
                "arm": ["ArmA", "ArmB", "ArmC"],
            }
        )
        adata = AnnData(X=np.ones((3, 2)), obs=obs)
        adata.var_names = ["G0", "G1"]
        with pytest.raises(ValueError, match="3 arms"):
            st.auto_detect_design(adata)

    def test_single_arm_raises(self):
        """Only 1 arm raises ValueError."""
        obs = pd.DataFrame(
            {
                "participant_id": ["P1", "P2"],
                "visit": ["V1", "V1"],
                "arm": ["OnlyArm", "OnlyArm"],
            }
        )
        adata = AnnData(X=np.ones((2, 2)), obs=obs)
        adata.var_names = ["G0", "G1"]
        with pytest.raises(ValueError, match="only 1 arm"):
            st.auto_detect_design(adata)


# ===================================================================
# _find_column — Word-Boundary Matching
# ===================================================================


class TestFindColumn:
    """P2 #3: Word-boundary matching prevents false positives."""

    def test_exact_match(self):
        cols = ["arm", "visit", "patient_id"]
        cols_lower = [c.lower() for c in cols]
        assert _find_column(cols, cols_lower, ["arm"]) == "arm"

    def test_exact_match_case_insensitive(self):
        cols = ["ARM", "Visit"]
        cols_lower = [c.lower() for c in cols]
        assert _find_column(cols, cols_lower, ["arm"]) == "ARM"

    def test_word_boundary_prefix(self):
        """'arm' matches 'arm_id' (starts with 'arm')."""
        cols = ["arm_id", "other"]
        cols_lower = [c.lower() for c in cols]
        assert _find_column(cols, cols_lower, ["arm"]) == "arm_id"

    def test_word_boundary_suffix(self):
        """'arm' matches 'treatment_arm' (_arm at boundary)."""
        cols = ["treatment_arm", "other"]
        cols_lower = [c.lower() for c in cols]
        assert _find_column(cols, cols_lower, ["arm"]) == "treatment_arm"

    def test_no_match_mid_word(self):
        """'arm' does NOT match 'farm_id' (not at word boundary)."""
        cols = ["farm_id", "other"]
        cols_lower = [c.lower() for c in cols]
        with pytest.raises(ValueError):
            _find_column(cols, cols_lower, ["arm"])

    def test_no_match_mid_word_pharma(self):
        """'arm' does NOT match 'pharma_group'."""
        cols = ["pharma_group", "other"]
        cols_lower = [c.lower() for c in cols]
        with pytest.raises(ValueError):
            _find_column(cols, cols_lower, ["arm"])

    def test_treatment_no_match_pretreatment(self):
        """'treatment' does NOT match 'pretreatment_status'."""
        cols = ["pretreatment_status", "other"]
        cols_lower = [c.lower() for c in cols]
        with pytest.raises(ValueError):
            _find_column(cols, cols_lower, ["treatment"])

    def test_required_false_returns_none(self):
        cols = ["unrelated"]
        cols_lower = [c.lower() for c in cols]
        result = _find_column(cols, cols_lower, ["arm"], required=False)
        assert result is None

    def test_required_true_raises(self):
        cols = ["unrelated"]
        cols_lower = [c.lower() for c in cols]
        with pytest.raises(ValueError, match="Could not auto-detect"):
            _find_column(cols, cols_lower, ["arm"], required=True)


# ===================================================================
# auto_detect_design — Summary Logging
# ===================================================================


class TestDesignSummaryLogging:
    """Design summary is logged."""

    def test_summary_logged(self, caplog):
        adata = _make_trial_adata()
        with caplog.at_level(logging.INFO, logger="sctrial.convenience"):
            st.auto_detect_design(adata)
        assert "AUTO-DETECTED TRIAL DESIGN" in caplog.text
        assert "participant" in caplog.text.lower()

    def test_celltype_not_detected_logged(self, caplog):
        adata = _make_trial_adata(
            col_names={
                "participant": "participant_id",
                "visit": "visit",
                "arm": "arm",
                "celltype": "my_labels",
            },
        )
        with caplog.at_level(logging.INFO, logger="sctrial.convenience"):
            st.auto_detect_design(adata)
        assert "not detected" in caplog.text.lower()
