"""Comprehensive tests for sctrial.adata_tools."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st
from sctrial.adata_tools import _require_cols, _to_bool_series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial_adata(
    n: int = 6,
    *,
    arms: list[str] | None = None,
    visits: list[str] | None = None,
    celltypes: list[str] | None = None,
    crossover: list | None = None,
    n_genes: int = 3,
) -> AnnData:
    """Build a small trial-like AnnData for testing."""
    if arms is None:
        arms = ["T", "C"] * (n // 2) + ["T"] * (n % 2)
    if visits is None:
        visits = (["V1", "V2"] * ((n + 1) // 2))[:n]
    if celltypes is None:
        celltypes = (["A", "B"] * ((n + 1) // 2))[:n]

    obs = pd.DataFrame({
        "arm": arms[:n],
        "visit": visits[:n],
        "celltype": celltypes[:n],
        "participant_id": [f"P{i}" for i in range(n)],
    })
    if crossover is not None:
        obs["is_crossover"] = crossover[:n]

    X = np.arange(n * n_genes, dtype=float).reshape(n, n_genes)
    ad = AnnData(X=X, obs=obs)
    ad.var_names = [f"G{i}" for i in range(n_genes)]
    return ad


# ---------------------------------------------------------------------------
# _require_cols
# ---------------------------------------------------------------------------

class TestRequireCols:

    def test_all_present(self):
        obs = pd.DataFrame({"a": [1], "b": [2]})
        _require_cols(obs, ["a", "b"])  # should not raise

    def test_missing_raises(self):
        obs = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="Missing required obs columns"):
            _require_cols(obs, ["a", "missing"])

    def test_empty_list(self):
        obs = pd.DataFrame({"a": [1]})
        _require_cols(obs, [])  # should not raise


# ---------------------------------------------------------------------------
# _to_bool_series
# ---------------------------------------------------------------------------

class TestToBoolSeries:
    """Test all conversion paths of _to_bool_series."""

    def test_bool_dtype(self):
        s = pd.Series([True, False, True])
        result = _to_bool_series(s)
        pd.testing.assert_series_equal(result, pd.Series([True, False, True]))

    def test_bool_with_nan(self):
        s = pd.Series([True, None, False], dtype=object).astype("boolean")
        result = _to_bool_series(s)
        assert result.tolist() == [True, False, False]

    def test_numeric_int(self):
        s = pd.Series([0, 1, 0, 1])
        result = _to_bool_series(s)
        assert result.tolist() == [False, True, False, True]

    def test_numeric_float(self):
        s = pd.Series([0.0, 1.0, 0.0])
        result = _to_bool_series(s)
        assert result.tolist() == [False, True, False]

    def test_numeric_fractional_nonzero(self):
        """P2 fix: fractional non-zero values should be truthy."""
        s = pd.Series([0.0, 0.5, -0.3, 1.0])
        result = _to_bool_series(s)
        assert result.tolist() == [False, True, True, True]

    def test_numeric_nan(self):
        """NaN should be treated as False."""
        s = pd.Series([0.0, np.nan, 1.0])
        result = _to_bool_series(s)
        assert result.tolist() == [False, False, True]

    def test_numeric_inf(self):
        """P2 fix: inf should be treated as False (non-finite)."""
        s = pd.Series([0.0, np.inf, -np.inf, 1.0])
        result = _to_bool_series(s)
        assert result.tolist() == [False, False, False, True]

    def test_nonfinite_warns(self, caplog):
        """Non-finite values should emit a warning."""
        s = pd.Series([0.0, np.inf, np.nan], name="is_crossover")
        with caplog.at_level(logging.WARNING, logger="sctrial.adata_tools"):
            _to_bool_series(s)
        assert "non-finite" in caplog.text
        assert "2" in caplog.text  # 2 non-finite values

    def test_finite_no_warning(self, caplog):
        """No warning when all values are finite."""
        s = pd.Series([0.0, 1.0, 0.0], name="is_crossover")
        with caplog.at_level(logging.WARNING, logger="sctrial.adata_tools"):
            _to_bool_series(s)
        assert "non-finite" not in caplog.text

    def test_string_truthy(self):
        s = pd.Series(["true", "True", "TRUE", "t", "T", "yes", "Yes", "y", "Y", "1"])
        result = _to_bool_series(s)
        assert all(result), f"Expected all True, got {result.tolist()}"

    def test_string_falsy(self):
        s = pd.Series(["false", "False", "0", "no", "n", "N", "", "anything"])
        result = _to_bool_series(s)
        assert not any(result), f"Expected all False, got {result.tolist()}"

    def test_string_with_whitespace(self):
        s = pd.Series(["  true  ", " yes ", " 1 "])
        result = _to_bool_series(s)
        assert result.tolist() == [True, True, True]

    def test_categorical_truthy(self):
        s = pd.Categorical(["true", "false", "yes", "no"])
        result = _to_bool_series(pd.Series(s))
        assert result.tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# subset_primary
# ---------------------------------------------------------------------------

class TestSubsetPrimary:

    def test_basic_visit_filter(self):
        ad = _make_trial_adata(6, visits=["V1", "V2", "V3", "V1", "V2", "V3"])
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_primary(ad, design, visits=("V1", "V2"))
        assert len(result) == 4
        assert set(result.obs["visit"]) == {"V1", "V2"}

    def test_crossover_exclusion_true(self):
        ad = _make_trial_adata(4, crossover=[False, True, False, False])
        design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")
        result = st.subset_primary(
            ad, design, visits=("V1", "V2"), exclude_crossovers=True
        )
        assert len(result) == 3

    def test_crossover_exclusion_false(self):
        ad = _make_trial_adata(4, crossover=[False, True, False, False])
        design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")
        result = st.subset_primary(
            ad, design, visits=("V1", "V2"), exclude_crossovers=False
        )
        assert len(result) == 4

    def test_no_crossover_col(self):
        """exclude_crossovers=True but no crossover_col set → no filtering."""
        ad = _make_trial_adata(4)
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_primary(
            ad, design, visits=("V1", "V2"), exclude_crossovers=True
        )
        assert len(result) == 4

    def test_fractional_crossover_excluded(self):
        """P2 fix: fractional non-zero crossover values should be excluded."""
        ad = _make_trial_adata(4, crossover=[0.0, 0.5, 0.0, 0.0])
        design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")
        result = st.subset_primary(
            ad, design, visits=("V1", "V2"), exclude_crossovers=True
        )
        assert len(result) == 3

    def test_inf_crossover_treated_as_false(self):
        """P2 fix: inf crossover should not crash, treated as False."""
        ad = _make_trial_adata(4, crossover=[0.0, np.inf, 0.0, 1.0])
        design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")
        result = st.subset_primary(
            ad, design, visits=("V1", "V2"), exclude_crossovers=True
        )
        # inf → False (kept), 1.0 → True (excluded)
        assert len(result) == 3

    def test_string_crossover(self):
        ad = _make_trial_adata(4, crossover=["false", "true", "no", "yes"])
        design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")
        result = st.subset_primary(
            ad, design, visits=("V1", "V2"), exclude_crossovers=True
        )
        assert len(result) == 2

    def test_missing_visit_col_raises(self):
        ad = AnnData(X=np.zeros((2, 1)))
        design = st.TrialDesign(visit_col="visit")
        with pytest.raises(KeyError, match="Missing required"):
            st.subset_primary(ad, design, visits=("V1", "V2"))

    def test_returns_copy(self):
        ad = _make_trial_adata(4)
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_primary(ad, design, visits=("V1", "V2"))
        assert result is not ad

    def test_empty_result(self):
        """All cells filtered out → empty AnnData."""
        ad = _make_trial_adata(4, visits=["V3", "V3", "V3", "V3"])
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_primary(ad, design, visits=("V1", "V2"))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# subset_cells
# ---------------------------------------------------------------------------

class TestSubsetCells:

    def test_filter_by_arm(self):
        ad = _make_trial_adata(4, arms=["T", "T", "C", "C"])
        design = st.TrialDesign(arm_col="arm", visit_col="visit")
        result = st.subset_cells(ad, design, arm="T")
        assert len(result) == 2
        assert (result.obs["arm"] == "T").all()

    def test_filter_by_visit(self):
        ad = _make_trial_adata(4, visits=["V1", "V2", "V1", "V2"])
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_cells(ad, design, visit="V1")
        assert len(result) == 2
        assert (result.obs["visit"] == "V1").all()

    def test_filter_by_celltype(self):
        ad = _make_trial_adata(4, celltypes=["A", "B", "A", "B"])
        design = st.TrialDesign(visit_col="visit", celltype_col="celltype")
        result = st.subset_cells(ad, design, celltype="A")
        assert len(result) == 2
        assert (result.obs["celltype"] == "A").all()

    def test_filter_combined(self):
        ad = _make_trial_adata(
            4,
            arms=["T", "T", "C", "C"],
            visits=["V1", "V2", "V1", "V2"],
            celltypes=["A", "A", "B", "B"],
        )
        design = st.TrialDesign(
            arm_col="arm", visit_col="visit", celltype_col="celltype"
        )
        result = st.subset_cells(ad, design, arm="T", visit="V1", celltype="A")
        assert len(result) == 1

    def test_exclude_crossovers(self):
        ad = _make_trial_adata(4, crossover=[False, True, False, False])
        design = st.TrialDesign(visit_col="visit", crossover_col="is_crossover")
        result = st.subset_cells(ad, design, exclude_crossovers=True)
        assert len(result) == 3

    def test_celltype_col_not_set_raises(self):
        ad = _make_trial_adata(4)
        design = st.TrialDesign(visit_col="visit", celltype_col=None)
        with pytest.raises(ValueError, match="celltype_col must be set"):
            st.subset_cells(ad, design, celltype="A")

    def test_missing_column_raises(self):
        ad = AnnData(X=np.zeros((2, 1)))
        design = st.TrialDesign(arm_col="arm", visit_col="visit")
        with pytest.raises(KeyError, match="Missing required"):
            st.subset_cells(ad, design, arm="T")

    def test_no_filters_returns_all(self):
        ad = _make_trial_adata(4)
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_cells(ad, design)
        assert len(result) == 4

    def test_returns_copy(self):
        ad = _make_trial_adata(4)
        design = st.TrialDesign(visit_col="visit")
        result = st.subset_cells(ad, design)
        assert result is not ad

    def test_empty_result(self):
        ad = _make_trial_adata(4, arms=["T", "T", "T", "T"])
        design = st.TrialDesign(arm_col="arm", visit_col="visit")
        result = st.subset_cells(ad, design, arm="C")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# profile_features
# ---------------------------------------------------------------------------

class TestProfileFeatures:

    def test_gene_feature(self):
        ad = _make_trial_adata(4, arms=["T", "T", "C", "C"])
        result = st.profile_features(ad, features=["G0"], groupby="arm")
        assert "G0" in result.columns
        assert set(result.index) == {"T", "C"}

    def test_obs_feature(self):
        ad = _make_trial_adata(4, arms=["T", "T", "C", "C"])
        ad.obs["score"] = [1.0, 2.0, 3.0, 4.0]
        result = st.profile_features(ad, features=["score"], groupby="arm")
        assert "score" in result.columns
        assert result.loc["T", "score"] == pytest.approx(1.5)
        assert result.loc["C", "score"] == pytest.approx(3.5)

    def test_mixed_features(self):
        """Both gene and obs-column features in one call."""
        ad = _make_trial_adata(4, arms=["T", "T", "C", "C"])
        ad.obs["score"] = [10.0, 20.0, 30.0, 40.0]
        result = st.profile_features(
            ad, features=["G0", "score"], groupby="arm"
        )
        assert list(result.columns) == ["G0", "score"]

    def test_with_layer(self):
        ad = _make_trial_adata(4, arms=["T", "T", "C", "C"])
        ad.layers["norm"] = ad.X * 2
        result = st.profile_features(
            ad, features=["G0"], groupby="arm", layer="norm"
        )
        result_no_layer = st.profile_features(
            ad, features=["G0"], groupby="arm"
        )
        # layer values should be 2x
        assert result.loc["T", "G0"] == pytest.approx(
            result_no_layer.loc["T", "G0"] * 2
        )

    def test_agg_median(self):
        ad = _make_trial_adata(4, arms=["T", "T", "C", "C"])
        result = st.profile_features(
            ad, features=["G0"], groupby="arm", agg="median"
        )
        assert "G0" in result.columns

    def test_unknown_feature_raises(self):
        ad = _make_trial_adata(4)
        with pytest.raises(KeyError, match="not found in obs or var_names"):
            st.profile_features(ad, features=["NONEXISTENT"], groupby="arm")

    def test_missing_groupby_col_raises(self):
        ad = _make_trial_adata(4)
        with pytest.raises(KeyError, match="Missing required"):
            st.profile_features(ad, features=["G0"], groupby="nonexistent")

    def test_cell_weighted_behavior(self):
        """Verify cell-level aggregation (documented behavior)."""
        obs = pd.DataFrame({"group": ["A"] * 3 + ["B"] * 1})
        X = np.array([[10.0], [10.0], [10.0], [0.0]])
        ad = AnnData(X=X, obs=obs)
        ad.var_names = ["G0"]
        result = st.profile_features(ad, features=["G0"], groupby="group")
        assert result.loc["A", "G0"] == pytest.approx(10.0)
        assert result.loc["B", "G0"] == pytest.approx(0.0)
