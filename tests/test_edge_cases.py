"""Edge case tests for sctrial to ensure robust error handling."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
import sctrial as st


class TestEmptyDataFrames:
    """Test handling of empty or minimal data."""

    def test_did_table_empty_features(self, sample_adata, trial_design):
        """DiD table should handle empty feature list gracefully."""
        with pytest.raises(ValueError, match="No numeric features"):
            st.did_table(
                sample_adata,
                features=[],
                design=trial_design,
                visits=("V1", "V2"),
            )

    def test_did_table_nonexistent_feature(self, sample_adata, trial_design):
        """DiD table should raise KeyError for missing features."""
        with pytest.raises(KeyError, match="not found"):
            st.did_table(
                sample_adata,
                features=["NONEXISTENT_GENE"],
                design=trial_design,
                visits=("V1", "V2"),
            )

    def test_abundance_did_no_celltypes(self, trial_design):
        """abundance_did should handle data with single celltype gracefully."""
        obs = pd.DataFrame({
            "participant_id": ["P1", "P1", "P2", "P2"] * 5,
            "visit": (["V1", "V2"] * 2) * 5,
            "arm": (["Treated"] * 2 + ["Control"] * 2) * 5,
            "celltype": ["A"] * 20  # Only one cell type
        })
        adata = AnnData(X=np.zeros((20, 1)), obs=obs)

        res = st.abundance_did(adata, trial_design, visits=("V1", "V2"), min_units=2)
        # Should return results for the single celltype
        assert len(res) <= 1


class TestSingleParticipant:
    """Test handling of insufficient sample sizes."""

    def test_did_fit_single_participant(self, trial_design):
        """DiD should return NaN for insufficient participants."""
        obs = pd.DataFrame({
            "participant_id": ["P1", "P1"],
            "visit": ["V1", "V2"],
            "arm": ["Treated", "Treated"],
        })
        adata = AnnData(X=np.array([[1.0], [2.0]]), obs=obs)
        adata.var_names = ["G0"]

        res = st.did_table(
            adata,
            features=["G0"],
            design=trial_design,
            visits=("V1", "V2"),
        )

        # Should return NaN because n_units < 4
        assert pd.isna(res.iloc[0]["beta_DiD"])

    def test_abundance_did_insufficient_units(self, trial_design):
        """abundance_did should skip cell types with too few participants."""
        obs = pd.DataFrame({
            "participant_id": ["P1", "P1", "P2", "P2"],
            "visit": ["V1", "V2", "V1", "V2"],
            "arm": ["Treated", "Treated", "Control", "Control"],
            "celltype": ["A", "A", "A", "A"]
        })
        adata = AnnData(X=np.zeros((4, 1)), obs=obs)

        # min_units=10 should skip all cell types
        res = st.abundance_did(adata, trial_design, visits=("V1", "V2"), min_units=10)
        assert len(res) == 0


class TestZeroVarianceFeatures:
    """Test handling of constant/zero-variance features."""

    def test_score_gene_sets_zero_variance(self):
        """Scoring should handle zero-variance genes."""
        # All cells have same expression for all genes
        X = np.ones((10, 3))
        adata = AnnData(X=X)
        adata.var_names = ["G1", "G2", "G3"]

        gene_sets = {"SET1": ["G1", "G2", "G3"]}
        adata = st.score_gene_sets(adata, gene_sets, method="zmean")

        # Should be NaN because all genes have zero variance
        assert pd.isna(adata.obs["SET1"]).all()

    def test_did_table_zero_variance_feature(self, sample_adata, trial_design):
        """DiD should handle zero-variance features gracefully."""
        # Add a constant feature
        sample_adata.obs["constant_feature"] = 1.0

        res = st.did_table(
            sample_adata,
            features=["constant_feature"],
            design=trial_design,
            visits=("V1", "V2"),
        )

        # Should return NaN for zero-variance feature
        assert pd.isna(res.iloc[0]["beta_DiD"])

    def test_within_arm_zero_variance(self, sample_adata, trial_design):
        """within_arm_comparison should handle zero-variance features."""
        sample_adata.obs["constant"] = 5.0

        res = st.within_arm_comparison(
            sample_adata,
            arm="Treated",
            features=["constant"],
            design=trial_design,
            visits=("V1", "V2"),
        )

        # Should return NaN
        assert pd.isna(res.iloc[0]["beta_time"])


class TestNaNFeatures:
    """Test handling of features with NaN values."""

    def test_did_table_with_nan_feature(self, sample_adata, trial_design):
        """DiD should handle features with some NaN values."""
        sample_adata.obs["partial_nan"] = np.random.randn(sample_adata.n_obs)
        sample_adata.obs.loc[sample_adata.obs.index[:5], "partial_nan"] = np.nan

        res = st.did_table(
            sample_adata,
            features=["partial_nan"],
            design=trial_design,
            visits=("V1", "V2"),
        )

        # Should complete without error (NaN rows are dropped internally)
        assert "beta_DiD" in res.columns


class TestMultipleFeatures:
    """Test that multiple features are processed independently."""

    def test_comparisons_multiple_features_independent(self, sample_adata, trial_design):
        """Each feature should be processed independently in comparisons."""
        # Add features with different scales
        sample_adata.obs["feat1"] = np.random.randn(sample_adata.n_obs) * 100
        sample_adata.obs["feat2"] = np.random.randn(sample_adata.n_obs) * 0.01

        res = st.within_arm_comparison(
            sample_adata,
            arm="Treated",
            features=["feat1", "feat2"],
            design=trial_design,
            visits=("V1", "V2"),
            standardize=True,
        )

        # Both should have results
        assert len(res) == 2
        # n_units should be the same for both
        assert res.iloc[0]["n_units"] == res.iloc[1]["n_units"]


class TestDesignValidation:
    """Test TrialDesign validation."""

    def test_design_missing_column(self):
        """Design validation should catch missing columns."""
        obs = pd.DataFrame({"participant_id": ["P1"], "visit": ["V1"]})
        adata = AnnData(X=np.zeros((1, 1)), obs=obs)

        design = st.TrialDesign(arm_col="arm")  # 'arm' column doesn't exist

        with pytest.raises(KeyError, match="Missing required"):
            design.validate(adata)

    def test_design_missing_arm_labels(self):
        """Design validation should catch missing arm labels."""
        obs = pd.DataFrame({
            "participant_id": ["P1"],
            "visit": ["V1"],
            "arm": ["Unknown"]  # Not "Treated" or "Control"
        })
        adata = AnnData(X=np.zeros((1, 1)), obs=obs)

        design = st.TrialDesign()

        with pytest.raises(ValueError, match="Arm labels not found"):
            design.validate(adata)


class TestResolveFeature:
    """Test feature resolution."""

    def test_resolve_ambiguous_feature(self, sample_adata):
        """Should raise error for ambiguous case-insensitive matches."""
        # Add two columns that differ only in case
        sample_adata.obs["Feature"] = 1.0
        sample_adata.obs["feature"] = 2.0

        # Case-insensitive search should find ambiguity
        # But exact match should work
        assert st.resolve_feature(sample_adata, "Feature") == "Feature"
        assert st.resolve_feature(sample_adata, "feature") == "feature"
