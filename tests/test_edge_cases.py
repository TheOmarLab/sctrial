"""Edge case tests for sctrial to ensure robust error handling."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

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
        obs = pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2", "P2"] * 5,
                "visit": (["V1", "V2"] * 2) * 5,
                "arm": (["Treated"] * 2 + ["Control"] * 2) * 5,
                "celltype": ["A"] * 20,  # Only one cell type
            }
        )
        adata = AnnData(X=np.zeros((20, 1)), obs=obs)

        res = st.abundance_did(adata, trial_design, visits=("V1", "V2"), min_units=2)
        # Should return results for the single celltype
        assert len(res) <= 1


class TestSingleParticipant:
    """Test handling of insufficient sample sizes."""

    def test_did_fit_single_participant(self, trial_design):
        """DiD should return NaN for insufficient participants."""
        obs = pd.DataFrame(
            {
                "participant_id": ["P1", "P1"],
                "visit": ["V1", "V2"],
                "arm": ["Treated", "Treated"],
            }
        )
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
        obs = pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2", "P2"],
                "visit": ["V1", "V2", "V1", "V2"],
                "arm": ["Treated", "Treated", "Control", "Control"],
                "celltype": ["A", "A", "A", "A"],
            }
        )
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
        # Zero-variance feature should produce NaN beta
        assert len(res) == 1
        assert pd.isna(res.iloc[0]["beta_time"])


class TestInputValidation:
    """Input validation edge cases."""

    def test_score_gene_sets_invalid_inputs(self):
        adata = AnnData(X=np.ones((3, 2)))
        with pytest.raises(ValueError):
            st.score_gene_sets(adata, {}, method="zmean")
        with pytest.raises(ValueError):
            st.score_gene_sets(adata, {"A": ["G1"]}, min_genes=0)
        with pytest.raises(KeyError):
            st.score_gene_sets(adata, {"A": ["G1"]}, layer="missing")

    def test_power_and_sample_size_validation(self):
        with pytest.raises(ValueError):
            st.power_did(n_per_group=1.5, effect_size=0.5)
        with pytest.raises(ValueError):
            st.sample_size_did(effect_size=-0.5)

    def test_did_fit_empty_df(self):
        with pytest.raises(ValueError):
            st.did_fit(pd.DataFrame(), y="y", unit="id", time="t", arm_bin="arm")


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
        obs = pd.DataFrame(
            {
                "participant_id": ["P1"],
                "visit": ["V1"],
                "arm": ["Unknown"],  # Not "Treated" or "Control"
            }
        )
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


class TestWeighting:
    """OLS is the default; weighting by cell count is an explicit opt-in only.

    The fixture has treated deltas (1.0, 2.0) and control deltas (0.2, 0.1), with
    100 and 10 cells per participant. With participant fixed effects over two
    visits and weights constant within a participant, the DiD reduces to a
    difference of (weighted) mean deltas, so both answers are exact:
    OLS 1.35, and WLS by cell count (120/110 - 21/110) = 0.90.
    """

    OLS_BETA = 1.35
    WLS_BETA = 0.90

    @staticmethod
    def _pv_frame():
        return pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
                "visit_num": [0, 1, 0, 1, 0, 1, 0, 1],
                "arm_bin": [1, 1, 1, 1, 0, 0, 0, 0],
                "outcome": [1.0, 2.0, 1.0, 3.0, 1.0, 1.2, 1.0, 1.1],
                "n_cells": [100, 100, 10, 10, 100, 100, 10, 10],
            }
        )

    @staticmethod
    def _fit(df, **kw):
        from sctrial.stats.did import did_fit

        return did_fit(
            df, y="outcome", unit="participant_id", time="visit_num",
            arm_bin="arm_bin", standardize=False, **kw,
        )

    @staticmethod
    def _cell_adata():
        """Cell-level data whose participant-visit means equal ``_pv_frame``."""
        spec = {
            "P1": ("T", 1.0, 2.0, 100), "P2": ("T", 1.0, 3.0, 10),
            "P3": ("C", 1.0, 1.2, 100), "P4": ("C", 1.0, 1.1, 10),
        }
        rows = []
        for pid, (arm, pre, post, n) in spec.items():
            for visit, value in (("Pre", pre), ("Post", post)):
                rows += [{"participant": pid, "visit": visit, "arm": arm, "score": value}] * n
        obs = pd.DataFrame(rows)
        obs.index = obs.index.astype(str)
        design = st.TrialDesign(
            participant_col="participant", visit_col="visit", arm_col="arm",
            arm_treated="T", arm_control="C", celltype_col=None,
        )
        return AnnData(X=np.zeros((len(obs), 1)), obs=obs), design

    def test_ols_is_default_even_when_n_cells_present(self):
        assert self._fit(self._pv_frame())["beta_DiD"] == pytest.approx(self.OLS_BETA, abs=1e-10)

    def test_weight_col_opts_into_wls(self):
        res = self._fit(self._pv_frame(), weight_col="n_cells")
        assert res["beta_DiD"] == pytest.approx(self.WLS_BETA, abs=1e-10)

    def test_weight_col_rejects_nonpositive_weights(self):
        df = self._pv_frame()
        df.loc[0, "n_cells"] = 0
        with pytest.raises(ValueError, match="finite and positive"):
            self._fit(df, weight_col="n_cells")

    def test_two_visit_ols_equals_participant_first_difference(self):
        """Equal weighting makes the FE estimate the participant first difference.

        This is the estimator the frozen benchmark validated, so the applied
        estimator and the benchmarked one must agree exactly.
        """
        rng = np.random.default_rng(0)
        n = 30
        arm = np.repeat([1, 0], n // 2)
        pre, post = rng.normal(size=n), rng.normal(size=n) + 0.4 * arm
        df = pd.DataFrame(
            {
                "participant_id": np.repeat(np.arange(n), 2),
                "visit_num": np.tile([0, 1], n),
                "arm_bin": np.repeat(arm, 2),
                "outcome": np.column_stack([pre, post]).ravel(),
                "n_cells": np.repeat(rng.integers(150, 27_000, size=n), 2),
            }
        )
        delta = post - pre
        expected = delta[arm == 1].mean() - delta[arm == 0].mean()
        assert self._fit(df)["beta_DiD"] == pytest.approx(expected, abs=1e-10)

    def test_bootstrap_refits_with_the_estimator_of_the_original_fit(self, monkeypatch):
        """OLS fits must be bootstrapped with OLS, WLS fits with WLS.

        statsmodels' OLS is a WLS subclass carrying all-ones weights, so the mere
        presence of ``model.weights`` cannot tell them apart.
        """
        import statsmodels.api as sm

        real_ols, real_wls = sm.OLS, sm.WLS
        used: list[str] = []

        def _ols(*a, **k):
            used.append("OLS")
            return real_ols(*a, **k)

        def _wls(*a, **k):
            used.append("WLS")
            return real_wls(*a, **k)

        monkeypatch.setattr(sm, "OLS", _ols)
        monkeypatch.setattr(sm, "WLS", _wls)

        self._fit(self._pv_frame(), use_bootstrap=True, n_boot=20, seed=1)
        assert used and set(used) == {"OLS"}, used

        used.clear()
        self._fit(self._pv_frame(), use_bootstrap=True, n_boot=20, seed=1, weight_col="n_cells")
        assert used and set(used) == {"WLS"}, used

    def test_did_table_and_its_variants_forward_the_weighting(self):
        adata, design = self._cell_adata()
        kw = dict(features=["score"], design=design, visits=("Pre", "Post"), standardize=False)

        def beta(res):
            return res.set_index("feature").loc["score", "beta_DiD"]

        assert beta(st.did_table(adata, **kw)) == pytest.approx(self.OLS_BETA, abs=1e-10)
        assert beta(st.did_table(adata, weight_col="n_cells", **kw)) == pytest.approx(
            self.WLS_BETA, abs=1e-10
        )

        cfg = st.DiDConfig(standardize=False, weight_col="n_cells")
        cfg_res = st.did_table(adata, features=["score"], design=design,
                               visits=("Pre", "Post"), config=cfg)
        assert beta(cfg_res) == pytest.approx(self.WLS_BETA, abs=1e-10)

        for n_jobs in (1, 2):
            par = st.did_table_parallel(adata, weight_col="n_cells", n_jobs=n_jobs, **kw)
            assert beta(par) == pytest.approx(self.WLS_BETA, abs=1e-10), n_jobs
