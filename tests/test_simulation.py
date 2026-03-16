"""Tests for Monte Carlo simulation engine."""
import numpy as np
import pandas as pd
import pytest

import sctrial as st


class TestSimulateDidData:
    """Tests for simulate_did_data()."""

    def test_shape_and_structure(self):
        result = st.simulate_did_data(
            n_participants=20,
            n_genes=50,
            n_cells_per_participant=100,
            effect_sizes={"gene_0": 1.0, "gene_1": 0.5},
            seed=42,
        )
        assert isinstance(result, dict)
        assert "pseudobulk" in result
        assert "truth" in result
        pb = result["pseudobulk"]
        assert "participant" in pb.columns
        assert "visit" in pb.columns
        assert "arm" in pb.columns
        assert pb["participant"].nunique() == 20
        assert set(pb["visit"].unique()) == {"Pre", "Post"}
        assert set(pb["arm"].unique()) == {"Treated", "Control"}
        # 20 participants x 2 visits = 40 rows
        assert len(pb) == 40

    def test_null_effects_unbiased(self):
        """Under null (all effects=0), no systematic bias."""
        result = st.simulate_did_data(
            n_participants=40,
            n_genes=20,
            n_cells_per_participant=200,
            effect_sizes={},
            noise_sd=1.0,
            seed=42,
        )
        pb = result["pseudobulk"]
        gene_cols = [c for c in pb.columns if c.startswith("gene_")]
        for g in gene_cols[:5]:
            treated_post = pb[(pb["arm"] == "Treated") & (pb["visit"] == "Post")][g].mean()
            treated_pre = pb[(pb["arm"] == "Treated") & (pb["visit"] == "Pre")][g].mean()
            control_post = pb[(pb["arm"] == "Control") & (pb["visit"] == "Post")][g].mean()
            control_pre = pb[(pb["arm"] == "Control") & (pb["visit"] == "Pre")][g].mean()
            did = (treated_post - treated_pre) - (control_post - control_pre)
            assert abs(did) < 1.5, f"Null DiD too large for {g}: {did}"

    def test_signal_recoverable(self):
        """Embedded signal is recoverable in crude DiD."""
        result = st.simulate_did_data(
            n_participants=40,
            n_genes=10,
            n_cells_per_participant=200,
            effect_sizes={"gene_0": 2.0},
            noise_sd=0.5,
            seed=42,
        )
        pb = result["pseudobulk"]
        g = "gene_0"
        treated_post = pb[(pb["arm"] == "Treated") & (pb["visit"] == "Post")][g].mean()
        treated_pre = pb[(pb["arm"] == "Treated") & (pb["visit"] == "Pre")][g].mean()
        control_post = pb[(pb["arm"] == "Control") & (pb["visit"] == "Post")][g].mean()
        control_pre = pb[(pb["arm"] == "Control") & (pb["visit"] == "Pre")][g].mean()
        did = (treated_post - treated_pre) - (control_post - control_pre)
        assert did > 1.0, f"Signal not recovered: DiD={did}"

    def test_truth_dict_complete(self):
        """Truth dict has entries for all genes."""
        result = st.simulate_did_data(n_genes=30, effect_sizes={"gene_5": 0.8})
        assert len(result["truth"]) == 30
        assert result["truth"]["gene_5"] == 0.8
        assert result["truth"]["gene_0"] == 0.0

    def test_reproducibility(self):
        """Same seed produces identical output."""
        r1 = st.simulate_did_data(seed=123)
        r2 = st.simulate_did_data(seed=123)
        pd.testing.assert_frame_equal(r1["pseudobulk"], r2["pseudobulk"])


class TestRunMethodComparison:
    """Tests for run_method_comparison()."""

    def test_smoke_single_method(self):
        """Smoke test with single method and few iterations."""
        result = st.run_method_comparison(
            n_participants=20,
            n_genes=10,
            effect_sizes={"gene_0": 1.0},
            noise_sd=1.0,
            n_iterations=3,
            methods=["sctrial_did"],
            seed=42,
        )
        assert isinstance(result, pd.DataFrame)
        assert "method" in result.columns
        assert "gene" in result.columns
        assert "true_beta" in result.columns
        assert "estimated_beta" in result.columns
        assert "pvalue" in result.columns
        assert "iteration" in result.columns
        assert len(result) == 3 * 10  # 3 iterations x 10 genes

    def test_all_methods_run(self):
        """All three methods produce results."""
        result = st.run_method_comparison(
            n_participants=20,
            n_genes=5,
            effect_sizes={"gene_0": 1.0},
            noise_sd=1.0,
            n_iterations=2,
            methods=["sctrial_did", "wilcoxon", "pseudobulk_ols"],
            seed=42,
        )
        assert set(result["method"].unique()) == {
            "sctrial_did", "wilcoxon", "pseudobulk_ols"
        }
        assert len(result) == 2 * 5 * 3  # 2 iter x 5 genes x 3 methods

    def test_true_beta_populated(self):
        """true_beta reflects the specified effect sizes."""
        result = st.run_method_comparison(
            n_participants=20,
            n_genes=5,
            effect_sizes={"gene_0": 0.8},
            n_iterations=2,
            methods=["pseudobulk_ols"],
            seed=42,
        )
        g0 = result[result["gene"] == "gene_0"]
        assert (g0["true_beta"] == 0.8).all()
        g1 = result[result["gene"] == "gene_1"]
        assert (g1["true_beta"] == 0.0).all()

    def test_signal_detected_by_sctrial(self):
        """sctrial detects large signal across iterations."""
        result = st.run_method_comparison(
            n_participants=40,
            n_genes=5,
            effect_sizes={"gene_0": 2.0},
            noise_sd=0.5,
            n_iterations=10,
            methods=["sctrial_did"],
            seed=42,
        )
        g0 = result[result["gene"] == "gene_0"]
        # Most iterations should detect the signal
        sig = (g0["pvalue"] < 0.05).mean()
        assert sig >= 0.5, f"Power too low: {sig}"
