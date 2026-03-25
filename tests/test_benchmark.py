"""Tests for the benchmark package.

Covers:
- Simulator output contracts
- Runner output contracts (standardized dict format)
- End-to-end tiny benchmark pass
- Metrics computation
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sctrial.benchmark.simulator import SimulationConfig, simulate_trial

# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------

class TestSimulator:
    """Test simulator output contracts."""

    def _make_sim(self, **kwargs):
        cfg = SimulationConfig(
            n_per_arm=6, n_genes=5, mean_cells_per_visit=50,
            effects={"gene_0": 0.5}, seed=42, **kwargs,
        )
        return simulate_trial(cfg)

    def test_two_arm_participant_count(self):
        """n_per_arm=6 with two arms should give 12 total participants."""
        sim = self._make_sim(design="two_arm")
        n_pids = sim["adata"].obs["participant"].nunique()
        assert n_pids == 12, f"Expected 12 participants, got {n_pids}"

    def test_single_arm_participant_count(self):
        """n_per_arm=6 with single arm gives 6 participants."""
        sim = self._make_sim(design="single_arm")
        n_pids = sim["adata"].obs["participant"].nunique()
        assert n_pids == 6

    def test_returns_both_pseudobulk(self):
        """simulate_trial returns both pseudobulk_means and pseudobulk_counts."""
        sim = self._make_sim()
        assert "pseudobulk_means" in sim
        assert "pseudobulk_counts" in sim
        assert "pseudobulk" not in sim  # old key should NOT be present

    def test_pseudobulk_counts_are_integers(self):
        """Summed count pseudobulk should have integer values."""
        sim = self._make_sim()
        gene_cols = [f"gene_{i}" for i in range(5)]
        counts = sim["pseudobulk_counts"][gene_cols]
        assert (counts == counts.astype(int)).all().all()

    def test_pseudobulk_counts_larger_than_means(self):
        """Summed counts should be much larger than means (sum vs avg)."""
        sim = self._make_sim()
        gene_cols = [f"gene_{i}" for i in range(5)]
        means = sim["pseudobulk_means"][gene_cols].values
        counts = sim["pseudobulk_counts"][gene_cols].values
        # Counts = sum over ~50 cells; means = mean over ~50 cells
        # So counts should be ~50x larger
        ratio = counts.mean() / (means.mean() + 1e-10)
        assert ratio > 10, f"Count/mean ratio={ratio:.1f}, expected >10"

    def test_adata_is_sparse_counts(self):
        """Cell-level X should be sparse integer counts."""
        from scipy import sparse
        sim = self._make_sim()
        assert sparse.issparse(sim["adata"].X)
        X = sim["adata"].X.toarray()
        assert (X >= 0).all()
        assert (X == X.astype(int)).all()

    def test_truth_dict(self):
        """Truth dict should have correct effects."""
        sim = self._make_sim()
        assert sim["truth"]["gene_0"] == 0.5
        assert sim["truth"]["gene_1"] == 0.0

    def test_missing_visits(self):
        """missing_rate > 0 should drop some post visits."""
        sim = self._make_sim(missing_rate=0.3)
        pv = sim["adata"].obs.groupby(["participant", "visit"]).size().unstack(fill_value=0)
        n_missing = (pv.get("Post", 0) == 0).sum()
        assert n_missing > 0, "Expected some missing post visits"

    def test_imbalanced_arms(self):
        """arm_ratio should produce unequal participant counts per arm."""
        sim = self._make_sim(arm_ratio=(2, 8))
        pid_arms = sim["adata"].obs.groupby("participant")["arm"].first()
        n_treat = (pid_arms == "Treated").sum()
        n_ctrl = (pid_arms == "Control").sum()
        assert n_treat == 2, f"Expected 2 treated participants, got {n_treat}"
        assert n_ctrl == 8, f"Expected 8 control participants, got {n_ctrl}"


# ---------------------------------------------------------------------------
# Runner contract tests
# ---------------------------------------------------------------------------

class TestRunnerContracts:
    """Test that all runners return standardized dicts."""

    @pytest.fixture
    def sim_data(self):
        cfg = SimulationConfig(
            n_per_arm=6, n_genes=5, mean_cells_per_visit=50,
            effects={"gene_0": 0.5}, seed=42,
        )
        return simulate_trial(cfg)

    def _check_runner_output(self, result, gene_cols):
        """Verify a runner's output matches the contract."""
        assert isinstance(result, dict)
        for gene in gene_cols:
            assert gene in result, f"Missing gene {gene}"
            r = result[gene]
            assert "beta" in r
            assert "pvalue" in r
            assert "converged" in r
            assert "failure_mode" in r
            # pvalue should be in [0, 1] or NaN
            if not np.isnan(r["pvalue"]):
                assert 0 <= r["pvalue"] <= 1, f"{gene}: pvalue={r['pvalue']}"

    def test_sctrial_fe_contract(self, sim_data):
        from sctrial.benchmark.runners.sctrial_fe import run
        gene_cols = [f"gene_{i}" for i in range(5)]
        result = run(sim_data["adata"], gene_cols)
        self._check_runner_output(result, gene_cols)

    def test_wilcoxon_paired_contract(self, sim_data):
        from sctrial.benchmark.runners.wilcoxon_paired import run
        gene_cols = [f"gene_{i}" for i in range(5)]
        result = run(sim_data["pseudobulk_means"], gene_cols)
        self._check_runner_output(result, gene_cols)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test metric computations."""

    def test_fpr_on_uniform(self):
        """FPR on uniform p-values should be ~0.05."""
        from sctrial.benchmark.metrics import compute_fpr
        rng = np.random.default_rng(42)
        pvals = rng.uniform(0, 1, size=10000)
        result = compute_fpr(pvals)
        assert abs(result["fpr"] - 0.05) < 0.01

    def test_fpr_on_zeros(self):
        """FPR on all-significant should be 1.0."""
        from sctrial.benchmark.metrics import compute_fpr
        pvals = np.full(100, 0.001)
        result = compute_fpr(pvals)
        assert result["fpr"] == 1.0

    def test_topk_jaccard_identical(self):
        """Identical rankings should give Jaccard = 1.0."""
        from sctrial.benchmark.metrics import compute_topk_jaccard
        s = pd.Series({"a": 0.01, "b": 0.02, "c": 0.5, "d": 0.9})
        assert compute_topk_jaccard(s, s, k=2) == 1.0

    def test_topk_jaccard_disjoint(self):
        """Completely different top-k should give Jaccard = 0.0."""
        from sctrial.benchmark.metrics import compute_topk_jaccard
        s1 = pd.Series({"a": 0.01, "b": 0.02, "c": 0.5, "d": 0.9})
        s2 = pd.Series({"a": 0.9, "b": 0.8, "c": 0.01, "d": 0.02})
        assert compute_topk_jaccard(s1, s2, k=2) == 0.0

    def test_ci_coverage_returns_none_for_nans(self):
        """CI coverage should return None when no intervals exist."""
        from sctrial.benchmark.metrics import compute_ci_coverage
        result = compute_ci_coverage(
            np.full(10, np.nan), np.full(10, np.nan), np.zeros(10),
        )
        assert result is None

    def test_failure_modes_split(self):
        """Failure rates should split by mode correctly."""
        from sctrial.benchmark.metrics import compute_failure_rates
        results = [
            {"failure_mode": None},
            {"failure_mode": "convergence"},
            {"failure_mode": "numerical"},
            {"failure_mode": None},
        ]
        rates = compute_failure_rates(results)
        assert rates["convergence_rate"] == 0.25
        assert rates["numerical_rate"] == 0.25
        assert rates["total_failure_rate"] == 0.5


# ---------------------------------------------------------------------------
# End-to-end mini benchmark
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Tiny end-to-end benchmark pass."""

    def test_mini_benchmark(self):
        """Run 2 iterations of 1 scenario with 2 methods."""
        from sctrial.benchmark.orchestrator import _run_single_iteration

        args = (
            "test_null",  # scenario name
            0,            # iteration
            42,           # seed
            {             # config_kwargs
                "design": "two_arm",
                "n_per_arm": 6,
                "n_genes": 5,
                "effects": {},
                "mean_cells_per_visit": 50,
            },
            ["sctrial_fe", "wilcoxon_paired"],  # methods
        )

        rows = _run_single_iteration(args)
        df = pd.DataFrame(rows)

        assert len(df) == 2 * 5  # 2 methods × 5 genes
        assert set(df["method"]) == {"sctrial_fe", "wilcoxon_paired"}
        assert (df["true_beta"] == 0.0).all()  # null scenario
        assert df["pvalue"].notna().all()
