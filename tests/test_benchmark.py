"""Tests for the benchmark package.

Covers:
- simulator output contracts
- runner output contracts (standardised dict format)
- metrics computation
- an end-to-end tiny benchmark pass
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sctrial.benchmark.contracts import prepare_inputs
from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig, simulate_trial_v2

# A transcriptome large enough for the 2000-gene nested panel, small enough to
# run in a unit test. Panels are drawn FROM the transcriptome, so this is the
# feature universe, not the tested panel.
_TEST_GENES = 2500


def _cfg(**kwargs) -> TranscriptomeSimConfig:
    base = dict(
        n_per_arm=6,
        n_genes_transcriptome=_TEST_GENES,
        cells_per_pv_fixed=50,
        use_empirical_library=False,
        use_empirical_cells_per_pv=False,
        seed=42,
    )
    base.update(kwargs)
    return TranscriptomeSimConfig(**base)


# ---------------------------------------------------------------------------
# Simulator tests
# ---------------------------------------------------------------------------


class TestSimulator:
    """Simulator output contracts."""

    def _make_sim(self, **kwargs):
        return simulate_trial_v2(_cfg(effects={"gene_0": 0.5}, **kwargs))

    def test_two_arm_participant_count(self):
        sim = self._make_sim(design="two_arm")
        assert sim["adata"].obs["participant"].nunique() == 12

    def test_single_arm_participant_count(self):
        sim = self._make_sim(design="single_arm")
        assert sim["adata"].obs["participant"].nunique() == 6

    def test_returns_both_pseudobulk(self):
        sim = self._make_sim()
        assert "pseudobulk_means" in sim
        assert "pseudobulk_counts" in sim
        assert "pseudobulk" not in sim  # the ambiguous old key must not return

    def test_pseudobulk_counts_are_integers(self):
        sim = self._make_sim()
        counts = sim["pseudobulk_counts"][[f"gene_{i}" for i in range(5)]]
        assert (counts == counts.astype(int)).all().all()

    def test_pseudobulk_counts_larger_than_means(self):
        sim = self._make_sim()
        cols = [f"gene_{i}" for i in range(50)]
        ratio = (
            sim["pseudobulk_counts"][cols].values.mean()
            / (sim["pseudobulk_means"][cols].values.mean() + 1e-10)
        )
        assert ratio > 10, f"count/mean ratio={ratio:.1f}, expected ~n_cells"

    def test_adata_is_sparse_counts(self):
        from scipy import sparse

        sim = self._make_sim()
        assert sparse.issparse(sim["adata"].X)
        X = sim["adata"].X.toarray()
        assert (X >= 0).all()
        assert (X == X.astype(int)).all()

    def test_truth_dict(self):
        sim = self._make_sim()
        assert sim["truth"]["gene_0"] == 0.5

    def test_panels_are_nested(self):
        """Each larger panel must CONTAIN every smaller one.

        Without nesting, a panel-size effect and a gene-identity effect are
        confounded, and "progressive miscalibration with panel size" cannot be
        distinguished from a change in which genes were tested.
        """
        sim = self._make_sim()
        panels = sim["panels"]
        sizes = sorted(panels)
        for small, large in zip(sizes[:-1], sizes[1:]):
            assert set(panels[small]).issubset(set(panels[large])), (
                f"panel {small} is not a subset of panel {large}"
            )

    def test_library_totals_track_the_drawn_library_size(self):
        """Observed per-cell totals must match the latent L, not a multiple of it.

        With ``sum_g exp(alpha_g) = 1`` the per-cell total is
        ``L * E[exp(b + u)]``, which is ``exp((sd_b^2 + sd_u^2)/2)`` times too
        large unless the Jensen term is subtracted. That bug inflated every
        simulated library 1.7x.
        """
        sim = self._make_sim()
        obs = sim["adata"].obs
        ratio = obs["observed_library_size"].mean() / obs["true_library_size"].mean()
        assert 0.85 < ratio < 1.15, f"library inflation ratio {ratio:.3f}"

    def test_missing_visits(self):
        sim = self._make_sim(missing_rate=0.3)
        pv = sim["adata"].obs.groupby(["participant", "visit"]).size().unstack(fill_value=0)
        assert (pv.get("Post", 0) == 0).sum() > 0

    def test_imbalanced_arms(self):
        sim = self._make_sim(arm_ratio=(2, 8))
        pid_arms = sim["adata"].obs.groupby("participant")["arm"].first()
        assert (pid_arms == "Treated").sum() == 2
        assert (pid_arms == "Control").sum() == 8


# ---------------------------------------------------------------------------
# Runner contract tests
# ---------------------------------------------------------------------------


class TestRunnerContracts:
    """All runners must return the standardised dict."""

    @pytest.fixture
    def sim_data(self):
        return simulate_trial_v2(_cfg(effects={"gene_0": 0.5}))

    def _check_runner_output(self, result, gene_cols):
        assert isinstance(result, dict)
        for gene in gene_cols:
            assert gene in result, f"missing gene {gene}"
            r = result[gene]
            for key in ("beta", "pvalue", "converged", "failure_mode"):
                assert key in r, f"{gene}: missing {key}"
            if not np.isnan(r["pvalue"]):
                assert 0 <= r["pvalue"] <= 1, f"{gene}: pvalue={r['pvalue']}"

    def test_sctrial_did_contract(self, sim_data):
        from sctrial.benchmark.runners.sctrial_did import run

        panel = sim_data["panels"][50]
        inputs = prepare_inputs(sim_data, panel)
        result = run(inputs["participant_log1p_cpm"], panel, from_pseudobulk=True)
        self._check_runner_output(result, panel)

    def test_wilcoxon_paired_contract(self, sim_data):
        from sctrial.benchmark.runners.wilcoxon_paired import run

        panel = sim_data["panels"][50]
        inputs = prepare_inputs(sim_data, panel)
        result = run(inputs["participant_log1p_cpm"], panel)
        self._check_runner_output(result, panel)


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
            np.full(10, np.nan),
            np.full(10, np.nan),
            np.zeros(10),
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
        """Two Python methods, one null scenario, one iteration."""
        from sctrial.benchmark.orchestrator import _run_single_iteration

        scenario = {
            "name": "test_null",
            "description": "unit-test null",
            "panel_size": 50,
            "signal_fraction": 0.0,
            "architecture": "balanced",
            "magnitude": 0.5,
            "config_kwargs": {
                "design": "two_arm",
                "n_per_arm": 6,
                "n_genes_transcriptome": _TEST_GENES,
                "cells_per_pv_fixed": 50,
                "use_empirical_library": False,
                "use_empirical_cells_per_pv": False,
            },
        }
        rows = _run_single_iteration(
            ("test_null", 0, 42, scenario, ["sctrial_did", "wilcoxon_paired"])
        )
        df = pd.DataFrame(rows)

        assert len(df) == 2 * 50
        assert set(df["method"]) == {"sctrial_did", "wilcoxon_paired"}
        assert (df["injected_beta"] == 0.0).all()
        assert df["pvalue"].notna().all()
        # Under a true null the method-specific oracle must also be zero: an
        # oracle that drifts from zero when nothing was injected would silently
        # define a non-null truth for a null scenario.
        assert df["true_beta"].abs().max() < 1e-9
        assert (df["runtime_scope"] == "per_iteration").all()
