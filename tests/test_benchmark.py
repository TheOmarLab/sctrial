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
        # A 2,500-gene test transcriptome yields ~1,100 detectable genes,
        # so the production 2,000-gene panel does not fit. Stated here
        # rather than silently skipped.
        panel_sizes=(50, 200, 500),
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
            "n_signal": 0,
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
                "panel_sizes": (50, 200, 500),
            },
        }
        rows = _run_single_iteration(
            ("test_null", 0, 42, scenario, ["sctrial_did", "wilcoxon_paired"], None)
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


# ---------------------------------------------------------------------------
# Panel eligibility
# ---------------------------------------------------------------------------


class TestPanelEligibility:
    """Tested panels must contain genes a real analysis would report on."""

    def test_baseline_rates_match_the_simulator(self):
        """The standalone rate draw must equal what build_params uses.

        Panel selection happens BEFORE the effects are defined, so it reproduces
        alpha from the seed rather than simulating. If the two ever diverge, the
        panel would be chosen from one gene-rate vector and the data generated
        from another, and nothing downstream would notice.
        """
        from sctrial.benchmark.simulator_v2 import build_params, gene_baseline_rates

        cfg = _cfg()
        np.testing.assert_allclose(
            gene_baseline_rates(cfg), build_params(cfg)["alpha"], atol=0, rtol=0
        )

    def test_panels_avoid_undetectable_genes(self):
        """A panel drawn uniformly from the transcriptome is mostly noise.

        With TNBC's rate distribution a random 50-gene panel held ~9 detectable
        genes; filterByExpr discarded the rest and dreamlet returned 30-54%
        finite p-values. Every panel gene must clear the detection threshold.
        """
        from sctrial.benchmark.simulator_v2 import expected_counts_per_cell, nested_panels

        cfg = _cfg()
        exp_counts = expected_counts_per_cell(cfg)
        panels = nested_panels(cfg, rng=np.random.default_rng(1))
        for size, idx in panels.items():
            assert min(exp_counts[idx]) >= cfg.panel_min_mean_count, (
                f"panel {size} contains a gene below "
                f"{cfg.panel_min_mean_count} expected counts per cell"
            )

    def test_panel_larger_than_the_eligible_set_raises(self):
        """Padding a panel with undetectable genes must not be possible silently."""
        from sctrial.benchmark.simulator_v2 import eligible_panel_genes, nested_panels

        cfg = _cfg()
        n_eligible = len(eligible_panel_genes(cfg))
        with pytest.raises(ValueError, match="expected counts per cell"):
            nested_panels(cfg, sizes=(n_eligible + 1,), rng=np.random.default_rng(1))


def test_fixed_effect_ablation_rung_uses_a_full_rank_design():
    # The arm is constant within a participant. As a column beside the participant
    # indicators it left the estimate alone and enlarged the cluster-robust standard error.
    import statsmodels.formula.api as smf

    from sctrial.benchmark.ablation import _pseudobulk_fe_crse

    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        arm, base = ("Treated" if i < 6 else "Control"), rng.normal()
        for visit in ("Pre", "Post"):
            lift = 0.9 if (arm == "Treated" and visit == "Post") else 0.0
            rows.append({"participant": f"p{i:02d}", "visit": visit, "arm": arm,
                         "g": base + lift + rng.normal(0, 0.5)})
    pb = pd.DataFrame(rows)
    out = _pseudobulk_fe_crse(pb, ["g"])["g"]

    pb["t"] = (pb["visit"] == "Post").astype(int)
    pb["a"] = (pb["arm"] == "Treated").astype(int)
    ref = smf.ols("g ~ t + t:a + C(participant)", data=pb).fit(
        cov_type="cluster", cov_kwds={"groups": pb["participant"]}
    )
    assert ref.model.exog.shape[1] == np.linalg.matrix_rank(ref.model.exog)
    assert out["beta"] == pytest.approx(ref.params["t:a"], rel=1e-9)
    assert out["pvalue"] == pytest.approx(ref.pvalues["t:a"], rel=1e-9)
    assert (out["ci_lo"], out["ci_hi"]) == pytest.approx(tuple(ref.conf_int().loc["t:a"]), rel=1e-9)


# ---------------------------------------------------------------------------
# A gene without a p-value was not tested
# ---------------------------------------------------------------------------


def _participant_frame(n_per_arm: int = 6, seed: int = 0) -> pd.DataFrame:
    """Participant by visit outcomes in the vocabulary of the runners.

    ``g_ok`` varies; ``g_flat`` is the same value in every row, so no participant changes
    and a test of the changes has no variance to divide by.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(2 * n_per_arm):
        arm, base = ("Treated" if i < n_per_arm else "Control"), rng.normal()
        for visit in ("Pre", "Post"):
            lift = 0.9 if (arm == "Treated" and visit == "Post") else 0.0
            rows.append({"participant": f"p{i:02d}", "visit": visit, "arm": arm,
                         "g_ok": base + lift + rng.normal(0, 0.5), "g_flat": 3.0})
    return pd.DataFrame(rows)


def test_a_row_without_a_p_value_is_not_a_converged_fit():
    from sctrial.benchmark.runners import finalize_results

    nan = float("nan")
    rows = {
        "tested": {"beta": 0.4, "pvalue": 0.03, "ci_lo": 0.1, "ci_hi": 0.7, "converged": True,
                   "failure_mode": None},
        "tested_no_flags": {"beta": 0.4, "pvalue": 0.03},
        "estimates_without_convergence": {"beta": 0.4, "pvalue": 0.03, "converged": False,
                                          "failure_mode": "convergence"},
        "no_test": {"beta": 0.0, "pvalue": nan, "ci_lo": 0.0, "ci_hi": 0.0, "converged": True,
                    "failure_mode": None},
        "no_test_no_flags": {"beta": nan, "pvalue": nan},
        "timed_out": {"beta": nan, "pvalue": nan, "converged": False, "failure_mode": "timeout"},
        "no_p_key": {"beta": 0.1, "converged": True, "failure_mode": None},
        "p_none": {"beta": 0.1, "pvalue": None, "converged": True, "failure_mode": None},
        "p_infinite": {"beta": 0.1, "pvalue": float("inf"), "converged": True, "failure_mode": None},
    }
    expected = {
        "tested": (True, None),
        "tested_no_flags": (True, None),
        "estimates_without_convergence": (False, "convergence"),
        "no_test": (False, "numerical"),
        "no_test_no_flags": (False, "numerical"),
        "timed_out": (False, "timeout"),
        "no_p_key": (False, "numerical"),
        "p_none": (False, "numerical"),
        "p_infinite": (False, "numerical"),
    }
    flags_before = {g: (r.get("converged"), r.get("failure_mode")) for g, r in rows.items()}

    out = finalize_results(rows)

    assert {g: (r["converged"], r["failure_mode"]) for g, r in out.items()} == expected
    assert {g: (r.get("converged"), r.get("failure_mode")) for g, r in rows.items()} == flags_before
    assert out["tested"] == rows["tested"]
    assert "ci_lo" not in out["tested_no_flags"]
    # no test, no interval: a standard error of zero gave the interval [0, 0]
    assert out["no_test"]["beta"] == 0.0
    for gene, row in out.items():
        if expected[gene][0] is False and gene != "estimates_without_convergence":
            assert np.isnan([row["pvalue"], row["ci_lo"], row["ci_hi"]]).all(), gene


@pytest.mark.parametrize("design_type", ["two_arm", "single_arm"])
def test_dispatcher_reports_a_gene_without_variation_as_not_tested(design_type):
    # Every change of g_flat is zero: the t-test returns a missing p-value without raising,
    # and the row said `converged: True, failure_mode: None`.
    from sctrial.benchmark.endpoints import replicate_endpoints
    from sctrial.benchmark.orchestrator import _dispatch_method

    frame = _participant_frame()
    genes = ["g_ok", "g_flat"]
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _dispatch_method("sctrial_did", {"participant_log1p_cpm": frame, "panel_genes": genes},
                                   design_type=design_type)

    assert np.isfinite(res["g_ok"]["pvalue"])
    assert (res["g_ok"]["converged"], res["g_ok"]["failure_mode"]) == (True, None)
    assert np.isnan(res["g_flat"]["pvalue"])
    assert (res["g_flat"]["converged"], res["g_flat"]["failure_mode"]) == (False, "numerical")

    table = pd.DataFrame(
        [{"gene": g, "pvalue": res[g]["pvalue"], "is_signal": False, "converged": res[g]["converged"]}
         for g in genes]
    )
    endpoints = replicate_endpoints(table)
    assert endpoints["evaluability"] == 0.5 and endpoints["convergence_rate"] == 0.5


def test_ablation_counts_a_gene_without_a_test_as_a_failure():
    # The model rungs returned an estimate, a p-value and an interval, and no flag, so the
    # failure rates of an iteration were zero whatever happened in it. For a gene without
    # variation their p-value was made of rounding error (0.57, 0.60 and 0.43 here).
    import warnings

    import anndata as ad
    import statsmodels.formula.api as smf

    from sctrial.benchmark.ablation import run_ablation
    from sctrial.benchmark.metrics import summarize_iteration

    frame = _participant_frame()
    genes = ["g_ok", "g_flat"]
    variants = ["cell_ols", "pb_ols", "pb_fe", "pb_fe_crse", "sctrial_full"]
    # Cells for the cell-level rung: g_flat is never detected, so its outcome is zero in
    # every cell.
    rng = np.random.default_rng(1)
    obs = frame[["participant", "visit", "arm"]].loc[frame.index.repeat(8)].reset_index(drop=True)
    obs.index = [f"c{i}" for i in range(len(obs))]
    counts = np.column_stack([rng.poisson(6, len(obs)), np.zeros(len(obs))]).astype(float)
    cells = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=genes))
    inputs = {"participant_log1p_cpm": frame, "cell_counts": cells,
              "cell_lib_size": rng.integers(900, 1100, len(obs)).astype(float)}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        results = run_ablation(inputs, genes, variants=variants)

    # a gene that varies keeps the numbers of its model
    ref = frame.assign(t=(frame["visit"] == "Post").astype(float), a=(frame["arm"] == "Treated").astype(float))
    pooled = smf.ols("g_ok ~ t + a + t:a", data=ref).fit()
    within = smf.ols("g_ok ~ t + t:a + C(participant)", data=ref).fit()
    for variant, fit in (("pb_ols", pooled), ("pb_fe", within)):
        row = results[variant]["g_ok"]
        assert row["beta"] == pytest.approx(fit.params["t:a"], rel=1e-9), variant
        assert row["pvalue"] == pytest.approx(fit.pvalues["t:a"], rel=1e-9), variant
        assert (row["ci_lo"], row["ci_hi"]) == pytest.approx(tuple(fit.conf_int().loc["t:a"]), rel=1e-9), variant

    for variant in variants:
        rows = results[variant]
        assert (rows["g_ok"]["converged"], rows["g_ok"]["failure_mode"]) == (True, None), variant
        flat = rows["g_flat"]
        assert np.isnan([flat["pvalue"], flat["ci_lo"], flat["ci_hi"]]).all(), variant
        assert flat["beta"] == pytest.approx(0.0, abs=1e-9), variant
        assert (flat["converged"], flat["failure_mode"]) == (False, "numerical"), variant
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            metrics = summarize_iteration(rows, {g: 0.0 for g in genes}, set())
        assert metrics["total_failure_rate"] == 0.5, variant
        assert metrics["numerical_failure_rate"] == 0.5, variant
        assert metrics["convergence_failure_rate"] == 0.0 and metrics["timeout_failure_rate"] == 0.0, variant
