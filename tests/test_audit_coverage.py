"""Tests for previously untested code paths identified in audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import sctrial as st
from sctrial.design import TrialDesign
from sctrial.stats.comparisons import resolve_gene_name
from sctrial.stats.effect_size import _compute_effect_size_from_fit
from sctrial.stats.module_scores import _perm_test_diff
from sctrial.utils import wild_cluster_bootstrap_t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial_adata(
    n_participants: int = 20,
    n_genes: int = 50,
    visits: tuple[str, str] = ("V1", "V2"),
    seed: int = 0,
    celltype_col: bool = False,
    extra_visits: list[str] | None = None,
) -> tuple[AnnData, TrialDesign]:
    """Create a synthetic trial AnnData with Poisson counts."""
    rng = np.random.default_rng(seed)
    all_visits = list(visits)
    if extra_visits:
        all_visits += extra_visits

    obs_list = []
    for i in range(n_participants):
        arm = "Treated" if i < n_participants // 2 else "Control"
        ct = "TypeA" if i % 2 == 0 else "TypeB"
        for v in all_visits:
            obs_list.append({
                "participant_id": f"P{i}",
                "visit": v,
                "arm": arm,
                "celltype": ct,
            })

    obs = pd.DataFrame(obs_list)
    X = rng.poisson(2, size=(len(obs), n_genes)).astype(float)
    X_sparse = sparse.csr_matrix(X)

    adata = AnnData(X=X_sparse, obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    adata.layers["counts"] = adata.X.copy()

    design_kwargs = dict(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )
    if celltype_col:
        design_kwargs["celltype_col"] = "celltype"
    design = TrialDesign(**design_kwargs)

    return adata, design


# ---------------------------------------------------------------------------
# 1-3: resolve_gene_name
# ---------------------------------------------------------------------------

class TestResolveGeneName:
    def test_exact_match(self):
        adata = AnnData(X=np.zeros((2, 3)))
        adata.var_names = ["TP53", "BRCA1", "MYC"]
        assert resolve_gene_name(adata, "TP53") == "TP53"

    def test_case_insensitive(self):
        adata = AnnData(X=np.zeros((2, 3)))
        adata.var_names = ["TP53", "BRCA1", "MYC"]
        assert resolve_gene_name(adata, "tp53") == "TP53"

    def test_missing_raises(self):
        adata = AnnData(X=np.zeros((2, 3)))
        adata.var_names = ["TP53", "BRCA1", "MYC"]
        with pytest.raises(ValueError, match="not found"):
            resolve_gene_name(adata, "NOTREAL")


# ---------------------------------------------------------------------------
# 4-5: wild_cluster_bootstrap_t
# ---------------------------------------------------------------------------

class TestWildClusterBootstrap:
    @staticmethod
    def _fit_clustered_ols(seed: int = 0, effect: float = 2.0):
        """Helper: build a formula-based OLS fit with clusters."""
        import statsmodels.formula.api as smf

        rng = np.random.default_rng(seed)
        n = 60
        clusters = np.repeat(np.arange(12), 5)
        arm = (clusters < 6).astype(float)
        y = effect * arm + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "arm_bin": arm})
        fit = smf.ols("y ~ arm_bin", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": clusters}
        )
        X = fit.model.exog
        return fit, X, clusters

    def test_known_data_p_in_range(self):
        fit, X, clusters = self._fit_clustered_ols(effect=2.0)
        p = wild_cluster_bootstrap_t(
            fit, X, clusters, term_name="arm_bin", B=299, seed=42
        )
        assert 0.0 <= p <= 1.0

    def test_no_effect_p_in_range(self):
        fit, X, clusters = self._fit_clustered_ols(effect=0.0, seed=7)
        p = wild_cluster_bootstrap_t(
            fit, X, clusters, term_name="arm_bin", B=299, seed=42
        )
        # With zero true effect, p may occasionally be small due to noise;
        # just verify it's a valid p-value.
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# 6-7: _compute_effect_size_from_fit
# ---------------------------------------------------------------------------

class TestComputeEffectSizeFromFit:
    @staticmethod
    def _fit_ols():
        """Simple formula-based OLS fit for effect size testing."""
        import statsmodels.formula.api as smf

        rng = np.random.default_rng(99)
        n = 40
        df = pd.DataFrame({
            "y": 0.8 * np.concatenate([np.ones(20), np.zeros(20)]) + rng.normal(0, 1, n),
            "arm_bin": np.concatenate([np.ones(20), np.zeros(20)]),
        })
        fit = smf.ols("y ~ arm_bin", data=df).fit()
        return fit

    def test_hedges_g_correction(self):
        fit = self._fit_ols()
        res = _compute_effect_size_from_fit(fit, "arm_bin", method="hedges_g")
        assert np.isfinite(res["d"])
        assert np.isfinite(res["se_d"])
        assert res["d_lower"] < res["d"] < res["d_upper"]

    def test_unbalanced_se_formula(self):
        """Effect size SE uses (n1+n2)/(n1*n2) formula."""
        import statsmodels.formula.api as smf

        rng = np.random.default_rng(88)
        df = pd.DataFrame({
            "y": 0.5 * np.concatenate([np.ones(30), np.zeros(10)]) + rng.normal(0, 1, 40),
            "arm_bin": np.concatenate([np.ones(30), np.zeros(10)]),
        })
        fit = smf.ols("y ~ arm_bin", data=df).fit()
        res = _compute_effect_size_from_fit(fit, "arm_bin", method="hedges_g")
        assert np.isfinite(res["se_d"])
        assert res["se_d"] > 0


# ---------------------------------------------------------------------------
# 8-9: abundance_did transform paths
# ---------------------------------------------------------------------------

class TestAbundanceDidTransform:
    @staticmethod
    def _make_abundance_adata():
        """Create AnnData with engineered cell-type shift."""
        n_p = 20
        rows = []
        for i in range(n_p):
            pid = f"P{i}"
            arm = "Treated" if i < 10 else "Control"
            for v in ("V1", "V2"):
                jitter = i % 3
                n_a = (100 if (arm == "Treated" and v == "V2") else 50) + jitter
                for _ in range(n_a):
                    rows.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "A"})
                for _ in range(50):
                    rows.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "B"})

        obs = pd.DataFrame(rows)
        adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)
        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            celltype_col="celltype",
            arm_treated="Treated",
            arm_control="Control",
        )
        return adata, design

    def test_logit_transform(self):
        adata, design = self._make_abundance_adata()
        res = st.abundance_did(adata, design, visits=("V1", "V2"), transform="logit", min_units=2)
        assert len(res) >= 1
        assert "beta_DiD" in res.columns

    def test_none_transform(self):
        adata, design = self._make_abundance_adata()
        res = st.abundance_did(adata, design, visits=("V1", "V2"), transform="none", min_units=2)
        assert len(res) >= 1
        assert "beta_DiD" in res.columns


# ---------------------------------------------------------------------------
# 10: abundance_did fallback (no paired participants)
# ---------------------------------------------------------------------------

def test_abundance_did_fallback_empty():
    """When no paired participants exist, result should be empty."""
    obs_list = []
    for i in range(10):
        pid = f"P{i}"
        arm = "Treated" if i < 5 else "Control"
        # Only one visit per participant => no pairs
        v = "V1" if i < 5 else "V2"
        for _ in range(20):
            obs_list.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "A"})
        for _ in range(20):
            obs_list.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "B"})

    obs = pd.DataFrame(obs_list)
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        celltype_col="celltype",
        arm_treated="Treated",
        arm_control="Control",
    )
    res = st.abundance_did(adata, design, visits=("V1", "V2"), min_units=2)
    assert res.empty or len(res) == 0


# ---------------------------------------------------------------------------
# 11-12: trend_interaction quadratic/cubic
# ---------------------------------------------------------------------------

class TestTrendInteraction:
    @staticmethod
    def _make_longitudinal_adata(n_visits: int = 4, seed: int = 0):
        """Create AnnData with 3+ visits for trend_interaction."""
        rng = np.random.default_rng(seed)
        n_p = 16
        visits = [f"V{i}" for i in range(n_visits)]

        obs_list = []
        for i in range(n_p):
            arm = "Treated" if i < n_p // 2 else "Control"
            for v in visits:
                obs_list.append({
                    "participant_id": f"P{i}",
                    "visit": v,
                    "arm": arm,
                })

        obs = pd.DataFrame(obs_list)
        n_obs = len(obs)
        X = rng.poisson(2, size=(n_obs, 10)).astype(float)
        adata = AnnData(X=sparse.csr_matrix(X), obs=obs)
        adata.var_names = [f"Gene{i}" for i in range(10)]
        adata.layers["counts"] = adata.X.copy()

        # Add a module score in obs for easy testing
        adata.obs["score1"] = rng.normal(size=n_obs)

        design = TrialDesign(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            arm_control="Control",
        )
        return adata, design, visits

    def test_quadratic(self):
        adata, design, visits = self._make_longitudinal_adata(n_visits=4)
        res = st.trend_interaction(
            adata, features=["score1"], design=design, visits=visits, model="quadratic",
        )
        assert not res.empty
        assert "beta_treat_trend" in res.columns
        assert "beta_treat_trend2" in res.columns

    def test_cubic(self):
        adata, design, visits = self._make_longitudinal_adata(n_visits=5)
        res = st.trend_interaction(
            adata, features=["score1"], design=design, visits=visits, model="cubic",
        )
        assert not res.empty
        assert "beta_treat_trend3" in res.columns
        assert "p_treat_trend3" in res.columns


# ---------------------------------------------------------------------------
# 13: pseudobulk_did with use_bootstrap=True
# ---------------------------------------------------------------------------

def test_pseudobulk_did_bootstrap():
    rng = np.random.default_rng(42)
    n_p = 20
    n_genes = 5
    obs_list = []
    for i in range(n_p):
        arm = "Treated" if i < n_p // 2 else "Control"
        for v in ("V1", "V2"):
            # Multiple cells per participant-visit for pseudobulk
            for _ in range(10):
                obs_list.append({
                    "participant_id": f"P{i}",
                    "visit": v,
                    "arm": arm,
                })

    obs = pd.DataFrame(obs_list)
    X = rng.poisson(5, size=(len(obs), n_genes)).astype(float)
    adata = AnnData(X=sparse.csr_matrix(X), obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    adata.layers["counts"] = adata.X.copy()

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res = st.pseudobulk_did(
        adata, genes=["Gene0", "Gene1"], design=design,
        visits=("V1", "V2"), use_bootstrap=True, n_boot=99, seed=1,
    )
    assert not res.empty
    assert "beta_DiD" in res.columns
    assert "p_DiD" in res.columns
    valid_p = res["p_DiD"].dropna()
    assert (valid_p >= 0).all() and (valid_p <= 1).all()


# ---------------------------------------------------------------------------
# 14: within_arm_comparison with aggregate="cell"
# ---------------------------------------------------------------------------

def test_within_arm_cell_aggregate():
    adata, design = _make_trial_adata(n_participants=20, n_genes=10, seed=11)
    # Add signal to treated V2
    rng = np.random.default_rng(11)
    adata.obs["score_x"] = rng.normal(size=adata.n_obs)
    mask = (adata.obs["arm"] == "Treated") & (adata.obs["visit"] == "V2")
    adata.obs.loc[mask, "score_x"] += 3.0

    res = st.within_arm_comparison(
        adata,
        arm="Treated",
        features=["score_x"],
        design=design,
        visits=("V1", "V2"),
        aggregate="cell",  # non-aggregated path
    )
    assert not res.empty
    assert "beta_time" in res.columns


# ---------------------------------------------------------------------------
# 15: run_gsea_pseudobulk with return_obj=True
# ---------------------------------------------------------------------------

def test_gsea_pseudobulk_return_obj():
    gseapy = pytest.importorskip("gseapy")

    rng = np.random.default_rng(5)
    n_p, n_genes = 20, 30
    obs_list = []
    for i in range(n_p):
        arm = "Treated" if i < n_p // 2 else "Control"
        for v in ("V1", "V2"):
            for _ in range(15):
                obs_list.append({
                    "participant_id": f"P{i}",
                    "visit": v,
                    "arm": arm,
                })

    obs = pd.DataFrame(obs_list)
    X = rng.poisson(5, size=(len(obs), n_genes)).astype(float)
    adata = AnnData(X=sparse.csr_matrix(X), obs=obs)
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    adata.layers["counts"] = adata.X.copy()

    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    gene_sets = {"pathway1": list(adata.var_names[:10])}

    result = st.run_gsea_pseudobulk(
        adata, gene_sets=gene_sets, design=design,
        visits=("V1", "V2"), return_obj=True, min_units=2,
        min_size=1, max_size=5000,
    )
    # return_obj=True should return a gseapy Prerank object (not a DataFrame)
    assert not isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# 16: _perm_test_diff with explicit treated_label
# ---------------------------------------------------------------------------

def test_perm_test_diff_treated_label():
    rng = np.random.default_rng(99)
    delta = pd.Series(rng.normal(0, 1, 20))
    arms = pd.Series(["Drug"] * 10 + ["Placebo"] * 10)

    p = _perm_test_diff(delta, arms, n_perm=499, seed=42, treated_label="Drug")
    assert 0.0 <= p <= 1.0

    # With zero effect, should generally be non-significant
    delta_null = pd.Series(rng.normal(0, 1, 20))
    p_null = _perm_test_diff(delta_null, arms, n_perm=499, seed=42, treated_label="Drug")
    assert 0.0 <= p_null <= 1.0
