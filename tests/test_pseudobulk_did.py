import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def test_pseudobulk_did_basic():
    participants = [f"P{i}" for i in range(8)]
    arms = ["Treated"] * 4 + ["Control"] * 4
    visits = ["V1", "V2"]

    obs_rows = []
    expr = []
    for pid, arm in zip(participants, arms):
        for visit in visits:
            for _ in range(10):
                obs_rows.append({
                    "participant_id": pid,
                    "arm": arm,
                    "visit": visit,
                    "celltype": "CT1",
                })
                if arm == "Treated" and visit == "V2":
                    expr.append([20.0, 10.0])
                else:
                    expr.append([10.0, 10.0])

    obs = pd.DataFrame(obs_rows)
    X = np.asarray(expr, dtype=float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["GENE1", "GENE2"]
    adata.layers["counts"] = adata.X.copy()

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )

    res = st.pseudobulk_did(
        adata,
        genes=["GENE1"],
        design=design,
        visits=("V1", "V2"),
        celltype_col="celltype",
        min_cells_per_group=1,
        min_paired=2,
        use_bootstrap=False,
    )

    assert not res.empty
    assert res.loc[0, "beta_DiD"] > 0
    assert np.isfinite(res.loc[0, "p_DiD"])


def test_pseudobulk_expression_matches_groupby():
    obs = pd.DataFrame({
        "participant_id": ["P1", "P1", "P2", "P2"],
        "visit": ["V1", "V2", "V1", "V2"],
    })
    X = np.asarray([
        [10.0, 5.0],
        [20.0, 5.0],
        [15.0, 5.0],
        [25.0, 5.0],
    ])
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["G1", "G2"]
    adata.layers["counts"] = adata.X.copy()

    pb = st.pseudobulk_expression(
        adata,
        genes=["G1", "G2"],
        groupby=["participant_id", "visit"],
        counts_layer="counts",
        log1p=False,
        include_n_cells=True,
    )

    # Manual groupby for comparison (pure numpy)
    manual_rows = []
    for (pid, visit), idx in obs.groupby(["participant_id", "visit"], observed=True).indices.items():
        sub = X[idx]
        sums = sub.sum(axis=0)
        total = sums.sum()
        manual_rows.append([pid, visit, sums[0] / (total + 1e-12) * 1e6, sums[1] / (total + 1e-12) * 1e6])
    manual = pd.DataFrame(manual_rows, columns=["participant_id", "visit", "G1", "G2"])
    manual = manual.sort_values(["participant_id", "visit"]).reset_index(drop=True)
    pb_sorted = pb.sort_values(["participant_id", "visit"]).reset_index(drop=True)

    assert np.allclose(pb_sorted[["G1", "G2"]].values, manual[["G1", "G2"]].values, atol=1e-6)


def _make_did_adata(n_per_arm=4, n_cells=5, seed=42):
    """Helper: create a small AnnData for DiD tests."""
    rng = np.random.default_rng(seed)
    participants = [f"P{i}" for i in range(n_per_arm * 2)]
    arms = ["Treated"] * n_per_arm + ["Control"] * n_per_arm

    obs_rows, expr = [], []
    for pid, arm in zip(participants, arms):
        for visit in ["Pre", "Post"]:
            for _ in range(n_cells):
                obs_rows.append({
                    "participant_id": pid, "arm": arm, "visit": visit,
                })
                base = 15.0 if (arm == "Treated" and visit == "Post") else 10.0
                expr.append([base + rng.normal(0, 1)])

    obs = pd.DataFrame(obs_rows)
    X = np.asarray(expr, dtype=float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = ["GENE1"]
    return adata


def test_did_table_surfaces_cov_type_used():
    """did_table output must include ``cov_type_used`` for transparency."""
    adata = _make_did_adata(n_per_arm=4, seed=42)
    design = st.TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col="arm",
        arm_treated="Treated", arm_control="Control",
    )

    import warnings as _w
    with _w.catch_warnings(record=True):
        _w.simplefilter("always")
        res = st.did_table(
            adata, features=["GENE1"], design=design, visits=("Pre", "Post"),
            aggregate="participant_visit", use_bootstrap=False,
        )

    assert not res.empty
    assert "cov_type_used" in res.columns, "cov_type_used missing from output"
    assert np.isfinite(res.loc[0, "se_DiD"])
    assert np.isfinite(res.loc[0, "p_DiD"])
    # With enough participants cluster-robust should succeed
    assert res.loc[0, "cov_type_used"] == "cluster"


def test_did_table_nonrobust_fallback_on_nan_se():
    """Regression test: when cluster-robust SE is NaN, did_table falls back to
    nonrobust SE and emits a warning.

    We monkeypatch statsmodels.formula.api.ols inside the did module so that
    the first cluster-robust fit returns NaN SE, then verify the fallback
    produces finite outputs with cov_type_used='nonrobust'.
    """
    import warnings as _w
    from unittest.mock import patch

    import sctrial.stats.did as did_module

    adata = _make_did_adata(n_per_arm=4, seed=42)
    design = st.TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col="arm",
        arm_treated="Treated", arm_control="Control",
    )

    class _NanSEResult:
        """Wrapper that makes the cluster-robust fit return NaN SE."""
        def __init__(self, real_fit):
            self.params = real_fit.params
            self.pvalues = real_fit.pvalues
            self.scale = real_fit.scale
            self.model = real_fit.model
            self.bse = real_fit.bse.copy()
            for k in self.bse.index:
                self.bse[k] = np.nan

    class _FakeModel:
        def __init__(self, real_model):
            self._real = real_model
        def fit(self, **kwargs):
            real_fit = self._real.fit(**kwargs)
            if kwargs.get("cov_type") == "cluster":
                return _NanSEResult(real_fit)
            return real_fit

    _orig_ols = did_module.smf.ols
    _orig_wls = did_module.smf.wls

    def _patched_ols(formula, data, **kw):
        return _FakeModel(_orig_ols(formula, data, **kw))

    def _patched_wls(formula, data, **kw):
        return _FakeModel(_orig_wls(formula, data, **kw))

    # Patch both ols and wls (aggregation adds n_cells → WLS path)
    with patch.object(did_module.smf, "ols", new=_patched_ols), \
         patch.object(did_module.smf, "wls", new=_patched_wls):
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            res = st.did_table(
                adata, features=["GENE1"], design=design,
                visits=("Pre", "Post"),
                aggregate="participant_visit", use_bootstrap=False,
            )

    # Core: finite outputs after fallback
    assert not res.empty
    assert np.isfinite(res.loc[0, "se_DiD"]), f"se_DiD not finite: {res.loc[0, 'se_DiD']}"
    assert np.isfinite(res.loc[0, "p_DiD"]), f"p_DiD not finite: {res.loc[0, 'p_DiD']}"
    assert res.loc[0, "cov_type_used"] == "nonrobust"

    # Warning emitted
    fallback_warns = [w for w in caught if "degenerate" in str(w.message)]
    assert len(fallback_warns) > 0, "No fallback warning emitted"


def test_did_table_bootstrap_with_nonrobust_fallback():
    """When cluster-robust falls back, bootstrap should also use nonrobust."""
    adata = _make_did_adata(n_per_arm=4, seed=99)
    design = st.TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col="arm",
        arm_treated="Treated", arm_control="Control",
    )

    import warnings as _w
    with _w.catch_warnings(record=True):
        _w.simplefilter("always")
        res = st.did_table(
            adata, features=["GENE1"], design=design, visits=("Pre", "Post"),
            aggregate="participant_visit", use_bootstrap=True, n_boot=99, seed=42,
        )

    assert not res.empty
    assert np.isfinite(res.loc[0, "p_DiD"]), "Bootstrap p_DiD is not finite"
    assert np.isfinite(res.loc[0, "se_DiD"]), "se_DiD is not finite"
    assert "cov_type_used" in res.columns


def test_did_table_bootstrap_returns_ci_columns():
    """Bootstrap mode should populate se_DiD_boot, ci_lo_boot, ci_hi_boot."""
    adata = _make_did_adata(n_per_arm=5, seed=42)
    design = st.TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col="arm",
        arm_treated="Treated", arm_control="Control",
    )

    import warnings as _w
    with _w.catch_warnings(record=True):
        _w.simplefilter("always")
        res = st.did_table(
            adata, features=["GENE1"], design=design, visits=("Pre", "Post"),
            aggregate="participant_visit", use_bootstrap=True, n_boot=199, seed=42,
        )

    assert not res.empty
    # New bootstrap CI columns must be present
    for col in ["se_DiD_boot", "ci_lo_boot", "ci_hi_boot", "p_DiD_boot"]:
        assert col in res.columns, f"{col} missing from bootstrap results"
        assert np.isfinite(res.loc[0, col]), f"{col} is not finite"

    # CI should bracket the point estimate for a clear signal
    beta = res.loc[0, "beta_DiD"]
    assert res.loc[0, "ci_lo_boot"] < res.loc[0, "ci_hi_boot"]
    # With 5 per arm and effect=5, CI should contain the estimate
    assert res.loc[0, "ci_lo_boot"] <= beta <= res.loc[0, "ci_hi_boot"]
    # Bootstrap SE should be positive
    assert res.loc[0, "se_DiD_boot"] > 0


def test_did_table_no_bootstrap_omits_ci_columns():
    """Without bootstrap, CI columns should not be present."""
    adata = _make_did_adata(n_per_arm=5, seed=42)
    design = st.TrialDesign(
        participant_col="participant_id", visit_col="visit", arm_col="arm",
        arm_treated="Treated", arm_control="Control",
    )

    import warnings as _w
    with _w.catch_warnings(record=True):
        _w.simplefilter("always")
        res = st.did_table(
            adata, features=["GENE1"], design=design, visits=("Pre", "Post"),
            aggregate="participant_visit", use_bootstrap=False,
        )

    assert not res.empty
    # Without bootstrap, these columns should not exist
    for col in ["se_DiD_boot", "ci_lo_boot", "ci_hi_boot", "p_DiD_boot"]:
        assert col not in res.columns, f"{col} should not be present without bootstrap"
