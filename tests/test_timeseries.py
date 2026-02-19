import numpy as np
import pandas as pd
from anndata import AnnData

import sctrial as st


def _make_three_visit_adata():
    visits = ["V1", "V2", "V3"]
    rows = []
    for i in range(12):
        arm = "Treated" if i < 6 else "Control"
        for v in visits:
            rows.append({
                "participant_id": f"P{i}",
                "visit": v,
                "arm": arm,
            })
    obs = pd.DataFrame(rows)
    X = np.random.RandomState(0).poisson(2.0, size=(len(obs), 5)).astype(float)
    adata = AnnData(X=X, obs=obs)
    adata.var_names = [f"G{i}" for i in range(5)]
    return adata


def test_trend_interaction():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.trend_interaction(ad, ["G0"], design, visits=("V1", "V2", "V3"))
    assert not res.empty
    assert "feature" in res.columns
    # Should have results for the feature
    assert len(res) >= 1
    # Coefficient values should be finite
    for col in ["beta_DiD", "p_DiD"]:
        if col in res.columns:
            valid = res[col].dropna()
            assert all(np.isfinite(valid))


def test_event_study():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.event_study_did(ad, ["G0"], design, visits=("V1", "V2", "V3"))
    assert not res.empty
    # Should have multiple visit-specific estimates
    assert len(res) >= 1


def test_polynomial_trend():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.polynomial_trend(ad, "G0", design, visits=("V1", "V2", "V3"), degree=2)
    assert "coefficients" in res
    # Coefficients should be finite
    for k, v in res["coefficients"].items():
        assert np.isfinite(v), f"Coefficient {k} is not finite"


def test_parallel_trends():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.test_parallel_trends(ad, ["G0"], design, pre_visits=("V1", "V2", "V3"))
    assert not res.empty
    assert "feature" in res.columns
    # p-values should be between 0 and 1
    for col in res.columns:
        if col.startswith("p_"):
            valid = res[col].dropna()
            assert all((valid >= 0) & (valid <= 1))
