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


def test_event_study():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.event_study_did(ad, ["G0"], design, visits=("V1", "V2", "V3"))
    assert not res.empty


def test_polynomial_trend():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.polynomial_trend(ad, "G0", design, visits=("V1", "V2", "V3"), degree=2)
    assert "coefficients" in res


def test_parallel_trends():
    ad = _make_three_visit_adata()
    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    res = st.test_parallel_trends(ad, ["G0"], design, pre_visits=("V1", "V2", "V3"))
    assert not res.empty
