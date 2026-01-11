import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

lifelines = pytest.importorskip("lifelines")


def test_hazard_regression_with_features_basic():
    obs = []
    for pid in range(10):
        for visit in ["V1"]:
            obs.append(
                {
                    "participant_id": f"P{pid}",
                    "visit": visit,
                    "time": 5 + pid,
                    "event": 1 if pid % 2 == 0 else 0,
                    "score": pid * 0.1,
                }
            )
    obs = pd.DataFrame(obs)
    X = np.random.RandomState(0).normal(size=(len(obs), 2))
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=["G1", "G2"]))

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="T",
        arm_control="C",
    )

    res = st.hazard_regression_with_features(
        adata,
        features=["score", "G1"],
        design=design,
        time_col="time",
        event_col="event",
        visit="V1",
    )

    assert set(res.columns) >= {"feature", "HR", "HR_low", "HR_high", "p", "n"}
    assert res.shape[0] == 2
