import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

pymc = pytest.importorskip("pymc")


def test_did_table_bayes_basic():
    obs = []
    for pid in ["P1", "P2", "P3", "P4"]:
        arm = "Treated" if pid in ["P1", "P2"] else "Control"
        for visit in ["V1", "V2"]:
            obs.append({"participant_id": pid, "visit": visit, "arm": arm})
    obs = pd.DataFrame(obs)
    X = np.random.RandomState(0).normal(size=(len(obs), 1))
    adata = AnnData(X=X, obs=obs, var=pd.DataFrame(index=["G1"]))

    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    res = st.did_table_bayes(
        adata,
        features=["G1"],
        design=design,
        visits=("V1", "V2"),
        draws=200,
        tune=200,
        seed=0,
    )

    assert res.shape[0] == 1
    assert {"beta_DiD", "ci_low", "ci_high", "p_bayes"}.issubset(res.columns)
