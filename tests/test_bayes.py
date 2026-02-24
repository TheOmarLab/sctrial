import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

# Ensure PyTensor has a writable compiledir before importing pymc
tmp_dir = tempfile.mkdtemp(prefix="pytensor_test_")
os.environ["PYTENSOR_FLAGS"] = f"compiledir={tmp_dir},cxx="

pymc = pytest.importorskip("pymc")


def test_did_table_bayes_basic():
    # Ensure PyTensor has a writable compiledir in CI/local tests
    tmp_dir = tempfile.mkdtemp(prefix="pytensor_test_")
    os.environ["PYTENSOR_FLAGS"] = f"compiledir={tmp_dir},cxx="
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
        chains=1,
        seed=0,
    )

    assert res.shape[0] == 1
    assert {"beta_DiD", "ci_low", "ci_high", "p_bayes"}.issubset(res.columns)


def test_did_table_bayes_categorical_covariate_raises():
    """Categorical (non-numeric) covariates should raise a clear TypeError."""
    obs = []
    for pid in ["P1", "P2", "P3", "P4"]:
        arm = "Treated" if pid in ["P1", "P2"] else "Control"
        for visit in ["V1", "V2"]:
            obs.append({
                "participant_id": pid,
                "visit": visit,
                "arm": arm,
                "sex": "M" if pid in ["P1", "P3"] else "F",
            })
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

    with pytest.raises(TypeError, match="(?i)non-numeric"):
        st.did_table_bayes(
            adata,
            features=["G1"],
            design=design,
            visits=("V1", "V2"),
            covariates=["sex"],
            draws=100,
            tune=100,
            chains=1,
            seed=0,
        )
