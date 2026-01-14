import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st


def test_validate_adata_basic(sample_adata, trial_design):
    issues = st.validate_adata(sample_adata, trial_design, strict=False)
    # Should at least contain counts layer warning or be empty depending on fixture
    assert isinstance(issues, list)


def test_validate_adata_strict_missing_col():
    obs = pd.DataFrame({"participant_id": ["P1"], "visit": ["V1"]})
    adata = AnnData(X=np.zeros((1, 1)), obs=obs)
    design = st.TrialDesign(arm_col="arm")
    with pytest.raises(KeyError):
        st.validate_adata(adata, design, strict=True)


def test_validate_features_missing(sample_adata):
    with pytest.raises(KeyError):
        st.validate_features(sample_adata, ["G0", "NOPE"], allow_missing=False)


def test_validate_features_allow_missing(sample_adata):
    valid, missing = st.validate_features(sample_adata, ["G0", "NOPE"], allow_missing=True)
    assert "G0" in valid
    assert "NOPE" in missing


def test_diagnose_trial_data(sample_adata, trial_design):
    report = st.diagnose_trial_data(sample_adata, trial_design, verbose=False)
    assert "n_cells" in report
    assert "n_participants" in report
    assert "warnings" in report


def test_check_covariate_balance_numeric_and_categorical():
    obs = []
    for pid in range(10):
        arm = "Treated" if pid < 5 else "Control"
        age = 30 + pid
        sex = "F" if pid % 2 == 0 else "M"
        for visit in ["V1", "V2"]:
            obs.append(
                {
                    "participant_id": f"P{pid}",
                    "visit": visit,
                    "arm": arm,
                    "age": age,
                    "sex": sex,
                }
            )
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=pd.DataFrame(obs))
    design = st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        baseline_visit="V1",
    )
    res = st.check_covariate_balance(
        adata,
        design,
        covariates=["age", "sex"],
        visit="V1",
    )
    assert not res.empty
    assert {"covariate", "smd", "mean_treated", "mean_control"}.issubset(res.columns)
