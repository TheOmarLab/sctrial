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
