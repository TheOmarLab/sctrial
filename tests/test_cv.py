import numpy as np

import sctrial as st


def test_loo_cv(sample_adata, trial_design):
    res = st.loo_cv_did(
        sample_adata,
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
        aggregate="participant_visit",
        standardize=True,
    )
    assert not res.empty
    # Check estimates are finite
    valid = res["beta_DiD"].dropna()
    assert len(valid) > 0
    assert all(np.isfinite(valid))

    summary = st.cv_summary(res)
    assert "n_loo" in summary.columns
    # CV mean should be close to full estimate
    assert np.isfinite(summary.iloc[0]["mean_loo"])
    assert np.isfinite(summary.iloc[0]["std_loo"])


def test_kfold_cv(sample_adata, trial_design):
    res = st.kfold_cv_did(
        sample_adata,
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
        k=3,
        seed=1,
    )
    assert not res.empty
    # Check that estimates are finite and reasonable
    row = res.iloc[0]
    assert np.isfinite(row["beta_full"])
    if np.isfinite(row["beta_cv_mean"]):
        assert np.isfinite(row["beta_cv_sd"])
        # Sign consistency should be between 0 and 1
        if np.isfinite(row["sign_consistency"]):
            assert 0 <= row["sign_consistency"] <= 1
    assert row["n_cv_samples"] > 0


def test_influence_diagnostics(sample_adata, trial_design):
    loo = st.loo_cv_did(
        sample_adata,
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
        aggregate="participant_visit",
        standardize=True,
    )
    infl = st.influence_diagnostics(loo)
    assert "influence" in infl.columns
    assert "is_influential" in infl.columns
    # Influence scores should be non-negative
    valid_infl = infl["influence"].dropna()
    assert all(valid_infl >= 0)
