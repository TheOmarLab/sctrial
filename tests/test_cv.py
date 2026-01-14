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
    summary = st.cv_summary(res)
    assert "n_loo" in summary.columns


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
