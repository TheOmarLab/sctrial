import sctrial as st


def test_did_table_mixed(sample_adata, trial_design):
    res = st.did_table_mixed(
        sample_adata,
        features=["G0", "G1"],
        design=trial_design,
        visits=("V1", "V2"),
    )
    assert not res.empty
    assert "beta_DiD" in res.columns


def test_compare_fixed_vs_mixed(sample_adata, trial_design):
    comp = st.compare_fixed_vs_mixed(
        sample_adata,
        features=["G0", "G1"],
        design=trial_design,
        visits=("V1", "V2"),
    )
    assert "beta_fixed" in comp.columns
    assert "beta_mixed" in comp.columns
