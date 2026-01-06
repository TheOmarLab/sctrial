import sctrial as st


def test_within_arm_comparison(sample_adata, trial_design):
    # Add some signal for V2 in Treated group
    treated_v2_mask = (sample_adata.obs["arm"] == "Treated") & (sample_adata.obs["visit"] == "V2")
    # Convert to dense for easy modification or handle sparse correctly
    X = sample_adata.X.toarray()
    X[treated_v2_mask, 0] += 10.0
    from scipy import sparse
    sample_adata.X = sparse.csr_matrix(X)

    res = st.within_arm_comparison(
        sample_adata,
        arm="Treated",
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
    )

    assert len(res) == 1
    assert res.iloc[0]["feature"] == "G0"
    assert res.iloc[0]["p_time"] < 0.05
    assert res.iloc[0]["beta_time"] > 0

def test_between_arm_comparison(sample_adata, trial_design):
    # Add some signal for Treated group in V2
    v2_mask = (sample_adata.obs["visit"] == "V2")
    treated_mask = (sample_adata.obs["arm"] == "Treated")

    X = sample_adata.X.toarray()
    X[v2_mask & treated_mask, 0] += 10.0
    from scipy import sparse
    sample_adata.X = sparse.csr_matrix(X)

    res = st.between_arm_comparison(
        sample_adata,
        visit="V2",
        features=["G0"],
        design=trial_design,
    )

    assert len(res) == 1
    assert res.iloc[0]["feature"] == "G0"
    assert res.iloc[0]["p_arm"] < 0.05
    assert res.iloc[0]["beta_arm"] > 0
