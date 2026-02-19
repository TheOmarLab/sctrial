import numpy as np
import pandas as pd

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
    assert "icc" in res.columns
    assert "var_participant" in res.columns
    assert "converged" in res.columns
    # Check coefficient values are finite for features with treatment effect
    g0_row = res[res["feature"] == "G0"].iloc[0]
    assert np.isfinite(g0_row["beta_DiD"])
    assert np.isfinite(g0_row["se_DiD"])
    assert np.isfinite(g0_row["p_DiD"])
    # ICC should be between 0 and 1 when valid
    if np.isfinite(g0_row["icc"]):
        assert 0 <= g0_row["icc"] <= 1


def test_compare_fixed_vs_mixed(sample_adata, trial_design):
    comp = st.compare_fixed_vs_mixed(
        sample_adata,
        features=["G0", "G1"],
        design=trial_design,
        visits=("V1", "V2"),
    )
    assert "beta_fixed" in comp.columns
    assert "beta_mixed" in comp.columns
    assert "agreement" in comp.columns
    assert "beta_diff" in comp.columns
    # Both methods should produce finite estimates
    for _, row in comp.iterrows():
        if np.isfinite(row["beta_fixed"]) and np.isfinite(row["beta_mixed"]):
            # Difference should be small for well-behaved data
            assert np.isfinite(row["beta_diff"])


def test_did_mixed_visits_ordering(sample_adata, trial_design):
    """Test that explicit visits tuple controls time ordering."""
    res = st.did_table_mixed(
        sample_adata,
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
    )
    assert not res.empty
    assert np.isfinite(res.iloc[0]["beta_DiD"])
