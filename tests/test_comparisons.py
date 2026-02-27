import numpy as np

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

    # --- SE / CI columns ---
    assert "se_time" in res.columns
    assert "ci_lo_time" in res.columns
    assert "ci_hi_time" in res.columns

    se = res.iloc[0]["se_time"]
    assert np.isfinite(se) and se > 0, "SE must be positive and finite"

    lo, hi = res.iloc[0]["ci_lo_time"], res.iloc[0]["ci_hi_time"]
    beta = res.iloc[0]["beta_time"]
    assert lo < beta < hi, "CI must bracket the point estimate"
    assert lo > 0, "CI lower bound should be positive for a strong signal"


def test_within_arm_bootstrap(sample_adata, trial_design):
    """Bootstrap within-arm comparison produces expected columns and valid CIs."""
    treated_v2_mask = (sample_adata.obs["arm"] == "Treated") & (sample_adata.obs["visit"] == "V2")
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
        use_bootstrap=True,
        n_boot=199,
        seed=42,
    )

    assert len(res) == 1
    row = res.iloc[0]

    # Bootstrap columns must exist
    assert "p_time_boot" in res.columns
    assert "se_time_boot" in res.columns
    assert "ci_lo_boot" in res.columns
    assert "ci_hi_boot" in res.columns

    # Primary p-value should be the bootstrap p-value
    assert row["p_time"] == row["p_time_boot"]

    # Bootstrap SE must be finite and positive
    assert np.isfinite(row["se_time_boot"]) and row["se_time_boot"] > 0

    # Bootstrap CI must bracket the point estimate
    beta = row["beta_time"]
    assert row["ci_lo_boot"] < beta < row["ci_hi_boot"]

    # Strong signal: CI lower bound should be positive
    assert row["ci_lo_boot"] > 0

    # FDR should be based on bootstrap p-values
    assert "FDR_time" in res.columns
    assert np.isfinite(row["FDR_time"])


def test_within_arm_bootstrap_schema_consistency(sample_adata, trial_design):
    """Bootstrap columns follow the same naming pattern as did_table bootstrap."""
    from scipy import sparse
    X = sample_adata.X.toarray()
    mask = (sample_adata.obs["arm"] == "Treated") & (sample_adata.obs["visit"] == "V2")
    X[mask, 0] += 10.0
    sample_adata.X = sparse.csr_matrix(X)

    # Run with and without bootstrap
    res_plain = st.within_arm_comparison(
        sample_adata, arm="Treated", features=["G0", "G1"],
        design=trial_design, visits=("V1", "V2"),
    )
    res_boot = st.within_arm_comparison(
        sample_adata, arm="Treated", features=["G0", "G1"],
        design=trial_design, visits=("V1", "V2"),
        use_bootstrap=True, n_boot=99, seed=0,
    )

    # Plain result must NOT have bootstrap columns
    assert "p_time_boot" not in res_plain.columns
    assert "se_time_boot" not in res_plain.columns
    assert "ci_lo_boot" not in res_plain.columns

    # Bootstrap result must have them
    for col in ["p_time_boot", "se_time_boot", "ci_lo_boot", "ci_hi_boot"]:
        assert col in res_boot.columns, f"Missing bootstrap column: {col}"

    # Both must share the same base columns
    base_cols = ["feature", "beta_time", "se_time", "ci_lo_time", "ci_hi_time",
                 "p_time", "n_units", "FDR_time"]
    for col in base_cols:
        assert col in res_plain.columns
        assert col in res_boot.columns

    # Point estimates must match (same data, same model)
    np.testing.assert_allclose(
        res_plain["beta_time"].values, res_boot["beta_time"].values,
    )


def test_within_arm_bootstrap_nan_feature(sample_adata, trial_design):
    """Zero-variance feature returns NaN rows even with bootstrap."""
    from scipy import sparse
    X = sample_adata.X.toarray()
    # Set feature G5 to constant across all Treated cells
    treated_mask = sample_adata.obs["arm"] == "Treated"
    X[treated_mask, 5] = 1.0
    sample_adata.X = sparse.csr_matrix(X)

    res = st.within_arm_comparison(
        sample_adata, arm="Treated", features=["G5"],
        design=trial_design, visits=("V1", "V2"),
        use_bootstrap=True, n_boot=99,
    )
    assert len(res) == 1
    assert np.isnan(res.iloc[0]["beta_time"])
    assert np.isnan(res.iloc[0]["p_time"])


def test_within_arm_bootstrap_nondefault_index(sample_adata, trial_design):
    """Bootstrap works when statsmodels drops rows with NaN values."""
    from scipy import sparse

    # Add signal
    treated_v2_mask = (sample_adata.obs["arm"] == "Treated") & (
        sample_adata.obs["visit"] == "V2"
    )
    X = sample_adata.X.toarray()
    X[treated_v2_mask, 0] += 10.0

    # Inject a NaN into one cell's expression to force statsmodels row-drop.
    # This creates a mismatch between df_feat rows and fit.model.exog rows
    # that would crash if .iloc were used instead of .loc for cluster alignment.
    treated_v1_idx = np.where(
        (sample_adata.obs["arm"] == "Treated") & (sample_adata.obs["visit"] == "V1")
    )[0]
    if len(treated_v1_idx) > 0:
        X[treated_v1_idx[0], 0] = np.nan
    sample_adata.X = sparse.csr_matrix(X)

    res = st.within_arm_comparison(
        sample_adata,
        arm="Treated",
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
        use_bootstrap=True,
        n_boot=99,
        seed=42,
    )

    assert len(res) == 1
    # Should not crash and should produce finite results
    # (the NaN cell is dropped, but remaining data has strong signal)
    row = res.iloc[0]
    assert np.isfinite(row["beta_time"])
    assert "p_time_boot" in res.columns


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

    # --- SE / CI columns ---
    assert "se_arm" in res.columns
    assert "ci_lo_arm" in res.columns
    assert "ci_hi_arm" in res.columns

    se = res.iloc[0]["se_arm"]
    assert np.isfinite(se) and se > 0, "SE must be positive and finite"

    lo, hi = res.iloc[0]["ci_lo_arm"], res.iloc[0]["ci_hi_arm"]
    beta = res.iloc[0]["beta_arm"]
    assert lo < beta < hi, "CI must bracket the point estimate"
    assert lo > 0, "CI lower bound should be positive for a strong signal"


def test_between_arm_wilcoxon_has_se_ci(sample_adata, trial_design):
    """Wilcoxon method should also return SE and CI columns."""
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
        method="wilcoxon",
    )

    assert "se_arm" in res.columns
    assert "ci_lo_arm" in res.columns
    assert "ci_hi_arm" in res.columns

    se = res.iloc[0]["se_arm"]
    assert np.isfinite(se) and se > 0

    lo, hi = res.iloc[0]["ci_lo_arm"], res.iloc[0]["ci_hi_arm"]
    beta = res.iloc[0]["beta_arm"]
    assert lo < beta < hi, "CI must bracket the point estimate"
