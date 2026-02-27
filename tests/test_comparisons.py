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
