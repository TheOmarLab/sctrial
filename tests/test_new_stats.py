import numpy as np
import pandas as pd
import pytest

import sctrial as st

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


def test_bootstrap_did(sample_adata, trial_design):
    # Run DiD with bootstrap - use very small n_boot for speed
    res = st.did_table(
        sample_adata,
        features=["G0"],
        design=trial_design,
        visits=("V1", "V2"),
        use_bootstrap=True,
        n_boot=10,
        seed=42
    )
    assert "p_DiD" in res.columns
    assert res["p_DiD"].notna().all()

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_forest_plot():
    df = pd.DataFrame({
        "feature": ["G1", "G2", "G3"],
        "beta_DiD": [0.5, -0.2, 0.8],
        "se_DiD": [0.1, 0.1, 0.2],
        "p_DiD": [0.01, 0.1, 0.001]
    })
    ax = st.plot_did_forest(df)
    assert isinstance(ax, plt.Axes)
    plt.close()

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_within_arm_plot(sample_adata, trial_design):
    ax = st.plot_within_arm_comparison(
        sample_adata,
        arm="Treated",
        feature="G0",
        design=trial_design,
        visits=("V1", "V2"),
        plot_type="paired"
    )
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_abundance_did_bootstrap(sample_adata, trial_design):
    # Create a dummy dataset with enough cells for abundance
    rows = []
    for i in range(10):
        pid = f"PX{i}"
        arm = "Treated" if i < 5 else "Control"
        for v in ["V1", "V2"]:
            n_a = 15 if (arm == "Treated" and v == "V2") else 10
            for _ in range(n_a):
                rows.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "A"})
            for _ in range(10):
                rows.append({"participant_id": pid, "visit": v, "arm": arm, "celltype": "B"})
    obs = pd.DataFrame(rows)
    from anndata import AnnData
    adata = AnnData(X=np.zeros((len(obs), 1)), obs=obs)

    res = st.abundance_did(
        adata,
        design=trial_design,
        visits=("V1", "V2"),
        use_bootstrap=True,
        n_boot=10,
        min_units=2
    )
    assert "p_DiD" in res.columns
    assert len(res) > 0

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_trial_umap(sample_adata, trial_design):
    # Mock X_umap
    sample_adata.obsm["X_umap"] = np.random.randn(sample_adata.n_obs, 2)
    fig = st.plot_trial_umap(sample_adata, feature="G0", design=trial_design, visits=("V1", "V2"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
