import pytest
import sctrial as st

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plotting_trial_interaction(sample_adata, trial_design):
    # Prepare data
    adata = st.add_log1p_cpm_layer(sample_adata, out_layer="log1p_cpm")
    
    # Plot gene
    ax = st.plot_trial_interaction(adata, feature="G0", design=trial_design, visits=("V1", "V2"), layer="log1p_cpm")
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Plot module score
    gene_sets = {"SET": ["G0", "G1", "G2"]}
    adata = st.score_gene_sets(adata, gene_sets, layer="log1p_cpm", method="mean", prefix="ms_")
    ax = st.plot_trial_interaction(adata, feature="ms_SET", design=trial_design, visits=("V1", "V2"))
    assert isinstance(ax, plt.Axes)
    plt.close()

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plotting_abundance_interaction(sample_adata, trial_design):
    ax = st.plot_abundance_interaction(sample_adata, celltype="TypeA", design=trial_design, visits=("V1", "V2"))
    assert isinstance(ax, plt.Axes)
    plt.close()

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_volcano_helpers():
    import pandas as pd
    df = pd.DataFrame({
        "beta_DiD": [1.0, -1.0, 0.0],
        "p_DiD": [0.01, 0.05, 1.0]
    })
    
    vf = st.plotting.did_volcano_frame(df)
    assert "neglog10p" in vf.columns
    assert vf["neglog10p"].iloc[0] == pytest.approx(2.0)
    
    slp = st.plotting.signed_logp(df)
    assert slp.iloc[0] > 0
    assert slp.iloc[1] < 0
