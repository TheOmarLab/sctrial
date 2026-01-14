import numpy as np
import pandas as pd
import pytest

import sctrial as st

try:
    import matplotlib.pyplot as plt  # noqa: F401
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def test_resolve_feature(sample_adata):
    # exact
    assert st.resolve_feature(sample_adata, "participant_id") == "participant_id"
    # case-insensitive
    assert st.resolve_feature(sample_adata, "PARTICIPANT_ID") == "participant_id"
    # var
    assert st.resolve_feature(sample_adata, "g0") == "G0"

    with pytest.raises(KeyError):
        st.resolve_feature(sample_adata, "nonexistent")

def test_between_arm_comparison_wilcoxon(sample_adata, trial_design):
    res = st.between_arm_comparison(
        sample_adata,
        visit="V1",
        features=["G0", "G1"],
        design=trial_design,
        method="wilcoxon"
    )
    assert "p_arm" in res.columns
    assert res.shape[0] == 2

def test_plot_trial_umap_panel(sample_adata, trial_design):
    import matplotlib.pyplot as plt
    # mock umap
    sample_adata.obsm["X_umap"] = np.random.rand(sample_adata.n_obs, 2)
    fig = st.plot_trial_umap_panel(sample_adata, "G0", trial_design, visits=("V1", "V2"))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_plot_gsea_heatmap():
    import matplotlib.pyplot as plt
    df = pd.DataFrame({
        "Term": ["Path1", "Path2", "Path1", "Path2"],
        "pool": ["CT1", "CT1", "CT2", "CT2"],
        "NES": [1.5, -1.2, 2.0, -0.5],
        "FDR q-val": [0.01, 0.02, 0.05, 0.1],
        "collection": ["H", "H", "H", "H"]
    })
    ax = st.plot_gsea_heatmap(df, collection="H")
    assert isinstance(ax, plt.Axes)
    plt.close(ax.get_figure())
