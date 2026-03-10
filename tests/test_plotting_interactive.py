import pandas as pd
import pytest

import sctrial as st

plotly = pytest.importorskip("plotly")


def test_plot_did_forest_interactive():
    df = pd.DataFrame(
        {"feature": ["A", "B"], "beta_DiD": [0.2, -0.1], "se_DiD": [0.1, 0.2], "p_DiD": [0.05, 0.2]}
    )
    fig = st.plot_did_forest_interactive(df)
    assert fig is not None
    assert len(fig.data) > 0, "Figure should have at least one trace"


def test_plot_did_volcano_interactive():
    df = pd.DataFrame({"beta_DiD": [0.2, -0.1], "p_DiD": [0.05, 0.2]})
    fig = st.plot_did_volcano_interactive(df)
    assert fig is not None
    assert len(fig.data) > 0, "Figure should have at least one trace"


def test_plot_did_forest_interactive_empty():
    """Empty DataFrame should raise ValueError."""
    df = pd.DataFrame(columns=["feature", "beta_DiD", "se_DiD", "p_DiD"])
    with pytest.raises(ValueError):
        st.plot_did_forest_interactive(df)


def test_plot_did_forest_interactive_celltype_fallback():
    """Interactive forest should accept 'celltype' column like static forest."""
    df = pd.DataFrame(
        {
            "celltype": ["TypeA", "TypeB"],
            "beta_DiD": [0.2, -0.1],
            "se_DiD": [0.1, 0.2],
            "p_DiD": [0.05, 0.2],
        }
    )
    fig = st.plot_did_forest_interactive(df)
    assert fig is not None
    assert len(fig.data) > 0
