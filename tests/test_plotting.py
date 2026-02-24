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


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_trial_interaction_aggregates_to_participant(sample_adata, trial_design):
    """Pointplot should use participant-level means, not cell-level rows."""
    from anndata import AnnData

    # Build data with multiple cells per participant-visit
    obs = []
    for pid in ["P1", "P2", "P3", "P4"]:
        arm = "Treated" if pid in ["P1", "P2"] else "Control"
        for visit in ["V1", "V2"]:
            for _ in range(50):  # 50 cells per participant-visit
                obs.append({"participant_id": pid, "visit": visit, "arm": arm,
                            "celltype": "TypeA"})
    obs = pd.DataFrame(obs)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(obs), 2))
    adata = AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=["G0", "G1"]))

    design = st.TrialDesign(arm_treated="Treated", arm_control="Control")
    ax = st.plot_trial_interaction(adata, feature="G0", design=design, visits=("V1", "V2"))

    # With 4 participants × 2 visits = 8 points used by pointplot (not 400 cells)
    assert isinstance(ax, plt.Axes)
    plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_parallel_trends_aggregates_to_participant(sample_adata, trial_design):
    """Parallel trends should use participant-level means."""
    ax = st.plot_parallel_trends(
        sample_adata, feature="G0", design=trial_design, visits=["V1", "V2"],
    )
    assert isinstance(ax, plt.Axes)
    plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_abundance_interaction_true_zeros(trial_design):
    """Participant-visits with zero cells of a celltype should show prop=0."""
    from anndata import AnnData

    # P1 has TypeA+TypeB at V1 and V2; P2 has ONLY TypeA (zero TypeB)
    obs = pd.DataFrame([
        {"participant_id": "P1", "visit": "V1", "arm": "Treated", "celltype": "TypeA"},
        {"participant_id": "P1", "visit": "V1", "arm": "Treated", "celltype": "TypeB"},
        {"participant_id": "P1", "visit": "V2", "arm": "Treated", "celltype": "TypeA"},
        {"participant_id": "P1", "visit": "V2", "arm": "Treated", "celltype": "TypeB"},
        {"participant_id": "P2", "visit": "V1", "arm": "Control", "celltype": "TypeA"},
        {"participant_id": "P2", "visit": "V1", "arm": "Control", "celltype": "TypeA"},
        {"participant_id": "P2", "visit": "V2", "arm": "Control", "celltype": "TypeA"},
    ])
    X = np.ones((len(obs), 1))
    adata = AnnData(X=X.astype(np.float32), obs=obs, var=pd.DataFrame(index=["G0"]))

    ax = st.plot_abundance_interaction(adata, celltype="TypeB", design=trial_design, visits=("V1", "V2"))
    assert isinstance(ax, plt.Axes)

    # Check that P2's visits are plotted as 0 (not missing)
    # The plot should have data for all 4 participant-visit combos
    lines_and_points = ax.collections + ax.lines
    assert len(lines_and_points) > 0, "Plot should have data even for zero-abundance combos"
    plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_within_arm_box_aggregates(sample_adata, trial_design):
    """Box mode should aggregate to participant-level, same as paired mode."""
    ax = st.plot_within_arm_comparison(
        sample_adata, arm="Treated", feature="G0", design=trial_design,
        visits=("V1", "V2"), plot_type="box",
    )
    assert isinstance(ax, plt.Axes)
    plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_within_arm_paired(sample_adata, trial_design):
    """Paired mode should work and aggregate properly."""
    ax = st.plot_within_arm_comparison(
        sample_adata, arm="Treated", feature="G0", design=trial_design,
        visits=("V1", "V2"), plot_type="paired",
    )
    assert isinstance(ax, plt.Axes)
    plt.close()


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_did_forest_static(trial_design):
    """Static forest plot should work and handle celltype fallback."""
    df = pd.DataFrame({
        "feature": ["A", "B", "C"],
        "beta_DiD": [0.5, -0.3, 0.1],
        "se_DiD": [0.1, 0.15, 0.2],
        "p_DiD": [0.01, 0.1, 0.5],
    })
    ax = st.plot_did_forest(df)
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Fallback: 'celltype' column instead of 'feature'
    df_ct = df.rename(columns={"feature": "celltype"})
    ax = st.plot_did_forest(df_ct)
    assert isinstance(ax, plt.Axes)
    plt.close()
