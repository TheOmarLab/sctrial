"""Regression tests for Categorical-level and permutation-estimand defects.

anndata stores string obs columns as Categoricals, and a slice keeps every level.
Each test reproduces one way that leaked into a result: phantom groups from
``groupby``/``value_counts``, and a mapped Categorical that sorts by category code
instead of by value. The last tests pin a permutation null to the same estimand as
the observed statistic.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from anndata import AnnData  # noqa: E402

import sctrial as st  # noqa: E402
from sctrial.benchmark.permutation import _permute_arms  # noqa: E402
from sctrial.stats.module_scores import module_score_did_by_pool  # noqa: E402
from sctrial.validation import diagnose_trial_data  # noqa: E402

SIG = "sig_NK Cell Activity"
COVID_DESIGN = st.TrialDesign(
    participant_col="participant_id", visit_col="dfo_bin", arm_col="severity",
    arm_treated="Severe", arm_control="Mild", celltype_col=None,
)


def _two_arm_cells(arm_col, arms, categories):
    """Three cells per participant at Pre and Post, with a Categorical arm column."""
    rng = np.random.default_rng(0)
    rows = [
        {"participant_id": pid, "visit": visit, arm_col: arm, SIG: rng.normal()}
        for pid, arm in arms.items() for visit in ("Pre", "Post") for _ in range(3)
    ]
    obs = pd.DataFrame(rows)
    obs[arm_col] = pd.Categorical(obs[arm_col], categories=categories)
    obs["participant_id"] = pd.Categorical(obs["participant_id"])
    obs["visit"] = pd.Categorical(obs["visit"], categories=["Pre", "Post"])
    obs.index = obs.index.astype(str)
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _separators(ax):
    return sorted(float(line.get_ydata()[0]) for line in ax.lines)


def test_figure2_heatmap_blocks_follow_the_rows_they_label(load_figure_module):
    """Arms of unequal size: the separator and captions sit on the real blocks."""
    f2 = load_figure_module("main/figure2_tnbc_analysis")
    arms = {"A": "Chemo", "B": "Chemo", "C": "anti-PDL1+Chemo",
            "D": "anti-PDL1+Chemo", "E": "anti-PDL1+Chemo", "F": "anti-PDL1+Chemo"}
    data = {
        "adata": _two_arm_cells("arm", arms, ["Chemo", "anti-PDL1+Chemo"]),
        "sig_cols": [SIG],
        "res_part": pd.DataFrame({"feature": [SIG], "beta_DiD": [0.1]}),
    }
    fig, ax = plt.subplots()
    f2._panel_f(ax, data)
    assert _separators(ax) == [1.5]
    captions = {t.get_text(): t.get_position()[1] for t in ax.texts}
    assert captions["Chemo"] == pytest.approx(0.5)
    assert captions["anti-PDL1+\nChemo"] == pytest.approx(3.5)
    plt.close(fig)


def test_supp_fig3_heatmap_puts_responders_first(load_figure_module):
    """Category order Non-responder < Responder must not decide the row order."""
    sf3 = load_figure_module("supp/supp_fig3_melanoma_tnbc_analysis")
    arms = {"A": "Responder", "B": "Non-responder", "C": "Non-responder",
            "D": "Non-responder", "E": "Non-responder"}
    data = {
        "adata": _two_arm_cells("response_harmonized", arms, ["Non-responder", "Responder"]),
        "sig_cols": [SIG],
        "did_res": pd.DataFrame({"feature": [SIG], "beta_DiD": [0.1]}),
    }
    fig, ax = plt.subplots()
    sf3._panel_i_heatmap(ax, data)
    assert ax.get_yticklabels()[0].get_text() == "A"
    assert _separators(ax) == [0.5]
    plt.close(fig)


def _raise():
    raise FileNotFoundError("dataset not used in this test")


def _tables_with(load_figure_module, monkeypatch, tmp_path, **loaders):
    tables = load_figure_module("supp/supp_tables")
    for name in ("get_sade_feldman", "get_tnbc_zhang", "get_stephenson",
                 "get_vaccine", "get_aml", "get_cart"):
        monkeypatch.setattr(tables, name, loaders.get(name, _raise))
    monkeypatch.setattr(tables, "SUPP_OUTPUT", tmp_path)
    return tables


def test_table8_lists_only_sampled_participant_visits(load_figure_module, monkeypatch, tmp_path):
    obs = pd.DataFrame({
        "participant_id": pd.Categorical(["P1", "P1", "P2"], categories=["P1", "P2", "P3"]),
        "visit": pd.Categorical(["Pre", "Post", "Pre"], categories=["Pre", "Post"]),
        "cell_type": ["T", "B", "T"],
    }, index=["0", "1", "2"])
    aml = AnnData(X=np.zeros((3, 1)), obs=obs)
    tables = _tables_with(load_figure_module, monkeypatch, tmp_path, get_aml=lambda: aml)
    out = tables.table8_dataset_metadata()
    assert len(out) == 3
    assert (out["n_cells"] > 0).all()
    assert out["n_cell_types"].tolist() == [1, 1, 1]


def _covid_cells():
    """Severe and Mild participants in DFO_8-14; one Mild participant only in another bin."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(10):
        pid, sev = f"C{i}", ("Severe" if i < 4 else "Mild")
        bins = ["DFO_15-21"] if i == 9 else ["DFO_8-14", "DFO_15-21"]
        for b in bins:
            for _ in range(3):
                rows.append({"participant_id": pid, "severity": sev, "dfo_bin": b,
                             SIG: rng.normal(0.8 if sev == "Severe" else 0.0)})
    obs = pd.DataFrame(rows)
    for c in ("participant_id", "severity", "dfo_bin"):
        obs[c] = pd.Categorical(obs[c])
    obs.index = obs.index.astype(str)
    X = np.zeros((len(obs), 1))
    return AnnData(X=X, obs=obs, layers={"log1p_cpm": X.copy()})


def test_between_arm_permutation_helpers_reproduce_between_arm_comparison():
    adata = _covid_cells()
    df = st.get_between_arm_aggregated_df(adata, [SIG], COVID_DESIGN, "DFO_8-14")
    assert len(df) == df["participant_id"].nunique() == 9
    ref = st.between_arm_comparison(adata, visit="DFO_8-14", features=[SIG], design=COVID_DESIGN)
    assert st.between_arm_fit_beta(df, SIG, COVID_DESIGN) == pytest.approx(
        ref["beta_arm"].iloc[0], abs=1e-12
    )


def test_table4_covid_null_permutes_the_observed_statistic(load_figure_module, monkeypatch, tmp_path):
    """Observed and null are both the DFO_8-14 between-arm beta (it was a D0->D28 DiD null)."""
    adata = _covid_cells()
    tables = _tables_with(load_figure_module, monkeypatch, tmp_path, get_stephenson=lambda: adata)
    monkeypatch.setattr(tables, "score_signatures", lambda ad, layer=None: (ad, [SIG]))
    n_perm = 40
    out = tables.table4_permutation_results(n_permutations=n_perm, n_jobs=1)
    row = out.loc[out["dataset"] == "COVID-19"].iloc[0]

    df = st.get_between_arm_aggregated_df(adata, [SIG], COVID_DESIGN, "DFO_8-14", layer="log1p_cpm")
    observed = st.between_arm_fit_beta(df, SIG, COVID_DESIGN)
    assert row["observed_beta"] == pytest.approx(observed, abs=1e-12)

    seed = np.random.default_rng(42).integers(0, 2**31)
    null = []
    for i in range(n_perm):
        perm = df.copy()
        perm["arm_bin"] = np.random.default_rng(seed + i).permutation(perm["arm_bin"].values)
        null.append(st.between_arm_fit_beta(perm, SIG, COVID_DESIGN))
    null = np.asarray(null)
    expected_p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    assert row["permutation_p"] == pytest.approx(expected_p)
    assert row["null_mean"] == pytest.approx(null.mean())


def test_unpaired_module_did_skips_an_arm_missing_a_visit():
    design = st.TrialDesign(participant_col="participant", visit_col="visit", arm_col="arm",
                            arm_treated="T", arm_control="C", celltype_col=None)
    rng = np.random.default_rng(2)
    rows = []
    for pool in ("complete", "treated_pre_only"):
        for p in range(4):
            arm = "T" if p < 2 else "C"
            for visit in ("Pre", "Post"):
                if pool == "treated_pre_only" and arm == "T" and visit == "Post":
                    continue
                rows.append({"pool": pool, "module": "m", "participant": f"{pool}{p}",
                             "visit": visit, "arm": arm, "module_score": rng.normal()})
    pb = pd.DataFrame(rows)
    pb["visit"] = pd.Categorical(pb["visit"], categories=["Pre", "Post"])
    with pytest.warns(UserWarning):
        out = module_score_did_by_pool(pb, design, ("Pre", "Post"), allow_unpaired=True, n_perm=10)
    assert out["pool"].tolist() == ["complete"]


def test_diagnose_counts_only_cell_types_present(sample_adata, trial_design):
    sample_adata.obs["celltype"] = pd.Categorical(
        sample_adata.obs["celltype"], categories=["TypeA", "TypeB", "TypeC"]
    )
    report = diagnose_trial_data(sample_adata, trial_design, verbose=False)
    counts = sample_adata.obs["celltype"].astype(str).value_counts().to_dict()
    assert report["n_celltypes"] == 2
    assert report["celltype_distribution"] == counts


def test_permute_arms_ignores_unused_participant_levels():
    obs = pd.DataFrame({
        "participant_id": pd.Categorical(
            ["P1", "P1", "P2", "P3", "P4"], categories=["P1", "P2", "P3", "P4", "P5", "P6"]
        ),
        "arm": ["T", "T", "T", "C", "C"],
    }, index=[str(i) for i in range(5)])
    adata = AnnData(X=np.zeros((5, 1)), obs=obs)
    for seed in range(20):
        out = _permute_arms(adata, "participant_id", "arm", np.random.default_rng(seed))
        per = out.obs.groupby("participant_id", observed=True)["arm"].first()
        assert sorted(per.astype(str)) == ["C", "C", "T", "T"], seed


def test_cell_vs_participant_panel_reports_spearman_below_the_axes(load_figure_module):
    """Figure 2B and Supp Figure 3B share one panel: Spearman rho, key outside the data."""
    from scipy import stats

    shared = load_figure_module("main/figure2_tnbc_analysis")  # loads _shared as well
    plot = shared.plot_cell_vs_participant_beta
    feats = [f"sig_S{i}" for i in range(8)]
    rng = np.random.default_rng(3)
    cell = pd.DataFrame({"feature": feats, "beta_DiD": rng.normal(size=8)})
    part = pd.DataFrame({"feature": feats, "beta_DiD": rng.normal(size=8)})
    for composite in (False, True):
        fig, ax = plt.subplots()
        plot(ax, cell, part, title="t", composite=composite)
        fig.canvas.draw()
        legend = ax.get_legend()
        rho, p = stats.spearmanr(cell["beta_DiD"], part["beta_DiD"])
        assert f"ρ = {rho:.2f}, p = {p:.1e}" in [t.get_text() for t in legend.get_texts()]
        legend_box = legend.get_window_extent()
        assert legend_box.y1 < ax.get_window_extent().y0
        plt.close(fig)
