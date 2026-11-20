import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import sctrial as st
from sctrial import TrialDesign
from sctrial.stats.mixed_effects import did_mixed, did_table_mixed


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


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

_FIT_KEYS = (
    "beta_DiD",
    "se_DiD",
    "p_DiD",
    "ci_lower",
    "ci_upper",
    "var_participant",
    "var_residual",
    "icc",
)
_MIXED_KW = {
    "outcome": "y",
    "participant_col": "pid",
    "time_col": "visit",
    "arm_col": "arm",
    "arm_treated": "Treated",
}


def _long_frame(n_per_arm=8, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ("Treated", "Control"):
        for p in range(n_per_arm):
            level = rng.normal(0, 0.5)
            for visit in ("V1", "V2"):
                rows.append((f"{arm[0]}{p:02d}", arm, visit, level + rng.normal(), rng.normal()))
    return pd.DataFrame(rows, columns=["pid", "arm", "visit", "y", "x"])


def _cohort(n_per_arm=6, n_cells=3, n_genes=6, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ("Treated", "Control"):
        for p in range(n_per_arm):
            for visit in ("V1", "V2"):
                rows.extend((f"{arm[0]}{p:02d}", arm, visit) for _ in range(n_cells))
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    pids = sorted(obs["participant_id"].unique())
    codes = obs["participant_id"].map({p: i for i, p in enumerate(pids)}).to_numpy()
    X = rng.normal(size=(len(obs), n_genes)) + rng.normal(0, 0.5, (len(pids), n_genes))[codes]
    genes = [f"G{i}" for i in range(n_genes)]
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col=None,
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes)), genes, design


def _row(table, feature):
    return table.set_index("feature").loc[feature]


def _assert_same_fit(row, reference):
    for key in _FIT_KEYS:
        assert row[key] == pytest.approx(reference[key], rel=1e-10, abs=1e-12, nan_ok=True), key


@pytest.mark.parametrize("column", ["y", "x", "pid", "visit"])
def test_did_mixed_fits_the_rows_that_hold_every_variable_of_the_model(column):
    """A row with a missing outcome, covariate, participant or visit is not in the model.

    statsmodels drops such a row from the model matrices but not from the participant labels
    passed beside the formula, which raised an IndexError.
    """
    df = _long_frame()
    holed = df.copy()
    holed[column] = holed[column].astype(object if column in ("pid", "visit") else float)
    holed.loc[[3, 10, 21], column] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_mixed(holed, covariates=["x"], visits=("V1", "V2"), **_MIXED_KW)
        ref = did_mixed(df.drop(index=[3, 10, 21]), covariates=["x"], visits=("V1", "V2"), **_MIXED_KW)

    assert np.isfinite(ref["beta_DiD"])
    _assert_same_fit(res, ref)


def test_did_mixed_returns_missing_when_one_visit_has_no_outcome():
    df = _long_frame()
    df.loc[df["visit"] == "V2", "y"] = np.nan

    res = did_mixed(df, visits=("V1", "V2"), **_MIXED_KW)

    assert np.isnan(res["beta_DiD"])
    assert res["converged"] is False


def test_did_mixed_returns_missing_when_the_frame_holds_other_labels_than_visits():
    res = did_mixed(_long_frame(), visits=("V1", "V9"), **_MIXED_KW)

    assert np.isnan(res["beta_DiD"])
    assert res["converged"] is False
    assert "V9" in res["error"]


@pytest.mark.parametrize("standardize", [True, False])
def test_did_table_mixed_uses_the_participants_observed_at_both_visits_of_each_feature(standardize):
    """As in ``did_table``: a participant without a value at one visit is not in that feature's model."""
    adata, genes, design = _cohort()
    obs = adata.obs
    t00, c03 = (obs["participant_id"] == "T00").to_numpy(), (obs["participant_id"] == "C03").to_numpy()
    X = adata.X.copy()
    X[t00 & (obs["visit"] == "V2").to_numpy(), 1] = np.nan  # one visit of T00
    X[c03, 2] = np.nan  # every value of C03
    X[:, 3] = np.nan  # no value at all
    one_cell = np.flatnonzero(t00 & (obs["visit"] == "V1").to_numpy())[0]
    X[one_cell, 4] = np.nan  # one cell of three: the visit keeps its other cells
    holed = ad.AnnData(X=X, obs=obs.copy(), var=adata.var.copy())
    kw = {"design": design, "visits": ("V1", "V2"), "standardize": standardize}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_table_mixed(holed, genes, **kw)
        complete = did_table_mixed(adata, genes, **kw)
        without_t00 = did_table_mixed(adata[~t00].copy(), ["G1"], **kw)
        without_c03 = did_table_mixed(adata[~c03].copy(), ["G2"], **kw)
        keep = np.ones(adata.n_obs, dtype=bool)
        keep[one_cell] = False
        without_cell = did_table_mixed(adata[keep].copy(), ["G4"], **kw)

    assert list(res["feature"]) == genes
    n = obs["participant_id"].nunique()
    assert _row(res, "G0")["n_units"] == n
    _assert_same_fit(_row(res, "G0"), _row(complete, "G0"))
    _assert_same_fit(_row(res, "G5"), _row(complete, "G5"))
    assert _row(res, "G1")["n_units"] == n - 1
    _assert_same_fit(_row(res, "G1"), _row(without_t00, "G1"))
    assert _row(res, "G2")["n_units"] == n - 1
    _assert_same_fit(_row(res, "G2"), _row(without_c03, "G2"))
    assert _row(res, "G3")["n_units"] == 0
    assert np.isnan(_row(res, "G3")["beta_DiD"])
    assert not _row(res, "G3")["converged"]
    assert _row(res, "G4")["n_units"] == n
    _assert_same_fit(_row(res, "G4"), _row(without_cell, "G4"))
    assert np.isfinite(res.loc[res["feature"] != "G3", "beta_DiD"]).all()


def test_did_table_mixed_at_cell_level_drops_the_cells_without_a_value():
    adata, genes, design = _cohort()
    obs = adata.obs
    t00 = (obs["participant_id"] == "T00").to_numpy()
    X = adata.X.copy()
    cells = np.flatnonzero(t00 & (obs["visit"] == "V1").to_numpy())[:2]
    X[cells, 0] = np.nan  # two cells of three
    X[t00 & (obs["visit"] == "V2").to_numpy(), 1] = np.nan  # every cell of one visit
    holed = ad.AnnData(X=X, obs=obs.copy(), var=adata.var.copy())
    kw = {"design": design, "visits": ("V1", "V2"), "aggregate": "cell"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_table_mixed(holed, ["G0", "G1"], **kw)
        keep = np.ones(adata.n_obs, dtype=bool)
        keep[cells] = False
        without_cells = did_table_mixed(adata[keep].copy(), ["G0"], **kw)
        without_t00 = did_table_mixed(adata[~t00].copy(), ["G1"], **kw)

    n = obs["participant_id"].nunique()
    assert _row(res, "G0")["n_units"] == n
    _assert_same_fit(_row(res, "G0"), _row(without_cells, "G0"))
    assert _row(res, "G1")["n_units"] == n - 1
    _assert_same_fit(_row(res, "G1"), _row(without_t00, "G1"))


def _with_covariates(adata, seed=11):
    """A numeric value and a label for each participant and visit, copied to its cells."""
    rng = np.random.default_rng(seed)
    obs = adata.obs
    keys = obs["participant_id"].astype(str) + "|" + obs["visit"].astype(str)
    levels = sorted(keys.unique())
    obs["load"] = keys.map(dict(zip(levels, rng.normal(size=len(levels))))).to_numpy()
    obs["batch"] = keys.map({k: f"b{i % 3}" for i, k in enumerate(levels)}).to_numpy()
    return adata


def _participant_visit_frame(adata, gene):
    frame = adata.obs[["participant_id", "arm", "visit", "load", "batch"]].copy()
    frame["y"] = np.asarray(adata.X)[:, list(adata.var_names).index(gene)]
    return (
        frame.groupby(["participant_id", "visit", "arm"], observed=True)
        .agg(y=("y", "mean"), load=("load", "mean"), batch=("batch", "first"))
        .reset_index()
    )


@pytest.mark.parametrize("covariates", [["load"], ["batch"], ["load", "batch"]])
def test_did_table_mixed_adjusts_for_covariates_of_obs(covariates):
    """The covariates were read from the feature columns only, so one of ``obs`` was not in the model."""
    adata, genes, design = _cohort(n_per_arm=10)
    adata = _with_covariates(adata)
    kw = {"design": design, "visits": ("V1", "V2"), "standardize": False}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_table_mixed(adata, ["G0"], covariates=covariates, **kw)
        plain = did_table_mixed(adata, ["G0"], **kw)
        ref = did_mixed(
            _participant_visit_frame(adata, "G0"),
            outcome="y",
            participant_col="participant_id",
            time_col="visit",
            arm_col="arm",
            arm_treated="Treated",
            covariates=covariates,
            visits=("V1", "V2"),
        )

    assert np.isfinite(ref["beta_DiD"])
    _assert_same_fit(_row(res, "G0"), ref)
    assert _row(res, "G0")["beta_DiD"] != pytest.approx(_row(plain, "G0")["beta_DiD"], rel=1e-6)


def test_did_table_mixed_drops_the_rows_without_a_covariate_value():
    adata, genes, design = _cohort()
    adata = _with_covariates(adata)
    obs = adata.obs
    c01 = (obs["participant_id"] == "C01").to_numpy()
    holed = adata.copy()
    holed.obs.loc[c01 & (obs["visit"] == "V1").to_numpy(), "load"] = np.nan
    kw = {"design": design, "visits": ("V1", "V2"), "covariates": ["load"]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_table_mixed(holed, ["G0", "G5"], **kw)
        ref = did_table_mixed(adata[~c01].copy(), ["G0", "G5"], **kw)

    assert _row(res, "G0")["n_units"] == obs["participant_id"].nunique() - 1
    _assert_same_fit(_row(res, "G0"), _row(ref, "G0"))


def test_mixed_model_refuses_covariates_it_cannot_use():
    adata, genes, design = _cohort()
    adata = _with_covariates(adata)
    kw = {"design": design, "visits": ("V1", "V2")}

    with pytest.raises(KeyError, match="not found in adata.obs"):
        did_table_mixed(adata, ["G0"], covariates=["absent"], **kw)
    with pytest.raises(ValueError, match="also a feature"):
        did_table_mixed(adata, ["G0", "G1"], covariates=["G1"], **kw)
    adata.obs["by_cell"] = np.where(np.arange(adata.n_obs) % 2, "x", "y")
    with pytest.raises(ValueError, match="varies within participant-visit"):
        did_table_mixed(adata, ["G0"], covariates=["by_cell"], **kw)
    with pytest.raises(KeyError, match="covariates not found in the table"):
        did_mixed(_long_frame(), covariates=["absent"], **_MIXED_KW)


def test_compare_fixed_vs_mixed_fits_both_models_on_the_same_participants():
    adata, genes, design = _cohort()
    obs = adata.obs
    t00 = (obs["participant_id"] == "T00").to_numpy()
    X = adata.X.copy()
    X[t00 & (obs["visit"] == "V2").to_numpy(), 1] = np.nan
    holed = ad.AnnData(X=X, obs=obs.copy(), var=adata.var.copy())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comp = st.compare_fixed_vs_mixed(holed, ["G0", "G1"], design, ("V1", "V2"))
        ref = st.compare_fixed_vs_mixed(adata[~t00].copy(), ["G1"], design, ("V1", "V2"))

    got, want = _row(comp, "G1"), _row(ref, "G1")
    for key in ("beta_fixed", "se_fixed", "beta_mixed", "se_mixed", "p_mixed", "icc"):
        assert got[key] == pytest.approx(want[key], rel=1e-10), key
