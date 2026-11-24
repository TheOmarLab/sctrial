"""A coefficient that the design does not identify is not reported.

Least squares returns numbers for a regressor that equals another one or a combination of
others, split arbitrarily among them, with a standard error and a p-value. Each function
below reports a missing estimate in that case.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sctrial import TrialDesign
from sctrial.stats.abundance import abundance_did
from sctrial.stats.comparisons import between_arm_comparison
from sctrial.stats.cv import influence_diagnostics, loo_cv_did
from sctrial.stats.did import did_fit
from sctrial.stats.mixed_effects import compare_fixed_vs_mixed
from sctrial.stats.summary import summarize_did_results
from sctrial.stats.timeseries import polynomial_trend, trend_interaction
from sctrial.stats.timeseries import test_parallel_trends as parallel_trends

GENES = ["G0", "G1"]
ESTIMATE = ["beta_trend", "beta_treat_trend", "se_treat_trend", "p_treat_trend"]


def _cohort(n_per_arm=10, visits=("V1", "V2", "V3"), n_cells=2, seed=1):
    rng = np.random.default_rng(seed)
    rows = [
        (f"{arm[0]}{p:02d}", arm, visit)
        for arm in ("Treated", "Control")
        for p in range(n_per_arm)
        for visit in visits
        for _ in range(n_cells)
    ]
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    pids = sorted(obs["participant_id"].unique())
    codes = obs["participant_id"].map({p: i for i, p in enumerate(pids)}).to_numpy()
    step = obs["visit"].map({v: i for i, v in enumerate(visits)}).to_numpy(dtype=float)
    X = rng.normal(size=(len(obs), 2)) + rng.normal(0, 0.5, (len(pids), 2))[codes] + 0.5 * step[:, None]
    obs["celltype"] = np.where(rng.random(len(obs)) < 0.5, "A", "B")
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=GENES)), design


def _without(adata, gene, arm, visit):
    X = adata.X.copy()
    X[((adata.obs["arm"] == arm) & (adata.obs["visit"] == visit)).to_numpy(), GENES.index(gene)] = np.nan
    return ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())


@pytest.mark.parametrize("model", ["linear", "quadratic"])
@pytest.mark.parametrize("arm", ["Control", "Treated"])
def test_trend_needs_both_arms_for_the_feature(arm, model):
    # Every participant of one arm lacks a visit for G1, so only the other arm is in its model.
    # The visit and the arm-by-visit columns are then one column (or the second is zero), and
    # the fit reported an interaction with p = 0.008 from one arm.
    adata, design = _cohort()
    visits = ["V1", "V2", "V3"]
    holed = _without(adata, "G1", arm, "V2")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = trend_interaction(holed, GENES, design, visits, model=model).set_index("feature")
        complete = trend_interaction(adata, GENES, design, visits, model=model).set_index("feature")

    assert "error" not in res.columns
    assert res.loc["G1", "n_units"] == 10
    assert res.loc["G1", ESTIMATE].isna().all()
    assert np.isnan(res.loc["G1", "FDR_treat_trend"])
    for column in ESTIMATE:
        assert res.loc["G0", column] == complete.loc["G0", column]


@pytest.mark.parametrize("arm", ["Control", "Treated"])
def test_parallel_trends_says_when_the_test_is_not_estimable(arm):
    adata, design = _cohort()
    visits = ["V1", "V2", "V3"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = parallel_trends(_without(adata, "G1", arm, "V2"), GENES, design, pre_visits=visits)
    res = res.set_index("feature")

    assert res.loc["G1", "warning"] == "Not estimable"
    assert np.isnan(res.loc["G1", "p_pretrend"]) and np.isnan(res.loc["G1", "beta_pretrend"])
    assert res.loc["G0", "warning"] in ("OK", "Potential violation")
    assert np.isfinite(res.loc["G0", "p_pretrend"])


def test_trend_gives_no_estimate_for_more_terms_than_the_visits_allow():
    # Three visits fix a quadratic. In a cubic model the three arm-by-time columns are one
    # combination of each other, so none of their coefficients is identified.
    adata, design = _cohort()
    visits = ["V1", "V2", "V3"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubic = trend_interaction(adata, GENES, design, visits, model="cubic")
        quadratic = trend_interaction(adata, GENES, design, visits, model="quadratic")
        four = trend_interaction(*_cohort(visits=("V1", "V2", "V3", "V4"))[:1], GENES, design,
                                 ["V1", "V2", "V3", "V4"], model="cubic")

    columns = [*ESTIMATE, "beta_treat_trend2", "p_treat_trend2", "beta_treat_trend3", "p_treat_trend3"]
    assert cubic[columns].isna().all().all()
    # one column order, with or without an estimate
    order = ["feature", *ESTIMATE, "n_units", "n_timepoints", *columns[4:], "FDR_treat_trend"]
    assert list(cubic.columns) == order and list(four.columns) == order
    assert np.isfinite(quadratic[[*ESTIMATE, "beta_treat_trend2", "p_treat_trend2"]]).all().all()
    assert np.isfinite(four[columns]).all().all()


def test_trend_blanks_the_higher_terms_that_determine_each_other():
    # At the times -2/3, 1 and 2 the square and the cube of time are one combination of each
    # other and a constant, while time itself is not. The linear interaction is identified;
    # the quadratic and the cubic ones are not, and a fit on independent columns would hand
    # one of them the share of both.
    adata, design = _cohort()
    visits = ["V1", "V2", "V3"]
    times = [-2.0 / 3.0, 1.0, 2.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubic = trend_interaction(adata, GENES, design, visits, time_values=times, model="cubic")
        linear = trend_interaction(adata, GENES, design, visits, time_values=times, model="linear")

    assert np.isfinite(cubic[ESTIMATE]).all().all()
    assert np.isfinite(linear[ESTIMATE]).all().all()
    higher = ["beta_treat_trend2", "p_treat_trend2", "beta_treat_trend3", "p_treat_trend3"]
    assert cubic[higher].isna().all().all()


def test_trend_with_as_many_columns_as_rows_does_not_raise():
    # One participant per arm, three visits, a quadratic: six rows and six parameters. The
    # cluster correction divided by rows minus columns.
    adata, design = _cohort(n_per_arm=1)
    visits = ["V1", "V2", "V3"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = trend_interaction(adata, GENES, design, visits, model="quadratic")
        three = trend_interaction(*_cohort(n_per_arm=2)[:1], GENES, design, visits, model="cubic")

    assert "error" not in res.columns
    assert np.isfinite(res["beta_treat_trend"]).all() and res["p_treat_trend"].isna().all()
    assert three["p_treat_trend"].isna().all()


@pytest.mark.parametrize("kind", ["label", "number"])
def test_between_arm_model_gives_no_estimate_when_a_covariate_holds_the_arm(kind):
    # A site that enrolled one arm only, or a dose that is zero in the control arm: the arm
    # difference was split between the two columns (half of it, or a tenth) and tested.
    adata, design = _cohort(visits=("V1", "V2"))
    treated = (adata.obs["arm"] == "Treated").to_numpy()
    adata.obs["held"] = np.where(treated, "s1", "s2") if kind == "label" else treated * 3.0
    adata.obs["load"] = np.random.default_rng(2).normal(size=adata.n_obs)
    kw = dict(visit="V2", features=GENES, design=design)

    res = between_arm_comparison(adata, covariates=["held"], **kw)
    plain = between_arm_comparison(adata, covariates=["load"], **kw)
    adata.obs["load_again"] = adata.obs["load"] * 2.0
    doubled = between_arm_comparison(adata, covariates=["load", "load_again"], **kw)

    assert res[["beta_arm", "se_arm", "p_arm", "ci_lo_arm", "ci_hi_arm"]].isna().all().all()
    assert (res["n_units"] == 20).all()
    for column in ("beta_arm", "se_arm", "p_arm"):
        assert doubled[column].to_numpy() == pytest.approx(plain[column].to_numpy(), rel=1e-9)


def test_abundance_gives_no_estimate_when_a_covariate_holds_the_arm():
    adata, design = _cohort(visits=("V1", "V2"), n_cells=30)
    adata.obs["dose"] = (adata.obs["arm"] == "Treated").astype(float) * 3.0
    adata.obs["age"] = adata.obs["participant_id"].map(
        {p: 40.0 + 3 * i for i, p in enumerate(sorted(adata.obs["participant_id"].unique()))}
    )
    kw = dict(design=design, visits=("V1", "V2"), min_units=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = abundance_did(adata, **kw)
        held = abundance_did(adata, covariates=["dose"], **kw)
        adjusted = abundance_did(adata, covariates=["age"], **kw)

    assert len(plain) == 2 and np.isfinite(plain["p_DiD"]).all()
    assert len(adjusted) == 2 and np.isfinite(adjusted["p_DiD"]).all()
    assert held.empty or held["p_DiD"].isna().all()


def test_polynomial_needs_one_more_visit_than_its_degree():
    # Two visits fix a line. The default quadratic had the square of time equal to time and
    # returned a share of the slope for each, with a p-value, and a curve between the visits.
    two, design = _cohort(visits=("V1", "V2"))
    three, _ = _cohort()
    four, _ = _cohort(visits=("V1", "V2", "V3", "V4"))

    with pytest.raises(ValueError, match=r"arm_bin:time_num2 not identified for 'G0'.*degree 2 needs 3"):
        polynomial_trend(two, "G0", design, visits=("V1", "V2"))
    with pytest.raises(ValueError, match=r"arm_bin:time_num3 not identified.*degree 3 needs 4"):
        polynomial_trend(three, "G0", design, visits=("V1", "V2", "V3"), degree=3)

    line = polynomial_trend(two, "G0", design, visits=("V1", "V2"), degree=1)
    curve = polynomial_trend(three, "G0", design, visits=("V1", "V2", "V3"))
    cubic = polynomial_trend(four, "G0", design, visits=("V1", "V2", "V3", "V4"), degree=3)
    assert np.isfinite([line["coefficients"]["arm_bin:time_num"], line["pvalues"]["arm_bin:time_num"]]).all()
    # a participant of each arm without the middle visit: fewer rows, the same terms
    gaps = three.X.copy()
    gaps[(three.obs["participant_id"].isin(["T00", "C00"]) & (three.obs["visit"] == "V2")).to_numpy(), 0] = np.nan
    unbalanced = polynomial_trend(ad.AnnData(X=gaps, obs=three.obs.copy(), var=three.var.copy()), "G0", design,
                                  visits=("V1", "V2", "V3"))
    for fit, top in ((curve, "arm_bin:time_num2"), (cubic, "arm_bin:time_num3"), (unbalanced, "arm_bin:time_num2")):
        assert np.isfinite([fit["coefficients"][top], fit["pvalues"][top]]).all()
        assert np.isfinite(fit["predictions"]["predicted"]).all()
    assert len(unbalanced["residuals"]) == len(curve["residuals"]) - 2


@pytest.mark.parametrize("arm", ["Control", "Treated"])
def test_polynomial_needs_both_arms(arm):
    adata, design = _cohort()
    X = adata.X.copy()
    X[(adata.obs["arm"] == arm).to_numpy(), 0] = np.nan
    one_arm = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())

    with pytest.raises(ValueError, match="not identified"):
        polynomial_trend(one_arm, "G0", design, visits=("V1", "V2", "V3"), degree=1)


def test_influence_is_not_decided_for_a_fit_that_gave_no_estimate():
    loo = pd.DataFrame(
        {
            "feature": ["G0"] * 3,
            "excluded": ["P1", "P2", "P3"],
            "beta_DiD": [0.4, np.nan, 1.9],
            "se_DiD": [0.2, np.nan, 0.2],
            "beta_full": [0.5] * 3,
            "influence": [0.5, np.nan, 7.0],
        }
    )

    flags = influence_diagnostics(loo, threshold=1.0)["is_influential"]

    assert flags.isna().tolist() == [False, True, False]
    assert not flags[0] and flags[2]
    assert flags.sum() == 1


def test_influence_of_a_feature_without_an_estimate_is_missing_in_the_leave_one_out_table():
    # G1 is constant: no fit of it has an estimate, with or without any participant.
    adata, design = _cohort(n_per_arm=3, visits=("V1", "V2"))
    X = adata.X.copy()
    X[:, 1] = 1.0
    constant = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = influence_diagnostics(loo_cv_did(constant, GENES, design, ("V1", "V2"))).set_index("feature")

    assert len(table) == 12
    assert table.loc["G1", "is_influential"].isna().all() and table.loc["G1", "influence"].isna().all()
    assert table.loc["G0", "is_influential"].notna().all() and np.isfinite(table.loc["G0", "influence"]).all()


def test_fixed_and_mixed_agreement_is_not_decided_without_both_tests():
    # G1 is constant, so neither model gives an estimate. G2 follows the arm means exactly, so
    # the fixed-effects fit has an estimate and no test. Neither row is an agreement or a
    # disagreement.
    adata, design = _cohort(visits=("V1", "V2"))
    X = np.column_stack([adata.X, np.zeros(adata.n_obs)])
    X[:, 1] = 1.0
    treated = (adata.obs["arm"] == "Treated").to_numpy()
    later = (adata.obs["visit"] == "V2").to_numpy()
    X[:, 2] = adata.obs["participant_id"].str[1:].astype(float).to_numpy() + later * (1.0 + 2.0 * treated)
    three = ad.AnnData(X=X, obs=adata.obs.copy(), var=pd.DataFrame(index=[*GENES, "G2"]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        comp = compare_fixed_vs_mixed(three, [*GENES, "G2"], design, ("V1", "V2")).set_index("feature")

    assert np.isnan(comp.loc["G1", "p_fixed"]) and np.isnan(comp.loc["G1", "p_mixed"])
    assert np.isfinite(comp.loc["G2", "beta_fixed"]) and np.isnan(comp.loc["G2", "p_fixed"])
    assert comp["agreement"].isna().tolist() == [False, True, True]
    assert comp.loc["G0", "agreement"] in (True, False)


def test_signature_forest_counts_and_draws_the_signatures_that_were_tested(load_figure_module):
    # A signature without an estimate was counted in the denominator of the share printed on
    # the panel and drawn in the colour of a negative effect.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection

    f4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    table = pd.DataFrame(
        {
            "feature": ["sig_a", "sig_b", "sig_c", "sig_d"],
            "beta_DiD": [0.8, -0.4, np.nan, 0.1],
            "se_DiD": [0.2, 0.2, np.nan, 0.2],
            "FDR_DiD": [0.01, 0.4, np.nan, 0.6],
        }
    )
    fig, ax = plt.subplots()
    f4.panel_D(ax, {"did_sig": table})

    assert "1/3 FDR < 0.1" in [t.get_text() for t in ax.texts]
    assert len(ax.get_yticks()) == 3
    assert sum(isinstance(c, PathCollection) for c in ax.collections) == 3
    plt.close(fig)


def test_leave_one_out_panel_has_no_value_where_a_fold_has_no_estimate(load_figure_module):
    # G1 is measured in three treated participants and one control. Without any of them its
    # model has fewer than four participants or one arm. Those folds were skipped, the
    # remaining folds leave out a participant who is not in the model of G1, and the panel
    # showed a deviation of zero: the most stable feature of the cohort.
    adata, design = _cohort(n_per_arm=6, visits=("V1", "V2"))
    measured = adata.obs["participant_id"].isin(["T00", "T01", "T02", "C00"]).to_numpy()
    X = adata.X.copy()
    X[~measured, 1] = np.nan
    toy = ad.AnnData(X=X, obs=adata.obs.copy(), var=adata.var.copy())
    toy.layers["log1p_norm"] = toy.X.copy()

    # With one control participant in the cohort, the table without that participant raises
    # (the data hold one arm). The fold was dropped and every feature kept a value.
    one_control = ~adata.obs["participant_id"].isin([f"C{i:02d}" for i in range(1, 6)]).to_numpy()
    thin = adata[one_control].copy()
    thin.layers["log1p_norm"] = thin.X.copy()
    whole = adata.copy()
    whole.layers["log1p_norm"] = whole.X.copy()

    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    sf5._FEATURES = list(GENES)
    cohort = {
        "design": "two_arm",
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "arm",
        "arm_treated": "Treated",
        "arm_control": "Control",
        "visits": ("V1", "V2"),
    }
    sf5._MDE_DATASET_CFG = {
        "Toy": {**cohort, "loader": lambda: toy},
        "Thin": {**cohort, "loader": lambda: thin},
        "Whole": {**cohort, "loader": lambda: whole},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = sf5._compute_loo_data()

    assert np.isfinite(mat.loc["G0", "Toy"]) and mat.loc["G0", "Toy"] > 0
    assert np.isnan(mat.loc["G1", "Toy"])
    assert mat["Thin"].isna().all()
    assert np.isfinite(mat["Whole"]).all() and (mat["Whole"] > 0).all()


def test_fixed_and_mixed_agreement_for_each_pattern_of_missing_results(monkeypatch):
    # The two tables are given, so that every pattern is present: both tests, one test, no
    # estimate on one side, nothing on either side.
    nan = np.nan
    rows = {
        #          beta_fixed p_fixed beta_mixed p_mixed  expected
        "same":      (0.9, 0.01, 0.8, 0.02, True),
        "neither":   (0.2, 0.60, 0.1, 0.70, True),
        "one_only":  (0.9, 0.01, 0.8, 0.30, False),
        "opposite":  (0.9, 0.01, -0.8, 0.02, False),
        "no_fixed_test": (0.4, nan, 0.4, 0.40, None),
        "no_mixed_test": (0.4, 0.40, 0.4, nan, None),
        "no_mixed_fit":  (0.4, 0.40, nan, nan, None),
        "no_fixed_fit":  (nan, nan, 0.4, 0.40, None),
        "nothing":       (nan, nan, nan, nan, None),
    }
    names = list(rows)
    fixed = pd.DataFrame({"feature": names, "beta_DiD": [rows[n][0] for n in names], "se_DiD": 0.1,
                          "p_DiD": [rows[n][1] for n in names]})
    mixed = pd.DataFrame({"feature": names, "beta_DiD": [rows[n][2] for n in names], "se_DiD": 0.1,
                          "p_DiD": [rows[n][3] for n in names], "icc": 0.2})
    monkeypatch.setattr("sctrial.stats.did.did_table", lambda *a, **k: fixed)
    monkeypatch.setattr("sctrial.stats.mixed_effects.did_table_mixed", lambda *a, **k: mixed)
    adata, design = _cohort(visits=("V1", "V2"))

    comp = compare_fixed_vs_mixed(adata, names, design, ("V1", "V2")).set_index("feature")

    for name in names:
        expected = rows[name][4]
        if expected is None:
            assert pd.isna(comp.loc[name, "agreement"]), name
        else:
            assert comp.loc[name, "agreement"] is not pd.NA and bool(comp.loc[name, "agreement"]) is expected, name
    assert comp["agreement"].mean() == 0.5


def test_summary_counts_the_features_that_were_tested():
    # A feature without a p-value was counted under "features tested", and one with an
    # estimate and no test was listed with "p=nan".
    table = pd.DataFrame(
        {
            "feature": ["G1", "G2", "G3", "G4"],
            "beta_DiD": [1.5, -2.0, np.nan, 0.7],
            "p_DiD": [0.001, 0.04, np.nan, np.nan],
            "FDR_DiD": [0.01, 0.1, np.nan, np.nan],
        }
    )

    summary = summarize_did_results(table)
    complete = summarize_did_results(table.iloc[:2])

    assert "Total features tested: 2" in summary
    assert "Features without a test: 2" in summary
    assert "G3" not in summary and "G4" not in summary and "nan" not in summary
    assert "Total features tested: 2" in complete and "without a test" not in complete
    assert summary.replace("Features without a test: 2\n", "") == complete


def test_leave_one_out_panel_has_no_value_when_a_fold_raises(load_figure_module, monkeypatch):
    # A fold whose table raised was dropped and the largest deviation taken over the others.
    import sctrial

    adata, design = _cohort(n_per_arm=4, visits=("V1", "V2"))
    adata.layers["log1p_norm"] = adata.X.copy()
    fitted = sctrial.did_table

    def raising_without_t00(data, *args, **kwargs):
        if "T00" not in set(data.obs["participant_id"]):
            raise ValueError("no table for this fold")
        return fitted(data, *args, **kwargs)

    sf5 = load_figure_module("supp/supp_fig5_sensitivity_robustness")
    sf5._FEATURES = list(GENES)
    sf5._MDE_DATASET_CFG = {
        "Toy": {
            "design": "two_arm",
            "loader": lambda: adata,
            "layer": "log1p_norm",
            "participant_col": "participant_id",
            "visit_col": "visit",
            "arm_col": "arm",
            "arm_treated": "Treated",
            "arm_control": "Control",
            "visits": ("V1", "V2"),
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complete = sf5._compute_loo_data()
        monkeypatch.setattr(sctrial, "did_table", raising_without_t00)
        with_gap = sf5._compute_loo_data()

    assert np.isfinite(complete["Toy"]).all()
    assert with_gap["Toy"].isna().all()


@pytest.mark.parametrize("step", [0.5, 21.0])
def test_abundance_gives_no_time_effect_when_a_covariate_holds_the_visit(step):
    # A covariate that changes by the same amount for everyone between the visits (a cycle
    # number, days on study) is the visit regressor. The interaction is identified and
    # unchanged; the time effect is not. The fit keeps the larger of the two columns, so with
    # a covariate in smaller units the whole time effect was reported for the visit, with a
    # p-value.
    adata, design = _cohort(visits=("V1", "V2"), n_cells=30)
    adata.obs["cycle"] = (adata.obs["visit"] == "V2").astype(float) * step
    adata.obs["load"] = np.random.default_rng(3).normal(size=adata.n_obs)
    kw = dict(design=design, visits=("V1", "V2"), min_units=4)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = abundance_did(adata, **kw).set_index("celltype")
        held = abundance_did(adata, covariates=["cycle"], **kw).set_index("celltype")
        other = abundance_did(adata, covariates=["load"], **kw).set_index("celltype")

    assert held[["beta_time", "p_time"]].isna().all().all()
    for column in ("beta_DiD", "se_DiD", "p_DiD"):
        assert held[column].to_numpy() == pytest.approx(plain.loc[held.index, column].to_numpy(), rel=1e-9)
    assert np.isfinite(plain[["beta_time", "p_time"]]).all().all()
    assert np.isfinite(other[["beta_time", "p_time", "beta_DiD", "p_DiD"]]).all().all()


@pytest.mark.parametrize("step", [0.5, 21.0])
def test_did_fit_gives_no_time_effect_when_a_covariate_holds_the_visit(step):
    # As in abundance_did: the covariate is the time regressor in other units, so the time
    # effect is not identified while the interaction is.
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        [(f"p{i:02d}", t, int(i < 6)) for i in range(12) for t in (0.0, 1.0)], columns=["pid", "time", "arm_bin"]
    )
    frame["y"] = rng.normal(size=len(frame)) + 0.6 * frame["time"] * frame["arm_bin"] + 0.3 * frame["time"]
    frame["cycle"] = frame["time"] * step
    frame["load"] = rng.normal(size=len(frame))

    plain = did_fit(frame, "y", "pid", "time", "arm_bin")
    held = did_fit(frame, "y", "pid", "time", "arm_bin", covariates=["cycle"])
    other = did_fit(frame, "y", "pid", "time", "arm_bin", covariates=["load"])

    assert np.isnan(held["beta_time"]) and np.isnan(held["p_time"])
    for key in ("beta_DiD", "se_DiD", "p_DiD"):
        assert held[key] == pytest.approx(plain[key], rel=1e-9)
    assert np.isfinite([plain["beta_time"], plain["p_time"], other["beta_time"], other["p_time"]]).all()


def test_naive_pseudobulk_comparator_gives_no_row_for_an_unidentified_interaction():
    # The treated arm seen after treatment only: the interaction column is the arm column.
    # The fit does not raise; it returned a share of the arm difference with a p-value.
    import statsmodels.formula.api as smf

    from sctrial.stats.simulation import _run_pseudobulk_ols

    rng = np.random.default_rng(6)
    rows = [(f"p{i:02d}", "Treated" if i < 5 else "Control", visit) for i in range(10) for visit in ("Pre", "Post")]
    complete = pd.DataFrame(rows, columns=["participant", "arm", "visit"])
    complete["gene_0"] = rng.normal(size=len(complete)) + 0.8 * (complete["arm"] == "Treated")
    one_cell_missing = complete[~((complete["arm"] == "Treated") & (complete["visit"] == "Pre"))]

    assert _run_pseudobulk_ols(one_cell_missing, ["gene_0"]) == {"gene_0": {}}

    row = _run_pseudobulk_ols(complete, ["gene_0"])["gene_0"]
    ref = smf.ols(
        "gene_0 ~ a + t + a:t",
        data=complete.assign(a=(complete["arm"] == "Treated") * 1.0, t=(complete["visit"] == "Post") * 1.0),
    ).fit()
    assert row["beta"] == pytest.approx(ref.params["a:t"], rel=1e-9)
    assert row["pvalue"] == pytest.approx(ref.pvalues["a:t"], rel=1e-9)
    assert (row["ci_lo"], row["ci_hi"]) == pytest.approx(tuple(ref.conf_int().loc["a:t"]), rel=1e-9)
