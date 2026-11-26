"""A model that fits its outcome exactly gives an estimate and no test.

Every residual is zero in exact arithmetic; the computed residuals are rounding error, and a
standard error or a p-value made of them is noise (0.0075 for a within-arm change that no
participant shows).
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from sctrial import TrialDesign
from sctrial.stats._utils import fits_exactly
from sctrial.stats.abundance import abundance_did
from sctrial.stats.comparisons import between_arm_comparison, within_arm_comparison
from sctrial.stats.did import did_fit, did_table
from sctrial.stats.heterogeneity import test_treatment_heterogeneity as heterogeneity
from sctrial.stats.module_scores import module_score_did_by_pool
from sctrial.stats.timeseries import trend_interaction

FEATURES = ["by_participant", "by_arm", "nudged", "noisy"]
VISITS = ("V1", "V2")


def _cohort(shift=0.0, scale=1.0, n_per_arm=8, seed=4):
    """Outcomes that the models fit exactly, one that misses by 1e-6, and an ordinary one."""
    rng = np.random.default_rng(seed)
    rows = []
    for arm in ("Treated", "Control"):
        for p in range(n_per_arm):
            for visit in ("V1", "V2", "V3"):
                rows.extend((f"{arm[0]}{p:02d}", arm, visit) for _ in range(2))
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit"])
    obs.index = [f"c{i}" for i in range(len(obs))]
    pids = sorted(obs["participant_id"].unique())
    level = obs["participant_id"].map({p: 0.37 * i for i, p in enumerate(pids)}).to_numpy()
    by_arm = np.where(obs["arm"] == "Treated", 1.3, 0.2)
    nudged = level.copy()
    nudged[((obs["participant_id"] == "T00") & (obs["visit"] == "V2")).to_numpy()] += 1e-6
    noisy = level + rng.normal(size=len(obs))
    X = shift + scale * np.column_stack([level, by_arm, nudged, noisy])
    obs["bmk"] = obs["participant_id"].map({p: float(i % 2) for i, p in enumerate(pids)})
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col=None,
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=FEATURES)), design


def _tables(adata, design, standardize):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "did": (did_table(adata, FEATURES, design, VISITS, standardize=standardize), "se_DiD", "p_DiD"),
            "did_boot": (
                did_table(adata, FEATURES, design, VISITS, standardize=standardize, use_bootstrap=True, n_boot=99),
                "se_DiD",
                "p_DiD",
            ),
            "within": (
                within_arm_comparison(adata, arm="Treated", features=FEATURES, design=design, visits=VISITS,
                                      standardize=standardize),
                "se_time",
                "p_time",
            ),
            "trend": (
                trend_interaction(adata, FEATURES, design, ["V1", "V2", "V3"]),
                "se_treat_trend",
                "p_treat_trend",
            ),
            "heterogeneity": (
                heterogeneity(adata, FEATURES, design, VISITS, "bmk", standardize=standardize),
                None,
                "p_heterogeneity",
            ),
        }


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("shift,scale", [(0.0, 1.0), (100.0, 1.0), (0.0, 1e-4), (0.0, 1e4), (1e4, 1e2)])
def test_longitudinal_tables_give_no_test_for_an_outcome_without_change(shift, scale, standardize):
    adata, design = _cohort(shift, scale)
    for name, (table, se, p) in _tables(adata, design, standardize).items():
        rows = table.set_index("feature")
        for feature in ("by_participant", "by_arm"):
            assert np.isnan(rows.loc[feature, p]), (name, feature)
            if se is not None:
                assert np.isnan(rows.loc[feature, se]), (name, feature)
        for feature in ("nudged", "noisy"):
            assert np.isfinite(rows.loc[feature, p]), (name, feature)


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("shift,scale", [(0.0, 1.0), (100.0, 1.0), (0.0, 1e-4), (1e4, 1e2)])
def test_between_arm_model_gives_no_test_when_the_arms_have_no_spread(shift, scale, standardize):
    # 1.3 in every treated participant and 0.2 in every control: the difference is exact and the
    # residual variance zero. A p-value of 1e-216 was reported from a standard error of 1e-16.
    adata, design = _cohort(shift, scale)
    res = between_arm_comparison(adata, visit="V2", features=FEATURES, design=design,
                                 standardize=standardize).set_index("feature")

    assert np.isfinite(res.loc["by_arm", "beta_arm"]) and res.loc["by_arm", "beta_arm"] != 0
    for column in ("se_arm", "p_arm", "ci_lo_arm", "ci_hi_arm"):
        assert np.isnan(res.loc["by_arm", column]), column
    assert np.isfinite(res.loc[["by_participant", "nudged", "noisy"], "p_arm"]).all()


def test_estimates_of_an_exactly_fitted_outcome_are_kept():
    adata, design = _cohort()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        did = did_table(adata, FEATURES, design, VISITS, standardize=False).set_index("feature")
        within = within_arm_comparison(adata, arm="Treated", features=FEATURES, design=design, visits=VISITS,
                                       standardize=False).set_index("feature")

    assert did.loc["by_participant", "beta_DiD"] == pytest.approx(0.0, abs=1e-12)
    assert within.loc["by_participant", "beta_time"] == pytest.approx(0.0, abs=1e-12)
    assert did.loc["by_participant", "resid_sd"] == pytest.approx(0.0, abs=1e-12)


def test_abundance_gives_no_test_when_no_proportion_changes():
    # the same cells at both visits: every change in proportion is exactly zero
    rows = []
    for arm in ("Treated", "Control"):
        for p in range(6):
            n_a = 10 + 3 * p
            for visit in VISITS:
                rows.extend((f"{arm[0]}{p}", arm, visit, "A") for _ in range(n_a))
                rows.extend((f"{arm[0]}{p}", arm, visit, "B") for _ in range(40 - n_a))
    obs = pd.DataFrame(rows, columns=["participant_id", "arm", "visit", "celltype"])
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="Treated", arm_control="Control", celltype_col="celltype")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = abundance_did(adata, design, VISITS, min_units=4)

    assert len(res) == 2
    assert res["p_DiD"].isna().all() and res["se_DiD"].isna().all()


def _paired_frame(changes, baselines):
    rows = []
    for i, (d, b) in enumerate(zip(changes, baselines)):
        arm = int(i < len(changes) // 2)
        rows += [(f"p{i}", 0, arm, b), (f"p{i}", 1, arm, b + d)]
    return pd.DataFrame(rows, columns=["pid", "visit_num", "arm_bin", "y"])


# every combination in which doubles at the origin hold the nudge of 2**-17 times the unit
_ORIGINS_AND_UNITS = [
    pytest.param(scale, shift, swap, id=f"{scale:g}-{shift:g}-{'swapped' if swap else 'as_coded'}")
    for scale in (2.0**-20, 1.0, 2.0**20)
    for shift in (0.0, 1.0, 1024.0, 2.0**20)
    for swap in (False, True)
    if 16 * np.spacing(shift) <= scale * 2.0**-17
]


@pytest.mark.parametrize("scale,shift,swap", _ORIGINS_AND_UNITS)
def test_the_criterion_follows_the_outcome_not_its_origin_or_its_unit(scale, shift, swap):
    # Baselines and changes are multiples of 1/8 and the shifts and scales powers of two, so
    # every outcome below is exact in double precision: the same change for each participant
    # of an arm stays the same change after shifting and scaling.
    baselines = [0.0, 1.75, -0.5, 2.25, 0.875, 3.125, -1.25, 0.5]
    exact = _paired_frame([0.5] * 4 + [-0.25] * 4, baselines)
    nudged = _paired_frame([0.5, 0.5, 0.5, 0.5 + 2.0**-17] + [-0.25] * 4, baselines)
    for frame, expected in ((exact, True), (nudged, False)):
        frame = frame.assign(y=shift + scale * frame["y"])
        if swap:
            frame = frame.assign(arm_bin=1 - frame["arm_bin"])
        out = did_fit(frame, "y", "pid", "visit_num", "arm_bin", standardize=False, cov_type="nonrobust")
        assert np.isnan(out["p_DiD"]) == expected, (shift, scale, swap, expected)
        assert np.isnan(out["se_DiD"]) == expected
        assert out["beta_DiD"] == pytest.approx((-0.75 if swap else 0.75) * scale, rel=1e-4)


@pytest.mark.parametrize("unit", [1.0, 1e5, 1e-5])
def test_the_criterion_holds_beside_a_covariate_on_another_scale(unit):
    # A regressor in a very different unit makes the design badly scaled, and the residuals
    # then carry an error far above their own rounding, along the regressors.
    rng = np.random.default_rng(6)
    baselines = [0.0, 1.75, -0.5, 2.25, 0.875, 3.125, -1.25, 0.5, 1.0, -2.0]
    exact = _paired_frame([0.5] * 5 + [-0.25] * 5, baselines)
    nudged = _paired_frame([0.5] * 4 + [0.5 + 2.0**-17] + [-0.25] * 5, baselines)
    load = unit * rng.normal(size=len(exact))
    kw = dict(y="y", unit="pid", time="visit_num", arm_bin="arm_bin", covariates=["load"],
              standardize=False, cov_type="nonrobust")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = did_fit(exact.assign(load=load), **kw)
        missed = did_fit(nudged.assign(load=load), **kw)

    assert np.isnan(fitted["p_DiD"]) and np.isnan(fitted["se_DiD"])
    assert fitted["beta_DiD"] == pytest.approx(0.75, rel=1e-6)
    assert np.isfinite(missed["p_DiD"])


def test_a_model_without_residual_degrees_of_freedom_is_not_called_an_exact_fit():
    # it fits any outcome; that case has its own report (a standard error that is not finite)
    rng = np.random.default_rng(0)
    design = sm.add_constant(rng.normal(size=(5, 4)))
    saturated = sm.OLS(rng.normal(size=5), design).fit()
    ordinary = sm.OLS(rng.normal(size=30), sm.add_constant(rng.normal(size=(30, 2)))).fit()
    constant = sm.OLS(np.full(30, 0.1), sm.add_constant(rng.normal(size=(30, 2)))).fit()
    weighted = sm.WLS(design[:, 1] * 2.0 + 1.0, design[:, :2], weights=rng.uniform(1, 3, 5)).fit()

    # 0.1 + 0.2 and 0.3 differ in the last bit: an outcome that varies by rounding only
    rounding_only = np.where(np.arange(30) % 2, 0.1 + 0.2, 0.3)
    assert len(set(rounding_only)) == 2
    nearly_constant = sm.OLS(rounding_only, sm.add_constant(rng.normal(size=(30, 2)))).fit()

    assert not fits_exactly(saturated)
    assert not fits_exactly(ordinary)
    assert fits_exactly(constant)
    assert fits_exactly(nearly_constant)
    assert fits_exactly(weighted)


def test_module_scores_give_no_test_for_scores_without_spread():
    rng = np.random.default_rng(3)
    design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="Treated", arm_control="Control", celltype_col=None)
    rows = []
    for arm, levels in (("Treated", (1.0, 2.5)), ("Control", (0.5, 0.5))):
        for p in range(5):
            for visit, level in zip(VISITS, levels):
                rows.append((f"{arm[0]}{p}", arm, visit, "all", "flat", level))
                rows.append((f"{arm[0]}{p}", arm, visit, "all", "noisy", level + rng.normal()))
    pb = pd.DataFrame(rows, columns=["participant_id", "arm", "visit", "pool", "module", "module_score"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = module_score_did_by_pool(pb, design, VISITS, allow_unpaired=True).set_index("module")

    assert res.loc["flat", "beta_DiD"] == pytest.approx(1.5)
    for column in ("p_DiD", "p_treated", "p_control"):
        assert np.isnan(res.loc["flat", column]), column
        assert np.isfinite(res.loc["noisy", column]), column


def test_module_scores_give_no_within_arm_test_without_residual_degrees_of_freedom():
    # One treated participant-visit at each visit: the within-arm model has two rows and two
    # parameters. Its robust standard error is infinite, and the p-value read 1.0.
    rng = np.random.default_rng(4)
    design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="Treated", arm_control="Control", celltype_col=None)
    rows = [("T0", "Treated", VISITS[0], "all", "m", 1.0), ("T1", "Treated", VISITS[1], "all", "m", 2.0)]
    for p in range(5):
        for visit in VISITS:
            rows.append((f"C{p}", "Control", visit, "all", "m", rng.normal()))
    pb = pd.DataFrame(rows, columns=["participant_id", "arm", "visit", "pool", "module", "module_score"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = module_score_did_by_pool(pb, design, VISITS, allow_unpaired=True).iloc[0]

    assert res["mean_delta_treated"] == pytest.approx(1.0)
    assert np.isnan(res["p_treated"])
    assert np.isfinite(res["p_control"]) and np.isfinite(res["beta_DiD"])


def test_module_scores_give_no_did_test_for_one_row_per_arm_and_visit():
    # Four rows and four parameters. The robust standard error of the interaction is infinite
    # and its p-value read 1.0, which then entered the FDR family of the other modules.
    rng = np.random.default_rng(8)
    design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="Treated", arm_control="Control", celltype_col=None)
    rows = [("T0", "Treated", VISITS[0], "all", "thin", 1.0), ("T1", "Treated", VISITS[1], "all", "thin", 2.5),
            ("C0", "Control", VISITS[0], "all", "thin", 0.5), ("C1", "Control", VISITS[1], "all", "thin", 0.7)]
    for arm in ("Treated", "Control"):
        for p in range(5):
            for visit in VISITS:
                rows.append((f"{arm[0]}{p + 2}", arm, visit, "all", "full", rng.normal()))
    pb = pd.DataFrame(rows, columns=["participant_id", "arm", "visit", "pool", "module", "module_score"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = module_score_did_by_pool(pb, design, VISITS, allow_unpaired=True).set_index("module")

    assert res.loc["thin", "beta_DiD"] == pytest.approx(1.3)
    assert res.loc["thin", ["p_DiD", "p_treated", "p_control"]].isna().all()
    assert np.isfinite(res.loc["full", ["p_DiD", "p_treated", "p_control"]].astype(float)).all()
    fdr_columns = [c for c in res.columns if c.startswith("FDR")]
    assert fdr_columns
    for column in fdr_columns:
        assert np.isnan(res.loc["thin", column]) and res.loc["full", column] == pytest.approx(res.loc["full", "p_DiD"])
