"""The wild cluster bootstrap refits the model under the null and inverts the test for intervals.

The reference below refits every bootstrap regression with statsmodels, so it shares no linear
algebra with the closed-form implementation; it uses the same cluster sign assignments.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from anndata import AnnData
from statsmodels.stats.sandwich_covariance import cov_cluster

from sctrial.design import TrialDesign
from sctrial.stats.did import did_table
from sctrial.utils import warn_unbounded_bootstrap_intervals, wild_cluster_bootstrap_t

B = 199
SEED = 11


def _panel(n_treated, n_control, *, effect=0.8, seed=0, weighted=False, single_arm=False):
    """Two visits per participant; returns the unfitted model, clusters, tested term, frame."""
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    pid = np.repeat(np.arange(n), 2)
    time = np.tile([0.0, 1.0], n)
    arm = np.repeat((np.arange(n) < n_treated).astype(float), 2)
    level = np.repeat(rng.normal(0, 1, n), 2)
    y = level + 0.3 * time + effect * time * arm + rng.normal(0, 0.6, 2 * n)
    df = pd.DataFrame({"y": y, "pid": pid, "time": time, "arm": arm})
    formula = "y ~ time + C(pid)" if single_arm else "y ~ time + time:arm + C(pid)"
    w = rng.uniform(0.5, 3.0, 2 * n) if weighted else None
    model = smf.wls(formula, data=df, weights=w) if weighted else smf.ols(formula, data=df)
    return model, pid, ("time" if single_arm else "time:arm"), df


def _fit(model, pid, cov_type):
    return model.fit(cov_type=cov_type, cov_kwds={"groups": pid} if cov_type == "cluster" else None)


def _signs(pid, n_draws=B):
    """Every sign assignment when there are at most ``n_draws`` of them, else ``n_draws`` draws."""
    n_clusters = len(np.unique(pid))
    if 2**n_clusters <= n_draws:
        flips = (np.arange(2**n_clusters)[:, None] >> np.arange(n_clusters)) & 1
        return flips.astype(float) * 2 - 1, True
    flips = np.random.default_rng(SEED).integers(0, 2, size=(n_draws, n_clusters))
    return flips.astype(float) * 2 - 1, False


def _reference_p(model, pid, term, cov_type, beta_null=0.0, n_draws=B):
    """Restricted wild cluster bootstrap p-value by brute-force statsmodels refits."""
    endog, exog = np.asarray(model.endog, float), np.asarray(model.exog, float)
    weights = np.asarray(model.weights, float)
    j = model.exog_names.index(term)
    _, idx = np.unique(pid, return_inverse=True)
    kw = {"cov_type": cov_type, "cov_kwds": {"groups": idx} if cov_type == "cluster" else None}
    full = sm.WLS(endog, exog, weights=weights).fit(**kw)
    restricted = sm.WLS(endog - beta_null * exog[:, j], np.delete(exog, j, axis=1), weights=weights).fit()
    fitted_r = restricted.fittedvalues + beta_null * exog[:, j]
    t_obs = (full.params[j] - beta_null) / full.bse[j]
    signs, exact = _signs(pid, n_draws)
    t_star = []
    for w in signs:
        fb = sm.WLS(fitted_r + restricted.resid * w[idx], exog, weights=weights).fit(**kw)
        t_star.append((fb.params[j] - beta_null) / fb.bse[j])
    n_extreme = np.sum(np.abs(np.asarray(t_star)) >= abs(t_obs) * (1 - 1e-10))
    return float(n_extreme / len(signs)) if exact else float((n_extreme + 1) / (len(signs) + 1))


DESIGNS = {
    "balanced_6v6": dict(n_treated=6, n_control=6),
    "unbalanced_3v8": dict(n_treated=3, n_control=8),
    "weighted_6v6": dict(n_treated=6, n_control=6, weighted=True),
    "single_arm_9": dict(n_treated=0, n_control=9, single_arm=True),
    "enumerated_3v3": dict(n_treated=3, n_control=3, effect=2.0),
    "enumerated_single_arm_6": dict(n_treated=0, n_control=6, single_arm=True, effect=0.0),
}


@pytest.mark.parametrize("design", list(DESIGNS))
@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
def test_p_value_and_interval_match_restricted_refit(design, cov_type):
    model, pid, term, _ = _panel(**DESIGNS[design])
    fit = _fit(model, pid, cov_type)
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=B, seed=SEED, cov_type=cov_type)

    assert res.p_boot == pytest.approx(_reference_p(model, pid, term, cov_type), abs=1e-12)

    eps = 1e-6 * float(fit.bse[term])
    for bound, outward in ((res.ci_lo, -1.0), (res.ci_hi, 1.0)):
        assert np.isfinite(bound)
        assert _reference_p(model, pid, term, cov_type, beta_null=bound - outward * eps) >= 0.05
        assert _reference_p(model, pid, term, cov_type, beta_null=bound + outward * eps) < 0.05


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize("n_treated,n_control", [(6, 6), (3, 8), (2, 4)])
def test_result_does_not_depend_on_which_arm_is_coded_one(n_treated, n_control, cov_type):
    model, pid, term, df = _panel(n_treated, n_control, effect=1.2, seed=3)
    fit = _fit(model, pid, cov_type)
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=B, seed=SEED, cov_type=cov_type)

    flipped = smf.ols("y ~ time + time:arm + C(pid)", data=df.assign(arm=1.0 - df["arm"]))
    fit_f = _fit(flipped, pid, cov_type)
    res_f = wild_cluster_bootstrap_t(
        fit_f, fit_f.model.exog, pid, term, B=B, seed=SEED, cov_type=cov_type
    )

    assert res_f.p_boot == res.p_boot
    assert (res_f.ci_lo, res_f.ci_hi) == pytest.approx((-res.ci_hi, -res.ci_lo), abs=1e-7)
    assert res_f.se_boot == pytest.approx(res.se_boot, rel=1e-10)


@pytest.mark.parametrize("seed", range(12))
def test_interval_excludes_zero_exactly_when_p_is_below_alpha(seed):
    model, pid, term, _ = _panel(6, 6, effect=0.15 * seed, seed=seed)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=499, seed=SEED, cov_type="nonrobust")
    assert (res.ci_lo > 0 or res.ci_hi < 0) == (res.p_boot < 0.05)
    assert res.ci_lo <= fit.params[term] <= res.ci_hi


@pytest.mark.parametrize("weighted", [False, True])
def test_standard_error_is_that_of_the_unrestricted_bootstrap(weighted):
    """Exactly the cluster-robust SE without small-sample correction, whatever the effect size."""
    model, pid, term, _ = _panel(6, 6, effect=1.5, seed=5, weighted=weighted)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=B, seed=SEED, cov_type="nonrobust")

    j = model.exog_names.index(term)
    _, idx = np.unique(pid, return_inverse=True)
    assert res.se_boot == pytest.approx(np.sqrt(cov_cluster(fit, idx, use_correction=False)[j, j]), rel=1e-9)

    exog, weights = np.asarray(model.exog, float), np.asarray(model.weights, float)
    signs, _ = _signs(pid)
    draws = [
        sm.WLS(fit.fittedvalues + fit.resid * w[idx], exog, weights=weights).fit().params[j]
        for w in signs
    ]
    np.testing.assert_allclose(res.boot_distribution, draws, rtol=1e-9, atol=1e-12)
    assert np.std(draws) == pytest.approx(res.se_boot, rel=0.15)


def test_observed_statistic_uses_the_requested_variance_estimator():
    """The fit's own covariance type must not leak into the bootstrap statistic."""
    model, pid, term, _ = _panel(6, 6, effect=1.0, seed=8)
    from_nonrobust = wild_cluster_bootstrap_t(
        _fit(model, pid, "nonrobust"), model.exog, pid, term, B=B, seed=SEED, cov_type="cluster"
    )
    from_cluster = wild_cluster_bootstrap_t(
        _fit(model, pid, "cluster"), model.exog, pid, term, B=B, seed=SEED, cov_type="cluster"
    )
    assert from_nonrobust[:4] == from_cluster[:4]


@pytest.mark.parametrize("seed", [0, 1, 18, 99])
def test_few_clusters_are_enumerated_so_the_result_does_not_depend_on_the_seed(seed):
    model, pid, term, _ = _panel(3, 3, effect=2.0, seed=4)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, seed=seed, cov_type="nonrobust")
    ref = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, seed=12345, cov_type="nonrobust")
    assert res[:4] == ref[:4]
    assert len(res.boot_distribution) == 2**6
    assert res.p_boot * 2**6 == pytest.approx(round(res.p_boot * 2**6), abs=1e-9)
    assert res.p_boot >= 2 / 2**6


@pytest.mark.parametrize("seed", [0, 1, 18, 99])
def test_five_clusters_cannot_reject_and_give_an_unbounded_interval(seed):
    model, pid, term, _ = _panel(3, 2, effect=8.0, seed=2)
    fit = _fit(model, pid, "nonrobust")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, seed=seed, cov_type="nonrobust")
    assert res.p_boot == pytest.approx(2 / 2**5)
    assert res.ci_lo == -np.inf and res.ci_hi == np.inf
    assert np.isfinite(res.se_boot)


def test_unbounded_intervals_raise_one_warning_per_table():
    """A table over many features with five participants warns once, not once per feature."""
    rng = np.random.default_rng(0)
    rows = [
        {"participant_id": f"P{i}", "visit": v, "arm": "T" if i < 3 else "C"}
        for i in range(5) for v in ("Pre", "Post") for _ in range(4)
    ]
    obs = pd.DataFrame(rows)
    adata = AnnData(X=rng.normal(size=(len(obs), 6)), obs=obs)
    adata.var_names = [f"g{k}" for k in range(6)]
    design = TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                         arm_treated="T", arm_control="C", celltype_col=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = did_table(adata, features=list(adata.var_names), design=design, visits=("Pre", "Post"),
                        aggregate="participant_visit", use_bootstrap=True, n_boot=999, seed=1)
    unbounded = [w for w in caught if "unbounded" in str(w.message)]
    assert len(unbounded) == 1 and "6 of 6" in str(unbounded[0].message)
    assert np.isinf(res["ci_lo_boot"]).all() and np.isinf(res["ci_hi_boot"]).all()


def test_no_warning_when_every_interval_is_bounded():
    res = pd.DataFrame({"ci_lo_boot": [-1.0, np.nan], "ci_hi_boot": [1.0, np.nan]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_unbounded_bootstrap_intervals(res)
        warn_unbounded_bootstrap_intervals(pd.DataFrame({"p": [0.1]}))


def test_unknown_covariance_type_is_rejected():
    model, pid, term, _ = _panel(6, 6)
    fit = _fit(model, pid, "nonrobust")
    with pytest.raises(ValueError, match="cov_type"):
        wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=B, seed=SEED, cov_type="HC3")
