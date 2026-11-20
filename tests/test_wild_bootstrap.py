"""The wild cluster bootstrap refits the model under the null and inverts the test for intervals.

The reference below refits every bootstrap regression with statsmodels, so it shares no linear
algebra with the closed-form implementation; it uses the same cluster sign assignments.
"""

import itertools
import warnings
from fractions import Fraction

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


def _from_outcomes(y, n_treated, *, single_arm=False):
    """Unfitted model, clusters and tested term for given outcomes, two visits per participant."""
    n = len(y) // 2
    pid = np.repeat(np.arange(n), 2)
    df = pd.DataFrame({
        "y": np.asarray(y, dtype=float),
        "pid": pid,
        "time": np.tile([0.0, 1.0], n),
        "arm": np.repeat((np.arange(n) < n_treated).astype(float), 2),
    })
    formula = "y ~ time + C(pid)" if single_arm else "y ~ time + time:arm + C(pid)"
    return smf.ols(formula, data=df), pid, ("time" if single_arm else "time:arm")


def _p_at(model, pid, term, beta_null, cov_type="nonrobust"):
    """Bootstrap p-value of ``beta = beta_null``: the test of zero after shifting the outcome."""
    shifted = np.asarray(model.endog) - beta_null * np.asarray(model.exog)[:, model.exog_names.index(term)]
    fit = _fit(sm.OLS(shifted, pd.DataFrame(model.exog, columns=model.exog_names)), pid, cov_type)
    return wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type=cov_type).p_boot


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize("baselines", [[0, 0, 0, 0, 0, 0], [100, 7, 23, 55, 3, 80]])
def test_assignments_that_reproduce_the_observed_statistic_are_counted(baselines, cov_type):
    # Flipping the participant whose outcome did not change leaves the data unchanged, so four
    # of the 64 assignments give the observed statistic. Counted exactly, the test cannot reject.
    changes = [0, 10, 11, 11, 4, 2]
    y = np.ravel([[b, b + d] for b, d in zip(baselines, changes)])
    model, pid, term = _from_outcomes(y, 0, single_arm=True)
    fit = _fit(model, pid, cov_type)
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type=cov_type)

    assert res.p_boot == 4 / 64
    assert res.ci_lo <= 0.0 <= res.ci_hi


def test_interval_spans_values_the_test_accepts_beyond_a_rejected_stretch():
    # Two treated and five control participants. The test rejects the values between about
    # -2.5 and -1 and accepts those on both sides, zero included.
    model, pid, term = _from_outcomes([6, -2, 7, -2, 1, 7, 7, 3, -2, 5, -2, 2, -2, 4], n_treated=2)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type="nonrobust")

    assert res.p_boot == pytest.approx(_reference_p(model, pid, term, "nonrobust", n_draws=999))
    assert res.p_boot == 0.0625
    assert _reference_p(model, pid, term, "nonrobust", beta_null=-2.0, n_draws=999) < 0.05
    assert res.ci_lo < -2.0 and res.ci_lo <= 0.0 <= res.ci_hi

    grid = np.linspace(res.ci_lo - 6.0, res.ci_hi + 6.0, 241)
    accepted = grid[[_p_at(model, pid, term, b) >= 0.05 for b in grid]]
    assert res.ci_lo <= accepted.min() and accepted.max() <= res.ci_hi
    eps = 1e-6 * float(fit.bse[term])
    for bound, outward in ((res.ci_lo, -1.0), (res.ci_hi, 1.0)):
        assert _reference_p(model, pid, term, "nonrobust", beta_null=bound - outward * eps, n_draws=999) >= 0.05
        assert _reference_p(model, pid, term, "nonrobust", beta_null=bound + outward * eps, n_draws=999) < 0.05


def test_interval_includes_a_value_accepted_only_through_ties():
    # At exactly -13 two more assignments tie with the observed statistic and the p-value is
    # 0.0625; just below and just above it is 0.047.
    model, pid, term = _from_outcomes([4, 2, 1, 3, -2, 6, 5, 1, 5, 0, 4, -2, 4, -1], n_treated=2)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type="nonrobust")

    assert _reference_p(model, pid, term, "nonrobust", beta_null=-13.0, n_draws=999) == 0.0625
    assert _reference_p(model, pid, term, "nonrobust", beta_null=-12.9, n_draws=999) < 0.05
    assert _reference_p(model, pid, term, "nonrobust", beta_null=-13.1, n_draws=999) < 0.05
    assert res.ci_lo == pytest.approx(-13.0, abs=1e-6) and res.ci_lo <= -13.0
    assert res.ci_hi == pytest.approx(13.25, abs=1e-6)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
def test_interval_holds_every_accepted_value_in_small_unbalanced_designs(seed, cov_type):
    rng = np.random.default_rng(seed)
    n_treated, n_control = int(rng.integers(2, 4)), int(rng.integers(4, 7))
    y = rng.choice(np.arange(-2, 7), size=2 * (n_treated + n_control)).astype(float)
    model, pid, term = _from_outcomes(y, n_treated)
    fit = _fit(model, pid, cov_type)
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type=cov_type)

    scale = float(fit.bse[term])
    grid = np.append(fit.params[term] + scale * np.linspace(-25.0, 25.0, 101), 0.0)
    for b in grid:
        if _p_at(model, pid, term, b, cov_type) >= 0.05:
            assert res.ci_lo - 1e-9 * scale <= b <= res.ci_hi + 1e-9 * scale


def test_noise_far_from_the_estimate_does_not_set_the_bound():
    # Flipping one whole arm leaves no residual variation in the tested regressor, and rounding
    # then puts a crossing hundreds of millions of standard errors away.
    rng = np.random.default_rng(3)
    model, pid, term = _from_outcomes(rng.choice(np.arange(-2, 7), size=12).astype(float), n_treated=4)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type="nonrobust")

    assert abs(res.ci_lo - fit.params[term]) < 100 * fit.bse[term]
    assert abs(res.ci_hi - fit.params[term]) < 100 * fit.bse[term]
    eps = 1e-6 * float(fit.bse[term])
    assert _p_at(model, pid, term, res.ci_lo - eps) < 0.05 <= _p_at(model, pid, term, res.ci_lo + eps)
    assert _p_at(model, pid, term, res.ci_hi + eps) < 0.05 <= _p_at(model, pid, term, res.ci_hi - eps)


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize("scale", [1.0, 0.1, 7.3])
def test_draws_without_residual_variance_count_whatever_the_coding_or_the_units(cov_type, scale):
    # Changes [4, 4] against [2, -2, -2, -2, -2, -2]: four of the 256 assignments make every
    # change equal within each arm, so the model fits them exactly and the statistic is
    # unbounded. They are as extreme as any observed statistic: 16 of 256 with the 12 others.
    changes = scale * np.array([4, 4, 2, -2, -2, -2, -2, -2], dtype=float)
    y = np.ravel([[0.0, d] for d in changes])
    model, pid, term = _from_outcomes(y, n_treated=2)
    fit = _fit(model, pid, cov_type)
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type=cov_type)

    reversed_coding = smf.ols(
        "y ~ time + time:arm + C(pid)",
        data=pd.DataFrame(
            {"y": y, "pid": pid, "time": np.tile([0.0, 1.0], 8), "arm": np.repeat(np.arange(8) >= 2, 2).astype(float)}
        ),
    )
    fit_r = _fit(reversed_coding, pid, cov_type)
    res_r = wild_cluster_bootstrap_t(fit_r, fit_r.model.exog, pid, term, B=999, cov_type=cov_type)

    assert res.p_boot == res_r.p_boot == 16 / 256
    assert res.ci_lo <= 0.0 <= res.ci_hi
    assert (res_r.ci_lo, res_r.ci_hi) == pytest.approx((-res.ci_hi, -res.ci_lo), abs=1e-6 * scale)


def _exact_count(y, n_treated, cov_type, margin):
    """Sign assignments at least as extreme as the data, in exact arithmetic on the doubles.

    The paired model with participant effects is the two-sample comparison of the changes.
    Under the null every change is the common mean plus a residual whose sign is flipped by
    participant. A draw without variance counts; a statistic within ``margin`` (relative) of
    the observed one is a tie. Returns the count and the number of draws without variance.
    """
    y = [Fraction(float(v)) for v in y]
    d = [b - a for a, b in zip(y[0::2], y[1::2])]
    n, n_t = len(d), n_treated
    n_c = n - n_t
    mean = sum(d) / n
    resid = [x - mean for x in d]

    def numerator_and_variance(values):
        m_t, m_c = sum(values[:n_t]) / n_t, sum(values[n_t:]) / n_c
        ss_t = sum((v - m_t) ** 2 for v in values[:n_t])
        ss_c = sum((v - m_c) ** 2 for v in values[n_t:])
        if cov_type == "nonrobust":
            var = (ss_t + ss_c) / (n - 2) * (Fraction(1, n_t) + Fraction(1, n_c))
        else:
            # statsmodels: G/(G-1) * (N-1)/(N-K) * sum of squared cluster scores, N = 2G, K = G + 2
            var = Fraction(n, n - 1) * Fraction(2 * n - 1, n - 2) * (ss_t / n_t**2 + ss_c / n_c**2)
        return m_t - m_c, var

    num, var = numerator_and_variance(d)
    t_sq = num**2 / var
    assert t_sq > 1
    bar = t_sq * (1 - margin) ** 2
    count = n_zero = 0
    for signs in itertools.product((1, -1), repeat=n):
        num_b, var_b = numerator_and_variance([mean + sg * r for sg, r in zip(signs, resid)])
        if var_b == 0:
            n_zero += 1
            count += 1
        else:
            count += num_b**2 / var_b >= bar
    return count, n_zero


def _paired_result(y, n_treated, cov_type, treated_is_one=True, arm_value=1.0, ci_level=0.95):
    """Bootstrap result for outcomes given as (baseline, follow-up) pairs, treated first."""
    n = len(y) // 2
    pid = np.repeat(np.arange(n), 2)
    treated = np.arange(n) < n_treated
    df = pd.DataFrame({
        "y": np.asarray(y, dtype=float),
        "pid": pid,
        "time": np.tile([0.0, 1.0], n),
        "arm": arm_value * np.repeat(treated if treated_is_one else ~treated, 2).astype(float),
    })
    fit = _fit(smf.ols("y ~ time + time:arm + C(pid)", data=df), pid, cov_type)
    return wild_cluster_bootstrap_t(
        fit, fit.model.exog, pid, "time:arm", B=999, cov_type=cov_type, ci_level=ci_level
    )


def _package_count(y, n_treated, cov_type, treated_is_one=True, arm_value=1.0):
    res = _paired_result(y, n_treated, cov_type, treated_is_one, arm_value)
    return res.p_boot * 2 ** (len(y) // 2)


# changes of the treated participants, then of the controls; n_treated; the value that is nudged
_NEAR_FITS = {
    "three_five": ([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0], 3),
    "two_six": ([4.0, 4.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0], 2),
}


# every combination in which doubles at the baseline hold the nudge
_NUDGED = [
    pytest.param(case, scale, baseline, nudge, cov_type, id=f"{case}-{scale:g}-{baseline:g}-{nudge:g}-{cov_type}")
    for case in _NEAR_FITS
    for scale in (1.0, 1e-3, 1e3)
    for baseline in (0.0, 100.0, 1e7, 2.0**32)
    for nudge in (1e-3, 1e-6, 1e-11)
    for cov_type in ("nonrobust", "cluster")
    if np.spacing(scale * baseline) <= 1e-3 * scale * nudge
]


@pytest.mark.parametrize("case,scale,baseline,nudge,cov_type", _NUDGED)
def test_a_draw_with_a_small_variance_is_not_a_draw_without_variance(case, scale, baseline, nudge, cov_type):
    # Changes [-1, -1, -1] against [1, 1, 1, 1, -1] leave four assignments that the model fits
    # exactly. Nudging one change gives each of them a small positive variance, and two of them
    # a statistic of 0.53 against an observed 3.0. A variance below 1e-12 of the largest was
    # taken for zero and the draw counted: 14 of 256 where 12 are as extreme.
    changes, n_treated = _NEAR_FITS[case]
    changes = scale * (np.array(changes) + np.r_[nudge, np.zeros(len(changes) - 1)])
    y = np.ravel([[scale * baseline, scale * baseline + d] for d in changes])
    exact, n_zero = _exact_count(y, n_treated, cov_type, margin=Fraction(1, 10**9))
    assert n_zero == 0
    # not a case that the width of the tie margin decides
    assert exact == _exact_count(y, n_treated, cov_type, Fraction(1, 10**8))[0]
    assert exact == _exact_count(y, n_treated, cov_type, Fraction(1, 10**10))[0]

    assert _package_count(y, n_treated, cov_type) == exact
    assert _package_count(y, n_treated, cov_type, treated_is_one=False) == exact


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize("baseline", [0.0, 100.0, 12345.0, 1e7, 2.0**32, 1e12])
@pytest.mark.parametrize("case", list(_NEAR_FITS))
def test_draws_that_the_model_fits_exactly_still_count(case, baseline, cov_type):
    # Far from zero the statistics of the assignments that tie with the observed one differed
    # from it by more than the tie margin: 6 of 256 where 14 are as extreme.
    changes, n_treated = _NEAR_FITS[case]
    y = np.ravel([[baseline, baseline + d] for d in changes])
    exact, n_zero = _exact_count(y, n_treated, cov_type, margin=Fraction(1, 10**9))
    assert n_zero == 4

    assert _package_count(y, n_treated, cov_type) == exact
    assert _package_count(y, n_treated, cov_type, treated_is_one=False) == exact


@pytest.mark.parametrize("arm_value", [1.0, 1e-4, 1e4])
@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize(
    "baselines",
    [
        [0.1] * 8,
        [1.0] * 8,
        [0.37 * i for i in range(8)],
        [1e6 + 0.1 * i for i in range(8)],
        [0.3, 0.1 + 0.2] * 4,
    ],
    ids=["tenth", "one", "by_participant", "far_by_participant", "equal_within_rounding"],
)
def test_an_outcome_that_the_model_fits_exactly_has_no_test(baselines, cov_type, arm_value):
    # No participant changes, so every residual is zero in exact arithmetic. Rounding left a
    # positive variance and a p-value made of rounding error. 0.1 + 0.2 and 0.3 differ in the
    # last bit: an outcome that varies by rounding only does not vary.
    y = np.ravel([[b, b] for b in baselines])
    res = _paired_result(y, 3, cov_type, arm_value=arm_value)

    assert np.isnan(res.p_boot) and np.isnan(res.ci_lo) and np.isnan(res.ci_hi)


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize("arm_value", [1e-4, 1e4])
def test_near_fits_are_told_from_fits_whatever_the_unit_of_the_regressor(arm_value, cov_type):
    # The statistic does not depend on the unit of the tested regressor; the cluster scores do.
    changes, n_treated = _NEAR_FITS["three_five"]
    y = np.ravel([[0.0, d] for d in np.array(changes) + np.r_[1e-6, np.zeros(7)]])
    exact = _exact_count(y, n_treated, cov_type, margin=Fraction(1, 10**9))[0]

    assert exact == 12
    assert _package_count(y, n_treated, cov_type, arm_value=arm_value) == exact
    fitted = np.ravel([[0.0, d] for d in changes])
    assert _package_count(fitted, n_treated, cov_type, arm_value=arm_value) == 14


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
def test_interval_of_a_near_fit_moves_with_the_effect(cov_type):
    # Adding 0.7 to every treated follow-up moves the estimate and the interval by 0.7: the
    # test of 0.7 on the moved outcome is the test of zero on the original one, where the
    # nearly fitted draws decide the 5% level. Their variance is computed at the tested value.
    changes, n_treated = _NEAR_FITS["three_five"]
    changes = np.array(changes) + np.r_[1e-6, np.zeros(7)]
    moved = changes + 0.7 * (np.arange(8) < n_treated)
    base = _paired_result(np.ravel([[0.0, d] for d in changes]), n_treated, cov_type)
    res = _paired_result(np.ravel([[0.0, d] for d in moved]), n_treated, cov_type)

    assert base.p_boot == 12 / 256 and not base.ci_lo <= 0.0 <= base.ci_hi
    assert (res.ci_lo, res.ci_hi) == pytest.approx((base.ci_lo + 0.7, base.ci_hi + 0.7), abs=1e-7)
    assert not res.ci_lo <= 0.7 <= res.ci_hi


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
def test_a_near_fit_is_judged_at_the_tested_value(cov_type):
    # The same outcome moved by 70 in the treated arm: its nearly fitted draws now occur at the
    # tested value 70, where 12 of 256 assignments are as extreme. At a level of 11.5/256 the
    # values just beyond 70 are accepted, as those just beyond zero are for the original
    # outcome. With the residuals of another tested value the two draws with an unbounded
    # statistic get a large variance and are missed, and the interval ends below 70.
    changes, n_treated = _NEAR_FITS["three_five"]
    changes = np.array(changes) + np.r_[1e-6, np.zeros(7)]
    moved = changes + 70.0 * (np.arange(8) < n_treated)
    level = 1.0 - 11.5 / 256
    base = _paired_result(np.ravel([[0.0, d] for d in changes]), n_treated, cov_type, ci_level=level)
    res = _paired_result(np.ravel([[0.0, d] for d in moved]), n_treated, cov_type, ci_level=level)

    assert base.p_boot == 12 / 256
    assert 1e-7 < base.ci_hi < 1e-6
    assert (res.ci_lo - 70.0, res.ci_hi - 70.0) == pytest.approx((base.ci_lo, base.ci_hi), abs=5e-8)


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
def test_an_outcome_that_varies_by_rounding_only_has_no_test(cov_type):
    # 0.3 at baseline and 0.1 + 0.2 at follow-up in half of the participants: changes of one
    # unit in the last place, which no regressor explains. They are not a signal.
    pairs = [[0.3, 0.1 + 0.2] if i % 2 else [0.3, 0.3] for i in range(8)]
    assert pairs[1][0] != pairs[1][1]
    res = _paired_result(np.ravel(pairs), 3, cov_type)

    assert np.isnan(res.p_boot) and np.isnan(res.ci_lo) and np.isnan(res.ci_hi)


@pytest.mark.parametrize("cov_type", ["nonrobust", "cluster"])
@pytest.mark.parametrize("term", ["x1", "const"])
def test_outcome_keeps_its_origin_when_the_other_regressors_hold_no_constant(term, cov_type):
    # Without an intercept among the other regressors the origin of the outcome matters to the
    # tested coefficient, so the outcome is not measured from its mean.
    rng = np.random.default_rng(8)
    pid = np.repeat(np.arange(7), 3)
    x1, x2 = rng.normal(size=21), rng.normal(size=21)
    y = 5.0 + 0.4 * x1 + rng.normal(size=21) + rng.normal(size=7)[pid]
    design = pd.DataFrame({"const": 1.0, "x1": x1} if term == "const" else {"x1": x1, "x2": x2})
    model = sm.OLS(y, design)
    fit = _fit(model, pid, cov_type)
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=B, seed=SEED, cov_type=cov_type)

    assert res.p_boot == pytest.approx(_reference_p(model, pid, term, cov_type), abs=1e-12)


def test_small_variance_draws_change_the_decision_at_five_percent():
    y = np.ravel([[0.0, d] for d in [-0.999999, -1, -1, 1, 1, 1, 1, -1]])
    model, pid, term = _from_outcomes(y, n_treated=3)
    fit = _fit(model, pid, "nonrobust")
    res = wild_cluster_bootstrap_t(fit, fit.model.exog, pid, term, B=999, cov_type="nonrobust")

    assert res.p_boot == 12 / 256
    assert not res.ci_lo <= 0.0 <= res.ci_hi


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
        sm.WLS(fit.fittedvalues + fit.resid * w[idx], exog, weights=weights).fit().params.iloc[j]
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
