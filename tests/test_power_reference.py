"""The t reference of the power functions and the residual SD the power table reads."""

import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import stats

import sctrial as st
from sctrial.stats.power import (
    power_paired,
    power_two_group,
    sample_size_paired,
    sample_size_two_group,
    sensitivity_paired,
    sensitivity_two_group,
)


def _nct_power(ncp, df, alpha=0.05):
    crit = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)


def test_default_reference_is_the_normal_approximation():
    assert power_two_group(5, 11, 1.0) == pytest.approx(0.4579, abs=1e-4)
    assert sample_size_two_group(1.0, allocation_ratio=5 / 11) == (12, 25)
    assert sensitivity_two_group(3, 8) == pytest.approx(1.8967, abs=1e-4)
    assert power_paired(6, 0.8) == power_paired(6, 0.8, reference="normal")
    assert sample_size_paired(0.5) == 63


@pytest.mark.parametrize("n1,n2,effect,sigma", [(6, 6, 1.4, 1.1), (3, 8, 1.2, 0.9), (20, 35, 0.4, 1.0)])
def test_two_group_t_power_is_the_two_sample_t_test(n1, n2, effect, sigma):
    ncp = effect / (sigma * np.sqrt(1 / n1 + 1 / n2))
    power = power_two_group(n1, n2, effect, sigma, reference="t")
    assert power == pytest.approx(_nct_power(ncp, n1 + n2 - 2))
    assert power < power_two_group(n1, n2, effect, sigma)
    one_sided = power_two_group(n1, n2, effect, sigma, two_sided=False, reference="t")
    assert one_sided == pytest.approx(stats.nct.sf(stats.t.ppf(0.95, n1 + n2 - 2), n1 + n2 - 2, ncp))


@pytest.mark.parametrize("n,effect,sigma", [(6, 0.8, 1.0), (11, 1.0, 0.6), (30, 0.3, 0.7)])
def test_paired_t_power_is_the_paired_t_test(n, effect, sigma):
    sd_change = np.sqrt(2) * sigma
    power = power_paired(n, effect, sigma, reference="t")
    assert power == pytest.approx(_nct_power(effect / (sd_change / np.sqrt(n)), n - 1))
    assert power < power_paired(n, effect, sigma)


def test_t_sample_sizes_are_the_smallest_that_reach_the_power():
    n = sample_size_paired(0.9, sigma=1.0, reference="t")
    assert power_paired(n, 0.9, reference="t") >= 0.80 > power_paired(n - 1, 0.9, reference="t")
    assert n >= sample_size_paired(0.9, sigma=1.0)

    n_t, n_c = sample_size_two_group(1.2, sigma=1.0, allocation_ratio=3 / 8, reference="t")
    assert power_two_group(n_t, n_c, 1.2, reference="t") >= 0.80
    assert n_t + n_c >= sum(sample_size_two_group(1.2, sigma=1.0, allocation_ratio=3 / 8))


@pytest.mark.parametrize("effect,power", [(1.0, 0.80), (0.6, 0.80), (1.4, 0.90), (2.5, 0.80)])
def test_t_sample_size_at_a_three_to_one_allocation_is_minimal_and_symmetric(effect, power):
    n_t, n_c = sample_size_two_group(effect, power=power, allocation_ratio=3.0, reference="t")
    assert 3 * (n_c - 1) < n_t <= 3 * n_c
    assert power_two_group(n_t, n_c, effect, reference="t") >= power
    assert power_two_group(3 * (n_c - 1), n_c - 1, effect, reference="t") < power
    assert sample_size_two_group(effect, power=power, allocation_ratio=1 / 3, reference="t") == (n_c, n_t)


def test_t_sample_size_does_not_skip_the_first_sufficient_allocation():
    # 33 and 11 participants reach 80% power for a difference of one SD; 36 and 12 are not needed.
    assert power_two_group(33, 11, 1.0, reference="t") >= 0.80 > power_two_group(30, 10, 1.0, reference="t")
    assert sample_size_two_group(1.0, allocation_ratio=3.0, reference="t") == (33, 11)


def test_t_sensitivity_inverts_the_power():
    mde = sensitivity_two_group(3, 8, sigma=1.3, reference="t")
    assert power_two_group(3, 8, mde, sigma=1.3, reference="t") == pytest.approx(0.80)
    mde = sensitivity_paired(6, sigma=0.7, power=0.9, reference="t")
    assert power_paired(6, mde, sigma=0.7, reference="t") == pytest.approx(0.90)
    assert sensitivity_paired(6, reference="t") > sensitivity_paired(6)


def test_unknown_reference_is_refused():
    for call in (lambda: power_two_group(5, 5, 1.0, reference="z"),
                 lambda: power_paired(5, 1.0, reference="z"),
                 lambda: sample_size_paired(1.0, reference="z"),
                 lambda: sample_size_two_group(1.0, reference="z"),
                 lambda: sensitivity_paired(5, reference="z"),
                 lambda: sensitivity_two_group(5, 5, reference="z")):
        with pytest.raises(ValueError, match="reference"):
            call()


def _cohort(n_treated, n_control, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_treated + n_control):
        arm = "T" if i < n_treated else "C"
        for visit in ("Pre", "Post"):
            for _ in range(2):
                rows.append({"participant_id": f"P{i:02d}", "arm": arm, "visit": visit,
                             "sig": float(rng.normal())})
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    return AnnData(X=np.zeros((len(obs), 1)), obs=obs)


def _changes(adata):
    means = adata.obs.groupby(["participant_id", "arm", "visit"], observed=True)["sig"].mean().unstack("visit")
    scale = means[["Pre", "Post"]].to_numpy().ravel().std(ddof=1)
    return ((means["Post"] - means["Pre"]) / scale).reset_index()


def test_did_resid_sd_times_sqrt2_is_the_pooled_sd_of_the_changes():
    adata = _cohort(3, 8, seed=11)
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col="arm",
                            arm_treated="T", arm_control="C", celltype_col=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.did_table(adata, features=["sig"], design=design, visits=("Pre", "Post"),
                           aggregate="participant_visit", standardize=True)
    change = _changes(adata)
    a, b = change.loc[change["arm"] == "T", 0], change.loc[change["arm"] == "C", 0]
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    assert np.sqrt(2) * res["resid_sd"].iloc[0] == pytest.approx(pooled, rel=1e-10)


def test_within_arm_resid_sd_times_sqrt2_is_the_sd_of_the_changes():
    adata = _cohort(7, 0, seed=12)
    design = st.TrialDesign(participant_col="participant_id", visit_col="visit", arm_col=None,
                            celltype_col=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.within_arm_comparison(adata, arm="All", features=["sig"], design=design,
                                       visits=("Pre", "Post"), standardize=True)
    change = _changes(adata)[0]
    assert np.sqrt(2) * res["resid_sd"].iloc[0] == pytest.approx(change.std(ddof=1), rel=1e-10)


@pytest.mark.parametrize("reference", ["t", "normal"])
@pytest.mark.parametrize("sizes", [(3, 8), (6, 6), (4, 12), (10, 30), (8, 3), (2, 9)])
def test_two_group_sample_size_returns_the_sizes_a_detectable_effect_came_from(sizes, reference):
    # 3 and 8 reach 80% power at their own detectable effect; rounding a solution that is an
    # integer up to rounding used to return 4 and 9.
    n_t, n_c = sizes
    effect = sensitivity_two_group(n_t, n_c, sigma=1.3, reference=reference)
    # the normal formula for the detectable effect leaves out the far tail (about 1e-6)
    assert power_two_group(n_t, n_c, effect, sigma=1.3, reference=reference) == pytest.approx(0.80, abs=2e-6)
    assert sample_size_two_group(
        effect, sigma=1.3, allocation_ratio=n_t / n_c, reference=reference
    ) == (n_t, n_c)


@pytest.mark.parametrize("reference", ["t", "normal"])
@pytest.mark.parametrize("n", [2, 3, 9, 17, 38])
def test_paired_sample_size_returns_the_size_a_detectable_effect_came_from(n, reference):
    effect = sensitivity_paired(n, sigma=0.7, reference=reference)
    assert sample_size_paired(effect, sigma=0.7, reference=reference) == n


@pytest.mark.parametrize("effect", [0.3, 0.5, 0.9, 1.4])
def test_did_sample_size_is_the_smallest_that_reaches_the_power(effect):
    from sctrial.stats.power import power_did, sample_size_did

    n = sample_size_did(effect)
    assert power_did(n, effect) >= 0.80
    assert n == 2 or power_did(n - 1, effect) < 0.80


def test_power_at_one_degree_of_freedom_raises_no_numerical_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert 0.05 < power_paired(2, 2.0, reference="t") < 1.0
        assert sample_size_paired(4.0, sigma=1.0, reference="t") >= 2
        assert power_paired(15, 11.0, two_sided=False, reference="t") == pytest.approx(1.0)


@pytest.mark.parametrize("df,ncp", [(1.0, 8.0), (1.0, 12.0), (9.0, 20.0), (30.0, 40.0)])
def test_two_sided_t_power_is_finite_where_the_lower_tail_routine_fails(df, ncp):
    # Recent SciPy returns NaN for the lower tail of the noncentral t at these values; the
    # two-sided power is the upper tail plus a far tail that is zero to many digits.
    from sctrial.stats.power import _power_at_ncp

    two_sided = _power_at_ncp(ncp, 0.05, True, df)
    upper = float(stats.nct.sf(stats.t.ppf(0.975, df), df, ncp))
    assert np.isfinite(two_sided)
    assert two_sided == pytest.approx(upper, abs=1e-12)
