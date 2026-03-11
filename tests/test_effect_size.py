"""Tests for effect size calculations."""

import numpy as np
import pandas as pd
import pytest

from sctrial.stats.effect_size import (
    add_effect_sizes_to_did,
    bootstrap_effect_size_ci,
    cohens_d,
    cohens_d_from_did,
    effect_size_ci,
    hedges_g,
)


class TestCohensD:
    """Test Cohen's d effect size."""

    def test_identical_groups_returns_zero(self):
        """Identical groups should have zero effect size."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(group, group) == pytest.approx(0.0, abs=1e-10)

    def test_positive_effect(self):
        """Higher group1 should give positive d."""
        group1 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(group1, group2)
        assert d > 0

    def test_negative_effect(self):
        """Lower group1 should give negative d."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        d = cohens_d(group1, group2)
        assert d < 0

    def test_known_value(self):
        """Test against known value."""
        group1 = np.array([-1.0, 0.0, 1.0])  # mean=0, sd=1
        group2 = np.array([0.0, 1.0, 2.0])  # mean=1, sd=1
        d = cohens_d(group1, group2)
        assert d == pytest.approx(-1.0, abs=0.1)

    def test_symmetric(self):
        """d(A, B) = -d(B, A)."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        d1 = cohens_d(group1, group2)
        d2 = cohens_d(group2, group1)
        assert d1 == pytest.approx(-d2, abs=1e-10)


class TestHedgesG:
    """Test Hedge's g (bias-corrected effect size)."""

    def test_smaller_than_cohens_d(self):
        """Hedge's g should be smaller than Cohen's d for small samples."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0])
        d = cohens_d(group1, group2)
        g = hedges_g(group1, group2)
        # g should be closer to zero (smaller in absolute value)
        assert abs(g) <= abs(d) + 1e-10

    def test_converges_to_d_for_large_samples(self):
        """Hedge's g should converge to Cohen's d for large n."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 1000)
        group2 = np.random.normal(1, 1, 1000)
        d = cohens_d(group1, group2)
        g = hedges_g(group1, group2)
        # Should be very close for large samples
        assert d == pytest.approx(g, rel=0.01)

    def test_identical_groups_returns_zero(self):
        """Identical groups should have zero effect size."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert hedges_g(group, group) == pytest.approx(0.0, abs=1e-10)


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_returns_three_values(self):
        """Should return (effect, ci_low, ci_high)."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)

        result = bootstrap_effect_size_ci(group1, group2, n_boot=100, seed=42)
        assert len(result) == 3
        effect, ci_low, ci_high = result
        assert ci_low < ci_high

    def test_reproducible_with_seed(self):
        """Same seed should give same results."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        result1 = bootstrap_effect_size_ci(group1, group2, n_boot=100, seed=123)
        result2 = bootstrap_effect_size_ci(group1, group2, n_boot=100, seed=123)

        assert result1[0] == pytest.approx(result2[0], abs=1e-10)
        assert result1[1] == pytest.approx(result2[1], abs=1e-10)
        assert result1[2] == pytest.approx(result2[2], abs=1e-10)


class TestEffectSizeCI:
    """Test confidence interval for effect sizes."""

    def test_ci_contains_point_estimate(self):
        """CI should contain the point estimate."""
        lower, upper = effect_size_ci(0.5, n1=20, n2=20)
        assert lower < 0.5 < upper

    def test_ci_widens_with_smaller_n(self):
        """CI should be wider with smaller sample sizes."""
        lo_big, hi_big = effect_size_ci(0.5, n1=100, n2=100)
        lo_small, hi_small = effect_size_ci(0.5, n1=5, n2=5)
        assert (hi_small - lo_small) > (hi_big - lo_big)

    def test_nan_for_insufficient_n(self):
        """Should return NaN for n < 2."""
        lower, upper = effect_size_ci(0.5, n1=1, n2=10)
        assert np.isnan(lower) and np.isnan(upper)

    def test_nan_for_nan_d(self):
        """Should return NaN for NaN effect size."""
        lower, upper = effect_size_ci(np.nan, n1=10, n2=10)
        assert np.isnan(lower) and np.isnan(upper)


class TestCohensDFromDiD:
    """Test Cohen's d from DiD change scores."""

    def test_known_difference(self):
        """Groups with known mean difference should produce expected d."""
        delta_t = np.array([2.0, 3.0, 4.0, 3.0, 2.5])
        delta_c = np.array([0.0, 0.5, -0.5, 0.0, 0.5])
        d = cohens_d_from_did(delta_t, delta_c)
        assert d > 0  # Treated deltas are larger
        assert np.isfinite(d)

    def test_identical_deltas_returns_zero(self):
        """Identical change scores should give d=0."""
        delta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d_from_did(delta, delta)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        """Should return NaN with too few data points."""
        d = cohens_d_from_did(np.array([1.0]), np.array([2.0]))
        assert np.isnan(d)


class TestAddEffectSizesToDiD:
    """Test adding effect sizes to DiD results DataFrame."""

    def test_adds_columns(self):
        """Should add effect_size, CI, and interpretation columns."""
        df = pd.DataFrame(
            {
                "feature": ["A", "B"],
                "beta_DiD": [0.5, -1.2],
                "se_DiD": [0.1, 0.3],
                "n_units": [20, 20],
            }
        )
        result = add_effect_sizes_to_did(df)
        assert "effect_size" in result.columns
        assert "effect_size_lower" in result.columns
        assert "effect_size_upper" in result.columns
        assert "effect_size_interpretation" in result.columns
        assert all(np.isfinite(result["effect_size"]))

    def test_ci_contains_effect_size(self):
        """CI should contain the point estimate."""
        df = pd.DataFrame(
            {
                "feature": ["A"],
                "beta_DiD": [0.5],
                "se_DiD": [0.1],
                "n_units": [30],
            }
        )
        result = add_effect_sizes_to_did(df)
        assert result.iloc[0]["effect_size_lower"] < result.iloc[0]["effect_size"]
        assert result.iloc[0]["effect_size"] < result.iloc[0]["effect_size_upper"]

    def test_nan_beta_produces_nan(self):
        """NaN beta should produce NaN effect size."""
        df = pd.DataFrame(
            {
                "feature": ["A"],
                "beta_DiD": [np.nan],
                "se_DiD": [0.1],
                "n_units": [20],
            }
        )
        result = add_effect_sizes_to_did(df)
        assert np.isnan(result.iloc[0]["effect_size"])

    def test_hedges_g_default(self):
        """Default method should be hedges_g (smaller than cohens_d)."""
        df = pd.DataFrame(
            {
                "feature": ["A"],
                "beta_DiD": [1.0],
                "se_DiD": [0.2],
                "n_units": [10],
            }
        )
        res_g = add_effect_sizes_to_did(df, method="hedges_g")
        res_d = add_effect_sizes_to_did(df, method="cohens_d")
        assert abs(res_g.iloc[0]["effect_size"]) <= abs(res_d.iloc[0]["effect_size"]) + 1e-10
