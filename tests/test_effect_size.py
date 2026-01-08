"""Tests for effect size calculations."""

import numpy as np
import pytest

from sctrial.stats.effect_size import (
    bootstrap_effect_size_ci,
    cohens_d,
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
        group2 = np.array([0.0, 1.0, 2.0])   # mean=1, sd=1
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
