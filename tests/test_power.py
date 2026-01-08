"""Tests for power analysis functions."""

import numpy as np
import pytest

from sctrial.stats.power import (
    power_did,
    sample_size_did,
    design_effect,
    effective_sample_size,
)


class TestPowerDiD:
    """Test power calculations for DiD."""

    def test_power_increases_with_n(self):
        """Power should increase with sample size."""
        power_10 = power_did(n_per_group=10, effect_size=0.5)
        power_20 = power_did(n_per_group=20, effect_size=0.5)
        power_50 = power_did(n_per_group=50, effect_size=0.5)

        assert power_10 < power_20 < power_50

    def test_power_increases_with_effect_size(self):
        """Power should increase with effect size."""
        power_small = power_did(n_per_group=20, effect_size=0.2)
        power_medium = power_did(n_per_group=20, effect_size=0.5)
        power_large = power_did(n_per_group=20, effect_size=0.8)

        assert power_small < power_medium < power_large

    def test_power_bounded_0_1(self):
        """Power should be between 0 and 1."""
        for n in [5, 10, 20, 50, 100]:
            for d in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
                power = power_did(n_per_group=n, effect_size=d)
                assert 0 <= power <= 1

    def test_known_value_80_power(self):
        """Test against known value for ~80% power."""
        # For d=0.8, n≈26 per group gives ~80% power (two-sample t-test)
        power = power_did(n_per_group=26, effect_size=0.8)
        assert 0.70 < power < 0.90


class TestSampleSizeDiD:
    """Test sample size calculations."""

    def test_larger_n_for_smaller_effect(self):
        """Smaller effects require larger samples."""
        n_small_effect = sample_size_did(effect_size=0.3, power=0.80)
        n_large_effect = sample_size_did(effect_size=0.8, power=0.80)

        assert n_small_effect > n_large_effect

    def test_larger_n_for_higher_power(self):
        """Higher power requires larger samples."""
        n_80 = sample_size_did(effect_size=0.5, power=0.80)
        n_90 = sample_size_did(effect_size=0.5, power=0.90)

        assert n_80 < n_90

    def test_returns_integer(self):
        """Sample size should be an integer."""
        n = sample_size_did(effect_size=0.5, power=0.80)
        assert isinstance(n, int) or n == int(n)


class TestDesignEffect:
    """Test design effect calculations."""

    def test_design_effect_1_for_icc_0(self):
        """Design effect should be 1 when ICC=0."""
        de = design_effect(cluster_size=100, icc=0.0)
        assert de == pytest.approx(1.0, abs=1e-10)

    def test_design_effect_increases_with_icc(self):
        """Design effect should increase with ICC."""
        de_01 = design_effect(cluster_size=50, icc=0.01)
        de_05 = design_effect(cluster_size=50, icc=0.05)
        de_10 = design_effect(cluster_size=50, icc=0.10)

        assert de_01 < de_05 < de_10

    def test_design_effect_increases_with_cluster_size(self):
        """Design effect should increase with cluster size."""
        de_10 = design_effect(cluster_size=10, icc=0.05)
        de_50 = design_effect(cluster_size=50, icc=0.05)
        de_100 = design_effect(cluster_size=100, icc=0.05)

        assert de_10 < de_50 < de_100

    def test_known_formula(self):
        """Test against known formula: DE = 1 + (m-1)*ICC."""
        m = 50  # cluster size
        icc = 0.10
        expected_de = 1 + (m - 1) * icc  # = 1 + 49*0.1 = 5.9
        de = design_effect(cluster_size=m, icc=icc)
        assert de == pytest.approx(expected_de, abs=1e-10)


class TestEffectiveSampleSize:
    """Test effective sample size calculations."""

    def test_effective_n_decreases_with_icc(self):
        """Effective n should decrease with higher ICC."""
        n_clusters = 20
        cluster_size = 50

        eff_01 = effective_sample_size(n_clusters, cluster_size, icc=0.01)
        eff_05 = effective_sample_size(n_clusters, cluster_size, icc=0.05)
        eff_10 = effective_sample_size(n_clusters, cluster_size, icc=0.10)

        assert eff_01 > eff_05 > eff_10

    def test_consistency_with_design_effect(self):
        """Effective n should equal n_total / design_effect."""
        n_clusters = 20
        cluster_size = 50
        icc = 0.05
        n_total = n_clusters * cluster_size

        de = design_effect(cluster_size, icc)
        eff_n = effective_sample_size(n_clusters, cluster_size, icc)

        expected_eff_n = n_total / de
        assert eff_n == pytest.approx(expected_eff_n, abs=1e-10)
