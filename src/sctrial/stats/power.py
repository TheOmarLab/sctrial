"""Power analysis utilities for trial planning.

This module provides sample size and power calculations for Difference-in-Differences
(DiD) analyses in single-cell studies.

Mathematical Background
-----------------------
For a DiD design with equal group sizes n per arm×time cell:

    Power = Φ(``|delta|``/SE - z_{1-α/2})

where:
    - delta = expected DiD effect size
    - SE = σ × √(4/n) for equal groups
    - σ = within-group standard deviation
    - Φ = standard normal CDF
    - z_{1-α/2} = critical value (1.96 for α=0.05)

Solving for n given desired power (1-β):

    n = 4σ²(z_{1-α/2} + z_{1-β})² / delta²

For cluster-robust inference with ICC (intraclass correlation):

    Design effect = 1 + (m-1)×ICC

where m = average cluster size.

The effective sample size is:
    n_eff = n / design_effect

Key Considerations for Single-Cell Studies
-------------------------------------------
1. **Unit of analysis**: Participants, not cells. Power is driven by the
   number of participants, not cells per participant.

2. **ICC matters**: High within-participant correlation (cells from same donor
   are similar) inflates variance estimates. Account for this in power calculations.

3. **Cell-type stratification**: Testing multiple cell types increases multiple
   testing burden but may reveal cell-type-specific effects.

4. **Effect size uncertainty**: Use pilot data or published studies to estimate
   expected effect sizes. Be conservative.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "power_did",
    "sample_size_did",
    "power_curve",
    "design_effect",
    "effective_sample_size",
]


def design_effect(
    cluster_size: float,
    icc: float,
) -> float:
    """Calculate the design effect for clustered data.

    The design effect (DEFF) quantifies how much clustering inflates
    variance compared to simple random sampling:

        DEFF = 1 + (m - 1) × ICC

    where m is the average cluster size and ICC is the intraclass
    correlation coefficient.

    Parameters
    ----------
    cluster_size
        Average number of observations per cluster (e.g., cells per participant).
    icc
        Intraclass correlation coefficient (0 to 1).
        - ICC ≈ 0: observations within clusters are independent
        - ICC ≈ 1: observations within clusters are identical

    Returns
    -------
    float
        Design effect. Values > 1 indicate variance inflation.

    Examples
    --------
    >>> # 100 cells per participant, ICC=0.05
    >>> deff = design_effect(100, 0.05)
    >>> print(f"Design effect: {deff:.1f}")
    Design effect: 5.95
    """
    return 1 + (cluster_size - 1) * icc


def effective_sample_size(
    n_clusters: int,
    cluster_size: float,
    icc: float,
) -> float:
    """Calculate the effective sample size accounting for clustering.

    The effective sample size is the equivalent number of independent
    observations after accounting for within-cluster correlation:

        n_eff = (n_clusters × cluster_size) / DEFF

    Parameters
    ----------
    n_clusters
        Number of clusters (e.g., participants).
    cluster_size
        Average observations per cluster.
    icc
        Intraclass correlation coefficient.

    Returns
    -------
    float
        Effective sample size.

    Notes
    -----
    For single-cell studies, the effective sample size is typically
    close to the number of participants, not the number of cells,
    when aggregating to participant level.
    """
    deff = design_effect(cluster_size, icc)
    n_total = n_clusters * cluster_size
    return n_total / deff


def power_did(
    n_per_group: int,
    effect_size: float,
    sigma: float = 1.0,
    alpha: float = 0.05,
    two_sided: bool = True,
    icc: float = 0.0,
    cluster_size: float = 1.0,
) -> float:
    """Calculate statistical power for a DiD analysis.

    This function computes the power to detect a given DiD effect
    (treatment × time interaction) with the specified sample size.

    Model:
        Y_it = α + β₁×Treat + β₂×Post + β₃×(Treat×Post) + ε

    H₀: β₃ = 0  (no differential change)
    H₁: β₃ ≠ 0  (treatment causes differential change)

    Parameters
    ----------
    n_per_group
        Number of participants per arm (assumes balanced design).
        Total participants = 2 × n_per_group.
    effect_size
        Expected DiD effect size (β₃) in original units.
        Can also be Cohen's d if sigma=1.
    sigma
        Within-group standard deviation of the outcome.
    alpha
        Significance level (default 0.05).
    two_sided
        If True (default), use two-sided test.
    icc
        Intraclass correlation for cluster adjustment.
    cluster_size
        Average cluster size (e.g., cells per participant).

    Returns
    -------
    float
        Statistical power (probability of rejecting H₀ when H₁ is true).

    Examples
    --------
    >>> # 20 participants per arm, effect = 0.5 SD
    >>> pwr = power_did(n_per_group=20, effect_size=0.5, sigma=1.0)
    >>> print(f"Power: {pwr:.1%}")
    Power: 69.8%

    >>> # With clustering (100 cells/participant, ICC=0.05)
    >>> pwr = power_did(20, 0.5, icc=0.05, cluster_size=100)
    >>> print(f"Power with clustering: {pwr:.1%}")

    Notes
    -----
    For DiD, the variance of the interaction term is approximately:

        Var(β₃) = 4σ² / n

    where n is the total number of participants (not cells).
    """
    if not np.isfinite(n_per_group) or n_per_group < 1:
        raise ValueError("n_per_group must be a finite integer >= 1.")
    if int(n_per_group) != n_per_group:
        raise ValueError("n_per_group must be an integer.")
    if not np.isfinite(effect_size):
        raise ValueError("effect_size must be finite.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite value > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    if not (0 <= icc <= 1):
        raise ValueError("icc must be between 0 and 1.")
    if not np.isfinite(cluster_size) or cluster_size < 1:
        raise ValueError("cluster_size must be >= 1.")
    if n_per_group < 2:
        return 0.0

    # Total participants
    n_total = 2 * n_per_group

    # Apply design effect if clustering
    if icc > 0 and cluster_size > 1:
        deff = design_effect(cluster_size, icc)
        n_eff = n_total / deff
    else:
        n_eff = n_total

    # Standard error of DiD coefficient
    # For balanced 2x2 DiD: SE = σ × √(4/n)
    se = sigma * np.sqrt(4 / n_eff)

    # Noncentrality parameter
    ncp = abs(effect_size) / se

    # Critical value
    if two_sided:
        z_crit = stats.norm.ppf(1 - alpha / 2)
    else:
        z_crit = stats.norm.ppf(1 - alpha)

    # Power = P(Z > z_crit - ncp) for one-sided
    # For two-sided, use both tails
    if two_sided:
        power = stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    else:
        power = stats.norm.sf(z_crit - ncp)

    return float(power)


def sample_size_did(
    effect_size: float,
    sigma: float = 1.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    icc: float = 0.0,
    cluster_size: float = 1.0,
) -> int:
    """Calculate required sample size for a DiD analysis.

    Determines the number of participants per arm needed to achieve
    the desired power for detecting a given effect size.

    Parameters
    ----------
    effect_size
        Minimum detectable DiD effect size.
    sigma
        Expected within-group standard deviation.
    power
        Desired statistical power (default 0.80).
    alpha
        Significance level (default 0.05).
    two_sided
        If True (default), use two-sided test.
    icc
        Intraclass correlation for cluster adjustment.
    cluster_size
        Average cluster size.

    Returns
    -------
    int
        Required participants per arm (total = 2 × this value).

    Examples
    --------
    >>> # Sample size to detect d=0.5 with 80% power
    >>> n = sample_size_did(effect_size=0.5, sigma=1.0, power=0.80)
    >>> print(f"Need {n} participants per arm ({2*n} total)")

    >>> # With multiple testing correction (Bonferroni for 10 tests)
    >>> n = sample_size_did(effect_size=0.5, alpha=0.05/10, power=0.80)
    >>> print(f"With Bonferroni: {n} per arm")
    """
    if not np.isfinite(effect_size) or effect_size == 0:
        raise ValueError("effect_size must be non-zero for sample size calculation.")
    if effect_size < 0:
        raise ValueError("effect_size must be > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite value > 0.")
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    if not (0 <= icc <= 1):
        raise ValueError("icc must be between 0 and 1.")
    if not np.isfinite(cluster_size) or cluster_size < 1:
        raise ValueError("cluster_size must be >= 1.")

    # Critical values
    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    # Base sample size formula for DiD
    # n = 4σ²(z_α + z_β)² / δ²
    n_base = 4 * (sigma ** 2) * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)

    # Apply design effect
    if icc > 0 and cluster_size > 1:
        deff = design_effect(cluster_size, icc)
        n_required = n_base * deff
    else:
        n_required = n_base

    # Participants per arm (divide total by 2)
    n_per_arm = int(np.ceil(n_required / 2))

    return max(n_per_arm, 2)  # Minimum 2 per arm


def power_curve(
    n_range: tuple[int, int] = (5, 50),
    effect_size: float = 0.5,
    sigma: float = 1.0,
    alpha: float = 0.05,
    icc: float = 0.0,
    cluster_size: float = 1.0,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a power curve across sample sizes.

    Parameters
    ----------
    n_range
        (min, max) participants per arm.
    effect_size
        Expected DiD effect size.
    sigma
        Within-group standard deviation.
    alpha
        Significance level.
    icc
        Intraclass correlation.
    cluster_size
        Average cluster size.
    n_points
        Number of points to compute.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (sample_sizes, powers) arrays.

    Examples
    --------
    >>> ns, powers = power_curve(n_range=(10, 100), effect_size=0.3)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ns, powers)
    >>> plt.axhline(0.8, linestyle='--', label='80% power')
    >>> plt.xlabel('Participants per arm')
    >>> plt.ylabel('Power')
    >>> plt.show()
    """
    ns = np.linspace(n_range[0], n_range[1], n_points).astype(int)
    powers = np.array([
        power_did(n, effect_size, sigma, alpha, icc=icc, cluster_size=cluster_size)
        for n in ns
    ])
    return ns, powers


def sensitivity_analysis(
    n_per_group: int,
    sigma: float = 1.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Calculate the minimum detectable effect size for given power.

    This is the inverse of sample_size_did: given a fixed sample size,
    what is the smallest effect that can be detected?

    Parameters
    ----------
    n_per_group
        Number of participants per arm.
    sigma
        Within-group standard deviation.
    power
        Desired power.
    alpha
        Significance level.
    two_sided
        Use two-sided test.

    Returns
    -------
    float
        Minimum detectable effect size.

    Examples
    --------
    >>> # What effect can we detect with 15 per arm?
    >>> mde = sensitivity_analysis(n_per_group=15, power=0.80)
    >>> print(f"Minimum detectable effect: {mde:.2f} SD")
    """
    if n_per_group < 2:
        return np.inf

    n_total = 2 * n_per_group

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    # Solve for effect size: δ = σ × √(4/n) × (z_α + z_β)
    se_unit = sigma * np.sqrt(4 / n_total)
    mde = se_unit * (z_alpha + z_beta)

    return mde
