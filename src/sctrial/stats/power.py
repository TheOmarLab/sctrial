r"""Power analysis utilities for trial planning.

This module provides sample size and power calculations for both
**two-arm DiD** and **single-arm paired** (pre/post) designs in
single-cell studies.

Mathematical Background
-----------------------
**Two-arm DiD** (treatment × time interaction):

    Power = Φ(\|δ\|/SE - z_{1-α/2}),   SE = σ × √(4/n)

    n = 4σ²(z_{1-α/2} + z_{1-β})² / δ²

**Single-arm paired** (within-arm pre→post change):

    Power = Φ(\|δ\|/SE - z_{1-α/2}),   SE = σ × √(2/n)

    n = 2σ²(z_{1-α/2} + z_{1-β})² / δ²

The paired design has half the variance of DiD (factor 2 vs 4)
because it involves one arm instead of two.

**Two groups of unequal size** (cross-sectional comparison, or a DiD whose
arms differ in size):

    Power = Φ(\|δ\|/SE - z_{1-α/2}),   SE = σ × √(1/n₁ + 1/n₂)

    n₂ = σ²(1 + r)/r × (z_{1-α/2} + z_{1-β})² / δ²,   n₁ = r × n₂

where r = n₁/n₂ is the allocation ratio. With n₁ = n₂ this is the two-arm
DiD formula.

These formulas use the normal distribution. The two-group and paired functions
also take ``reference="t"``, which gives the exact power of the two-sample or
paired t-test from the noncentral t distribution (n₁ + n₂ - 2 or n - 1 degrees of
freedom). The normal approximation overstates power in small samples, so
``reference="t"`` is the one to use for a handful of participants.

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

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy import optimize, stats

__all__ = [
    "power_did",
    "sample_size_did",
    "sensitivity_analysis",
    "power_curve",
    "power_paired",
    "sample_size_paired",
    "sensitivity_paired",
    "power_two_group",
    "sample_size_two_group",
    "sensitivity_two_group",
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
    if not (0 <= icc <= 1):
        raise ValueError("icc must be between 0 and 1.")
    if not np.isfinite(cluster_size) or cluster_size <= 0:
        raise ValueError("cluster_size must be > 0.")
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
    if not np.isfinite(n_clusters) or n_clusters <= 0:
        raise ValueError("n_clusters must be > 0.")
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
    Power: 35.3%

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
    n_base = 4 * (sigma**2) * ((z_alpha + z_beta) ** 2) / (effect_size**2)

    # Apply design effect
    if icc > 0 and cluster_size > 1:
        deff = design_effect(cluster_size, icc)
        n_required = n_base * deff
    else:
        n_required = n_base

    def reaches(n: int) -> bool:
        reached = power_did(n, effect_size, sigma, alpha, two_sided, icc, cluster_size)
        return reached >= power - _POWER_TOL

    # Participants per arm (the formula gives the total), at least 2
    return _smallest_sufficient(int(np.ceil(n_required / 2)), 2, reaches)


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
    powers = np.array(
        [power_did(n, effect_size, sigma, alpha, icc=icc, cluster_size=cluster_size) for n in ns]
    )
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


# ======================================================================
# Reference distribution shared by the paired and two-group functions
# ======================================================================

Reference = Literal["normal", "t"]


def _check_reference(reference: str) -> None:
    if reference not in ("normal", "t"):
        raise ValueError("reference must be 'normal' or 't'.")


# A power within this of the target reaches it: the target effect of a round trip through a
# sensitivity function is itself the root of a numerical solve.
_POWER_TOL = 1e-9


def _smallest_sufficient(n_start: int, n_min: int, reaches: Callable[[int], bool]) -> int:
    """Smallest size from ``n_min`` at which ``reaches`` holds, searched from ``n_start``.

    A closed-form or root-found size rounded up is one too large when the real-valued
    solution is an integer up to rounding, so the size is settled with the power function.
    """
    n = max(n_start, n_min)
    while not reaches(n):
        n += 1
    while n > n_min and reaches(n - 1):
        n -= 1
    return n


def _power_at_ncp(ncp: float, alpha: float, two_sided: bool, df: float | None) -> float:
    """Power of a test whose statistic has noncentrality ``ncp``.

    ``df=None`` is the normal approximation; otherwise the statistic follows a
    noncentral t distribution with ``df`` degrees of freedom.
    """
    tail = alpha / 2 if two_sided else alpha
    if df is None:
        crit = stats.norm.ppf(1 - tail)
        power = stats.norm.sf(crit - ncp)
        if two_sided:
            power += stats.norm.cdf(-crit - ncp)
        return float(power)
    crit = stats.t.ppf(1 - tail, df)
    # SciPy raises floating-point flags inside the noncentral t (a division by zero at one
    # degree of freedom, an invalid value at very large noncentrality) while returning the
    # correct probability.
    with np.errstate(divide="ignore", invalid="ignore"):
        power = stats.nct.sf(crit, df, ncp)
        if two_sided:
            power += stats.nct.cdf(-crit, df, ncp)
    return float(power)


def _ncp_at_power(power: float, alpha: float, two_sided: bool, df: float | None) -> float:
    """Noncentrality at which the test reaches ``power``."""
    if df is None:
        tail = alpha / 2 if two_sided else alpha
        return float(stats.norm.ppf(1 - tail) + stats.norm.ppf(power))
    if _power_at_ncp(0.0, alpha, two_sided, df) >= power:
        return 0.0
    upper = 1.0
    while _power_at_ncp(upper, alpha, two_sided, df) < power:
        upper *= 2.0
    return float(
        optimize.brentq(lambda ncp: _power_at_ncp(ncp, alpha, two_sided, df) - power, 0.0, upper)
    )


# ======================================================================
# Paired (single-arm pre/post) power functions
# ======================================================================


def power_paired(
    n_participants: int,
    effect_size: float,
    sigma: float = 1.0,
    alpha: float = 0.05,
    two_sided: bool = True,
    reference: Reference = "normal",
) -> float:
    """Calculate statistical power for a paired pre/post comparison.

    This function computes the power to detect a given within-arm
    longitudinal effect (post − pre) with the specified sample size.

    Model:
        Y_post - Y_pre = δ + ε   (paired differences)

    H₀: δ = 0  (no change)
    H₁: δ ≠ 0  (treatment causes change)

    Parameters
    ----------
    n_participants
        Number of participants with paired measurements.
    effect_size
        Expected pre-to-post effect size (δ) in original units.
        Can also be Cohen's d if sigma=1.
    sigma
        Within-group standard deviation of the outcome (not the
        paired difference SD; the paired difference variance is 2σ²).
    alpha
        Significance level (default 0.05).
    two_sided
        If True (default), use two-sided test.
    reference
        ``"normal"`` (default) for the normal approximation, ``"t"`` for the exact
        power of the paired t-test (noncentral t, ``n_participants - 1`` degrees of
        freedom). Use ``"t"`` for small samples.

    Returns
    -------
    float
        Statistical power (probability of rejecting H₀ when H₁ is true).

    Examples
    --------
    >>> # 6 participants, effect = 0.8 SD
    >>> pwr = power_paired(n_participants=6, effect_size=0.8, sigma=1.0)
    >>> print(f"Power: {pwr:.1%}")
    >>> pwr_t = power_paired(n_participants=6, effect_size=0.8, sigma=1.0, reference="t")

    Notes
    -----
    For paired pre/post, the variance of the time coefficient is:

        Var(β_time) = 2σ² / n

    where n is the number of paired participants.  This is half the
    variance of a DiD coefficient (4σ²/n) because there is only one
    arm instead of two.
    """
    if not np.isfinite(n_participants) or n_participants < 1:
        raise ValueError("n_participants must be a finite integer >= 1.")
    if int(n_participants) != n_participants:
        raise ValueError("n_participants must be an integer.")
    if not np.isfinite(effect_size):
        raise ValueError("effect_size must be finite.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite value > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    _check_reference(reference)
    if n_participants < 2:
        return 0.0

    # Standard error for paired comparison: SE = σ × √(2/n)
    se = sigma * np.sqrt(2.0 / n_participants)
    ncp = abs(effect_size) / se
    df = n_participants - 1 if reference == "t" else None
    return _power_at_ncp(ncp, alpha, two_sided, df)


def sample_size_paired(
    effect_size: float,
    sigma: float = 1.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    reference: Reference = "normal",
) -> int:
    """Calculate required sample size for a paired pre/post comparison.

    Determines the number of participants with paired measurements
    needed to achieve the desired power.

    Parameters
    ----------
    effect_size
        Minimum detectable effect size (must be > 0).
    sigma
        Expected within-group standard deviation.
    power
        Desired statistical power (default 0.80).
    alpha
        Significance level (default 0.05).
    two_sided
        If True (default), use two-sided test.
    reference
        ``"normal"`` (default) or ``"t"``. With ``"t"`` the result is the smallest
        number of participants at which the paired t-test reaches the power.

    Returns
    -------
    int
        Required number of paired participants.

    Examples
    --------
    >>> n = sample_size_paired(effect_size=0.5, sigma=1.0, power=0.80)
    >>> print(f"Need {n} paired participants")
    """
    if not np.isfinite(effect_size) or effect_size <= 0:
        raise ValueError("effect_size must be a finite value > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite value > 0.")
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    _check_reference(reference)

    if two_sided:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    # n = 2σ²(z_α + z_β)² / δ²
    n = 2.0 * sigma**2 * (z_alpha + z_beta) ** 2 / effect_size**2

    def reaches(n_participants: int) -> bool:
        reached = power_paired(n_participants, effect_size, sigma, alpha, two_sided, reference)
        return reached >= power - _POWER_TOL

    # The t-test needs at least as many participants as the normal formula.
    return _smallest_sufficient(int(np.ceil(n)), 2, reaches)


def sensitivity_paired(
    n_participants: int,
    sigma: float = 1.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    reference: Reference = "normal",
) -> float:
    """Calculate the minimum detectable effect for a paired comparison.

    Given a fixed sample size, what is the smallest effect that can
    be detected with the desired power?

    Parameters
    ----------
    n_participants
        Number of paired participants.
    sigma
        Within-group standard deviation.
    power
        Desired power.
    alpha
        Significance level.
    two_sided
        Use two-sided test.
    reference
        ``"normal"`` (default) or ``"t"`` (paired t-test, noncentral t).

    Returns
    -------
    float
        Minimum detectable effect size.

    Examples
    --------
    >>> mde = sensitivity_paired(n_participants=6, power=0.80)
    >>> print(f"Minimum detectable effect: {mde:.2f} SD")
    """
    _check_reference(reference)
    if n_participants < 2:
        return np.inf

    # δ = σ × √(2/n) × ncp, with ncp = z_α + z_β under the normal approximation
    se_unit = sigma * np.sqrt(2.0 / n_participants)
    df = n_participants - 1 if reference == "t" else None
    return float(se_unit * _ncp_at_power(power, alpha, two_sided, df))


# ======================================================================
# Two-group power functions (unequal group sizes)
# ======================================================================


def _check_two_group_inputs(
    n_treated: int, n_control: int, sigma: float, alpha: float
) -> None:
    for name, n in (("n_treated", n_treated), ("n_control", n_control)):
        if not np.isfinite(n) or n < 1:
            raise ValueError(f"{name} must be a finite integer >= 1.")
        if int(n) != n:
            raise ValueError(f"{name} must be an integer.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite value > 0.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")


def power_two_group(
    n_treated: int,
    n_control: int,
    effect_size: float,
    sigma: float = 1.0,
    alpha: float = 0.05,
    two_sided: bool = True,
    reference: Reference = "normal",
) -> float:
    """Calculate statistical power for a difference between two groups of any size.

    The compared quantity has one value per participant: the outcome itself in a
    cross-sectional comparison (for example severe versus mild disease), or the
    within-participant change in a DiD whose arms differ in size. Its standard
    error is ``sigma * sqrt(1/n_treated + 1/n_control)``. With equal group sizes
    this is ``power_did`` without clustering.

    Parameters
    ----------
    n_treated
        Number of participants in the first group.
    n_control
        Number of participants in the second group.
    effect_size
        Expected difference between the group means, in the units of ``sigma``.
    sigma
        Within-group standard deviation of the compared quantity.
    alpha
        Significance level (default 0.05).
    two_sided
        If True (default), use a two-sided test.
    reference
        ``"normal"`` (default) for the normal approximation, ``"t"`` for the exact
        power of the two-sample t-test with pooled variance (noncentral t,
        ``n_treated + n_control - 2`` degrees of freedom). Use ``"t"`` for small
        samples.

    Returns
    -------
    float
        Statistical power. A group with fewer than two participants gives 0.0.

    Examples
    --------
    >>> # 5 severe and 11 mild participants, difference of one SD
    >>> pwr = power_two_group(n_treated=5, n_control=11, effect_size=1.0)
    >>> print(f"Power: {pwr:.1%}")
    Power: 45.8%
    """
    _check_two_group_inputs(n_treated, n_control, sigma, alpha)
    if not np.isfinite(effect_size):
        raise ValueError("effect_size must be finite.")
    _check_reference(reference)
    if min(n_treated, n_control) < 2:
        return 0.0

    se = sigma * np.sqrt(1.0 / n_treated + 1.0 / n_control)
    ncp = abs(effect_size) / se
    df = n_treated + n_control - 2 if reference == "t" else None
    return _power_at_ncp(ncp, alpha, two_sided, df)


def sample_size_two_group(
    effect_size: float,
    sigma: float = 1.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    allocation_ratio: float = 1.0,
    reference: Reference = "normal",
) -> tuple[int, int]:
    """Calculate the group sizes required for a two-group comparison.

    Parameters
    ----------
    effect_size
        Expected difference between the group means, in the units of ``sigma``.
    sigma
        Within-group standard deviation of the compared quantity.
    power
        Desired statistical power (default 0.80).
    alpha
        Significance level (default 0.05).
    two_sided
        If True (default), use a two-sided test.
    allocation_ratio
        Size of the first group relative to the second, ``n_treated / n_control``.
        Use the ratio expected in the planned study, for example 3/8 when three
        responders are expected for every eight non-responders.
    reference
        ``"normal"`` (default) or ``"t"`` (two-sample t-test, noncentral t).

    Returns
    -------
    tuple[int, int]
        ``(n_treated, n_control)``: the smallest sizes that reach the desired power
        according to ``power_two_group``, with the smaller group equal to its share
        of the larger group rounded up to a whole participant. Each is at least 2.

    Examples
    --------
    >>> sample_size_two_group(effect_size=1.0, allocation_ratio=5 / 11)
    (12, 25)
    """
    if effect_size == 0 or not np.isfinite(effect_size):
        raise ValueError("effect_size must be finite and non-zero.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite value > 0.")
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")
    if not np.isfinite(allocation_ratio) or allocation_ratio <= 0:
        raise ValueError("allocation_ratio must be a finite value > 0.")
    _check_reference(reference)

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    # sigma^2 (1/(r n_c) + 1/n_c) <= (delta / (z_alpha + z_beta))^2, with r = n_t / n_c
    n_control = (
        sigma**2 * (1.0 + allocation_ratio) / allocation_ratio * (z_alpha + z_beta) ** 2
    ) / effect_size**2

    # Sizes are whole participants, so the allocation holds up to rounding: for a given size
    # of the larger group, the smaller group is its share rounded up.
    treated_is_smaller = allocation_ratio <= 1.0
    share = allocation_ratio if treated_is_smaller else 1.0 / allocation_ratio

    def sizes(n_larger: int) -> tuple[int, int]:
        n_smaller = max(int(np.ceil(share * n_larger - 1e-9)), 2)
        return (n_smaller, n_larger) if treated_is_smaller else (n_larger, n_smaller)

    def reaches(n_larger: int) -> bool:
        n_treated, n_reference = sizes(n_larger)
        reached = power_two_group(
            n_treated, n_reference, effect_size, sigma, alpha, two_sided, reference
        )
        return reached >= power - _POWER_TOL

    # The normal formula gives the starting point; the t-test needs at least as many.
    n_larger = n_control if treated_is_smaller else allocation_ratio * n_control
    return sizes(_smallest_sufficient(int(np.ceil(n_larger)), 2, reaches))


def sensitivity_two_group(
    n_treated: int,
    n_control: int,
    sigma: float = 1.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    reference: Reference = "normal",
) -> float:
    """Calculate the minimum detectable difference between two groups of any size.

    Parameters
    ----------
    n_treated
        Number of participants in the first group.
    n_control
        Number of participants in the second group.
    sigma
        Within-group standard deviation of the compared quantity.
    power
        Desired power.
    alpha
        Significance level.
    two_sided
        Use a two-sided test.
    reference
        ``"normal"`` (default) or ``"t"`` (two-sample t-test, noncentral t).

    Returns
    -------
    float
        Minimum detectable difference, in the units of ``sigma``. Infinite when
        a group has fewer than two participants.

    Examples
    --------
    >>> mde = sensitivity_two_group(n_treated=3, n_control=8)
    >>> print(f"Minimum detectable difference: {mde:.2f} SD")
    Minimum detectable difference: 1.90 SD
    """
    _check_two_group_inputs(n_treated, n_control, sigma, alpha)
    if not (0 < power < 1):
        raise ValueError("power must be between 0 and 1.")
    _check_reference(reference)
    if min(n_treated, n_control) < 2:
        return np.inf

    se_unit = sigma * np.sqrt(1.0 / n_treated + 1.0 / n_control)
    df = n_treated + n_control - 2 if reference == "t" else None
    return float(se_unit * _ncp_at_power(power, alpha, two_sided, df))
