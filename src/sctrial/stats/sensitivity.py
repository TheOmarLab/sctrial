from __future__ import annotations

import math


def e_value_rr(
    estimate: float,
    *,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> tuple[float, float | None]:
    """Compute the E-value for a risk ratio estimate (VanderWeele & Ding).
    Ref:VanderWeele, Tyler J., and Peng Ding. "Sensitivity analysis in observational research: introducing the E-value."
        Annals of internal medicine 167.4 (2017): 268-274.

    Parameters
    ----------
    estimate
        Risk ratio (RR) estimate (>0).
    ci_lower, ci_upper
        Optional confidence interval bounds on the RR scale.

    Returns
    -------
    (e_value, e_value_ci)
        E-value for the point estimate and for the confidence limit closest to 1.
    """
    if estimate <= 0:
        raise ValueError("estimate must be > 0 for E-value.")

    def _e_value(rr: float) -> float:
        """Compute the E-value."""
        if rr < 1:
            rr = 1 / rr
        return rr + math.sqrt(rr * (rr - 1))

    if ci_lower is not None and ci_upper is not None:
        if ci_lower <= 0 or ci_upper <= 0:
            raise ValueError("CI bounds must be > 0 for E-value.")
        if ci_lower > ci_upper:
            raise ValueError("ci_lower must be <= ci_upper.")
        if not (ci_lower <= estimate <= ci_upper):
            raise ValueError("estimate must fall within [ci_lower, ci_upper].")

    e_est = _e_value(estimate)

    e_ci = None
    if ci_lower is not None and ci_upper is not None:
        # use CI bound closest to 1
        if estimate >= 1:
            ci = ci_lower
        else:
            ci = ci_upper
        if ci is not None and ci > 0:
            e_ci = _e_value(ci)

    return e_est, e_ci
