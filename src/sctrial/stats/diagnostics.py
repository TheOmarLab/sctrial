from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

try:  # optional plotting
    import matplotlib.pyplot as plt  # type: ignore[assignment]
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def check_did_assumptions(
    fit: RegressionResultsWrapper,
    *,
    return_figures: bool = False,
) -> dict[str, Any]:
    """Run lightweight diagnostics for a fitted DiD model.

    Parameters
    ----------
    fit
        Fitted statsmodels RegressionResultsWrapper.
    return_figures
        If True, include matplotlib figures for residual diagnostics.

    Returns
    -------
    dict
        Dictionary with diagnostic statistics:
        - bp_pvalue: Breusch-Pagan p-value for heteroscedasticity
        - jb_pvalue: Jarque-Bera p-value for normality
        - resid_mean, resid_std
        - figures (optional): dict of matplotlib Figures
    """
    resid = np.asarray(fit.resid)
    exog = fit.model.exog

    bp = het_breuschpagan(resid, exog)
    jb = jarque_bera(resid)

    out: dict[str, Any] = {
        "bp_pvalue": float(bp[1]),
        "jb_pvalue": float(jb[1]),
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid, ddof=1)),
    }

    if return_figures:
        if plt is None:
            raise ImportError("matplotlib is required for return_figures=True")

        figs: dict[str, Any] = {}
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.scatter(fit.fittedvalues, resid, alpha=0.5)
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.set_xlabel("Fitted values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")
        figs["residuals_vs_fitted"] = fig1

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.hist(resid, bins=30, color="steelblue", alpha=0.7)
        ax2.set_title("Residuals Histogram")
        ax2.set_xlabel("Residual")
        figs["residuals_hist"] = fig2

        out["figures"] = figs

    return out
