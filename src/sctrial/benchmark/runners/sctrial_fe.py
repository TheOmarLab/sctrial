"""sctrial DiD (Fixed Effects) runner for benchmarking."""
from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def run(
    adata,
    gene_cols: list[str],
    design_kwargs: dict | None = None,
    visits: tuple[str, str] = ("Pre", "Post"),
) -> dict[str, dict]:
    """Run sctrial DiD with participant FE + cluster-robust SE.

    Parameters
    ----------
    adata : AnnData
        Cell-level data.
    gene_cols : list[str]
        Gene names to test.
    design_kwargs : dict, optional
        Override TrialDesign parameters.
    visits : tuple
        (pre, post) visit labels.

    Returns
    -------
    dict : gene → {"beta", "pvalue", "ci_lo", "ci_hi", "converged", "failure_mode"}
    """
    from sctrial.design import TrialDesign
    from sctrial.stats.did import did_table

    dk = {
        "participant_col": "participant",
        "visit_col": "visit",
        "arm_col": "arm",
        "arm_treated": "Treated",
        "arm_control": "Control",
    }
    if design_kwargs:
        dk.update(design_kwargs)

    design = TrialDesign(**dk)

    out = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = did_table(
                adata, gene_cols, design,
                visits=visits, aggregate="cell", standardize=False,
            )

        for _, row in res.iterrows():
            out[row["feature"]] = {
                "beta": row["beta_DiD"],
                "pvalue": row["p_DiD"],
                "ci_lo": row.get("ci_lo_DiD", np.nan),
                "ci_hi": row.get("ci_hi_DiD", np.nan),
                "converged": True,
                "failure_mode": None,
            }
    except Exception as exc:
        logger.warning("sctrial_fe failed: %s", exc)
        for g in gene_cols:
            out[g] = {
                "beta": np.nan, "pvalue": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "converged": False, "failure_mode": "numerical",
            }

    # Fill missing genes
    for g in gene_cols:
        if g not in out:
            out[g] = {
                "beta": np.nan, "pvalue": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "converged": False, "failure_mode": "numerical",
            }

    return out
