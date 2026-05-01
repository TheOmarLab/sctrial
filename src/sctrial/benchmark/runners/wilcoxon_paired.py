"""Wilcoxon paired-delta runner for benchmarking.

Computes per-participant post−pre deltas on pseudobulk means, then
applies Mann-Whitney U between arms on those deltas. This is a
participant-aware, paired comparator — not a naive baseline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def _fail_result(mode: str = "numerical") -> dict:
    """Return a NaN result dict for a failed gene."""
    return {
        "beta": float("nan"),
        "pvalue": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "converged": False,
        "failure_mode": mode,
    }


def run(
    pseudobulk: pd.DataFrame,
    gene_cols: list[str],
    arm_col: str = "arm",
    visit_col: str = "visit",
    participant_col: str = "participant",
    treated_label: str = "Treated",
    control_label: str = "Control",
    visits: tuple[str, str] = ("Pre", "Post"),
    design_type: str = "two_arm",
) -> dict[str, dict]:
    """Wilcoxon test on post−pre deltas.

    **Two-arm:** Mann-Whitney U on Δ_treated vs Δ_control.
    **Single-arm:** Wilcoxon signed-rank on Δ vs 0.

    Returns
    -------
    dict : gene → {"beta", "pvalue", "ci_lo", "ci_hi", "converged", "failure_mode"}
    """

    pre = pseudobulk[pseudobulk[visit_col] == visits[0]].set_index(participant_col)
    post = pseudobulk[pseudobulk[visit_col] == visits[1]].set_index(participant_col)

    common = pre.index.intersection(post.index)
    if len(common) == 0:
        return {
            g: {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }
            for g in gene_cols
        }

    delta = post.loc[common, gene_cols] - pre.loc[common, gene_cols]
    delta[arm_col] = post.loc[common, arm_col]

    out = {}
    for g in gene_cols:
        try:
            if design_type == "single_arm":
                # Single-arm: Wilcoxon signed-rank on Δ vs 0
                vals = delta[g].dropna().values
                if len(vals) < 2:
                    raise ValueError("Too few participants")

                beta = float(vals.mean())
                # wilcoxon signed-rank test (paired, one-sample)
                stat, pval = sp_stats.wilcoxon(vals, alternative="two-sided")
            else:
                # Two-arm: Mann-Whitney U on Δ_treated vs Δ_control
                treat_delta = delta[delta[arm_col] == treated_label]
                ctrl_delta = delta[delta[arm_col] == control_label]

                t_vals = treat_delta[g].dropna().values
                c_vals = ctrl_delta[g].dropna().values

                if len(t_vals) < 2 or len(c_vals) < 2:
                    raise ValueError("Too few participants per arm")

                stat, pval = sp_stats.mannwhitneyu(
                    t_vals, c_vals, alternative="two-sided",
                )
                beta = float(t_vals.mean() - c_vals.mean())

            out[g] = {
                "beta": float(beta),
                "pvalue": float(pval),
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": True,
                "failure_mode": None,
            }
        except Exception as exc:
            logger.debug("Wilcoxon failed for %s: %s", g, exc)
            out[g] = _fail_result("numerical")

    return out
