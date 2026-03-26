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


def run(
    pseudobulk: pd.DataFrame,
    gene_cols: list[str],
    arm_col: str = "arm",
    visit_col: str = "visit",
    participant_col: str = "participant",
    treated_label: str = "Treated",
    control_label: str = "Control",
    visits: tuple[str, str] = ("Pre", "Post"),
) -> dict[str, dict]:
    """Run Wilcoxon rank-sum on post−pre deltas between arms.

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

    treat_delta = delta[delta[arm_col] == treated_label]
    ctrl_delta = delta[delta[arm_col] == control_label]

    out = {}
    for g in gene_cols:
        try:
            t_vals = treat_delta[g].dropna().values
            c_vals = ctrl_delta[g].dropna().values

            if len(t_vals) < 2 or len(c_vals) < 2:
                out[g] = {
                    "beta": np.nan,
                    "pvalue": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "converged": False,
                    "failure_mode": "numerical",
                }
                continue

            stat, pval = sp_stats.mannwhitneyu(
                t_vals,
                c_vals,
                alternative="two-sided",
            )
            beta = t_vals.mean() - c_vals.mean()

            out[g] = {
                "beta": float(beta),
                "pvalue": float(pval),
                "ci_lo": np.nan,  # Wilcoxon does not produce CIs
                "ci_hi": np.nan,
                "converged": True,
                "failure_mode": None,
            }
        except Exception as exc:
            logger.debug("Wilcoxon failed for %s: %s", g, exc)
            out[g] = {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }

    return out
