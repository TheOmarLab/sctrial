"""sctrial DiD (first-difference) runner for benchmarking.

Benchmark representation of the participant-level DiD estimand.
With T=2 periods, first-differencing is equivalent to the FE estimator
(Wooldridge Ch.14) but avoids near-saturated models at small n.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fail_result(failure_mode: str = "numerical") -> dict:
    """Return a NaN result dict for a failed gene."""
    return {
        "beta": float("nan"),
        "pvalue": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "converged": False,
        "failure_mode": failure_mode,
    }


def _run_from_pseudobulk(
    pb: pd.DataFrame,
    gene_cols: list[str],
    design_type: str = "two_arm",
) -> dict:
    """Participant-level first-difference estimator on log-pseudobulk.

    **Two-arm (DiD):** β = mean(Δ_treated) − mean(Δ_control), Welch t-test.
    **Single-arm (paired):** β = mean(Δ), one-sample t-test vs 0.

    For a two-period design (Pre/Post), first-differencing is numerically
    equivalent to the FE estimator (Wooldridge, *Econometric Analysis*,
    Ch. 14). Computing Δ_i = Y_i,post − Y_i,pre per participant eliminates
    participant fixed effects by construction.
    """
    from scipy import stats as sp_stats

    out = {}
    df = pb.copy()

    # Pivot to one row per participant with Pre and Post values
    pre_df = df[df["visit"] == "Pre"].set_index("participant")
    post_df = df[df["visit"] == "Post"].set_index("participant")

    # Only participants with both visits
    common = pre_df.index.intersection(post_df.index)
    if len(common) == 0:
        for g in gene_cols:
            out[g] = _fail_result("numerical")
        return out

    pre_df = pre_df.loc[common]
    post_df = post_df.loc[common]

    # Detect arm labels — support non-standard labels (e.g. TNBC "anti-PDL1+Chemo"/"Chemo")
    arm_vals = df["arm"].dropna().unique().tolist() if "arm" in df.columns else []
    if "Treated" in arm_vals and "Control" in arm_vals:
        _treated_label, _control_label = "Treated", "Control"
    elif len(arm_vals) >= 2:
        _treated_label, _control_label = sorted(arm_vals)[-1], sorted(arm_vals)[0]
    else:
        _treated_label, _control_label = "Treated", "Control"

    for gene in gene_cols:
        try:
            # Participant-level change scores
            delta = post_df[gene].values - pre_df[gene].values

            if design_type == "single_arm":
                # Single-arm: one-sample t-test on Δ vs 0
                n = len(delta)
                if n < 2:
                    raise ValueError("Too few participants")

                beta = float(np.mean(delta))
                _, pval = sp_stats.ttest_1samp(delta, 0.0)

                se = float(np.std(delta, ddof=1) / np.sqrt(n))
                t_crit = sp_stats.t.ppf(0.975, n - 1)
                ci_lo = beta - t_crit * se
                ci_hi = beta + t_crit * se
            else:
                # Two-arm DiD: Welch t-test on Δ_treated vs Δ_control
                arms = pre_df["arm"]
                treated_mask = arms == _treated_label
                control_mask = arms == _control_label

                delta_treat = delta[treated_mask.values]
                delta_ctrl = delta[control_mask.values]

                n_treat = len(delta_treat)
                n_ctrl = len(delta_ctrl)

                if n_treat < 2 or n_ctrl < 2:
                    raise ValueError("Too few participants per arm")

                beta = float(np.mean(delta_treat) - np.mean(delta_ctrl))
                _, pval = sp_stats.ttest_ind(
                    delta_treat, delta_ctrl, equal_var=False
                )

                v1 = np.var(delta_treat, ddof=1) / n_treat
                v2 = np.var(delta_ctrl, ddof=1) / n_ctrl
                se = np.sqrt(v1 + v2)
                df_ws = (
                    (v1 + v2) ** 2
                    / (v1**2 / (n_treat - 1) + v2**2 / (n_ctrl - 1))
                    if (v1 + v2) > 0
                    else 1.0
                )
                t_crit = sp_stats.t.ppf(0.975, df_ws)
                ci_lo = beta - t_crit * se
                ci_hi = beta + t_crit * se

            out[gene] = {
                "beta": beta,
                "pvalue": float(pval),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "converged": True,
                "failure_mode": None,
            }
        except Exception as exc:
            logger.debug("sctrial_did gene %s failed: %s", gene, exc)
            out[gene] = _fail_result("numerical")

    return out


def run(
    data,
    gene_cols: list[str],
    design_kwargs: dict | None = None,
    visits: tuple[str, str] = ("Pre", "Post"),
    from_pseudobulk: bool = False,
    design_type: str = "two_arm",
) -> dict[str, dict]:
    """Run sctrial DiD / paired-difference estimator.

    Parameters
    ----------
    data : AnnData or DataFrame
        Cell-level AnnData (from_pseudobulk=False) or log-pseudobulk
        DataFrame (from_pseudobulk=True).
    gene_cols : list[str]
        Gene names to test.
    design_kwargs : dict, optional
        Override TrialDesign parameters (only used with AnnData path).
    visits : tuple
        (pre, post) visit labels.
    from_pseudobulk : bool
        If True, data is a pseudobulk DataFrame with columns:
        participant, arm, visit, gene_0, gene_1, ...
    design_type : str
        "two_arm" (DiD: Welch t on Δ_treat vs Δ_ctrl) or
        "single_arm" (paired: one-sample t on Δ vs 0).

    Returns
    -------
    dict : gene → {"beta", "pvalue", "ci_lo", "ci_hi", "converged", "failure_mode"}
    """
    if from_pseudobulk:
        return _run_from_pseudobulk(data, gene_cols, design_type=design_type)

    # Original cell-level path via sctrial.stats.did
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
                data,
                gene_cols,
                design,
                visits=visits,
                aggregate="cell",
                standardize=False,
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
        logger.warning("sctrial_did failed: %s", exc)
        for g in gene_cols:
            out[g] = {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }

    # Fill missing genes
    for g in gene_cols:
        if g not in out:
            out[g] = {
                "beta": np.nan,
                "pvalue": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "converged": False,
                "failure_mode": "numerical",
            }

    return out
