"""Monte Carlo simulation engine for DiD method comparison.

Generates synthetic pseudobulk-style data with known ground-truth
DiD effects, enabling controlled comparison of statistical methods.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_did_data(
    n_participants: int = 20,
    n_genes: int = 50,
    n_cells_per_participant: int = 100,
    effect_sizes: dict[str, float] | None = None,
    noise_sd: float = 1.0,
    baseline_mean: float = 5.0,
    participant_sd: float = 0.5,
    time_effect: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate synthetic DiD data with known ground-truth effects.

    Creates pseudobulk-level expression data for a two-arm (Treated vs
    Control), two-timepoint (Pre vs Post) design with participant random
    intercepts and optional DiD interaction effects.

    Parameters
    ----------
    n_participants : int
        Total participants (split equally between arms).
    n_genes : int
        Number of genes to simulate.
    n_cells_per_participant : int
        Cells per participant-visit (used to calibrate noise via sqrt(n)).
    effect_sizes : dict
        Mapping of gene_name -> true DiD effect (beta_DiD).
        Genes not in this dict have effect = 0 (null).
    noise_sd : float
        Residual standard deviation at pseudobulk level.
    baseline_mean : float
        Grand mean expression level.
    participant_sd : float
        Between-participant standard deviation (random intercept).
    time_effect : float
        Main effect of time (Pre->Post shift, same in both arms under null).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        "pseudobulk" : DataFrame [participant, visit, arm, gene_0, ...]
        "truth" : dict mapping gene_name -> true_beta_DiD
        "params" : dict of simulation parameters
    """
    if n_participants % 2 != 0:
        raise ValueError(
            f"n_participants must be even (got {n_participants}); "
            "participants are split equally between arms."
        )
    rng = np.random.default_rng(seed)
    effect_sizes = effect_sizes or {}

    n_per_arm = n_participants // 2
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    # Build truth vector
    truth = {g: effect_sizes.get(g, 0.0) for g in gene_names}

    rows = []
    for arm in ["Control", "Treated"]:
        for p in range(n_per_arm):
            pid = f"{arm[0]}{p}"
            # Participant random intercept (per-gene)
            participant_effect = rng.normal(0, participant_sd, size=n_genes)

            for visit_idx, visit in enumerate(["Pre", "Post"]):
                # Y_ij = baseline + participant_i + time_j + DiD_ij + noise
                y = np.full(n_genes, baseline_mean, dtype=np.float64)
                y += participant_effect
                y += time_effect * visit_idx  # time main effect

                # DiD interaction: only Treated x Post
                if arm == "Treated" and visit == "Post":
                    for gi, g in enumerate(gene_names):
                        y[gi] += truth[g]

                # Residual noise (scaled by sqrt(n_cells) to mimic
                # pseudobulk averaging)
                se = noise_sd / np.sqrt(n_cells_per_participant)
                y += rng.normal(0, se, size=n_genes)

                row = {"participant": pid, "visit": visit, "arm": arm}
                for gi, g in enumerate(gene_names):
                    row[g] = y[gi]
                rows.append(row)

    pb = pd.DataFrame(rows)

    return {
        "pseudobulk": pb,
        "truth": truth,
        "params": {
            "n_participants": n_participants,
            "n_genes": n_genes,
            "n_cells_per_participant": n_cells_per_participant,
            "noise_sd": noise_sd,
            "baseline_mean": baseline_mean,
            "participant_sd": participant_sd,
            "time_effect": time_effect,
            "seed": seed,
        },
    }


def run_method_comparison(
    n_participants: int = 20,
    n_genes: int = 50,
    effect_sizes: dict[str, float] | None = None,
    noise_sd: float = 1.0,
    n_iterations: int = 200,
    methods: list[str] | None = None,
    seed: int = 42,
    n_cells_per_participant: int = 100,
    **sim_kwargs,
) -> pd.DataFrame:
    """Run Monte Carlo comparison of DiD methods on simulated data.

    Parameters
    ----------
    methods : list of str
        Methods to compare. Options: "sctrial_did", "wilcoxon",
        "pseudobulk_ols". Default: all three.
    n_iterations : int
        Number of simulation repetitions.

    Returns
    -------
    DataFrame with columns: iteration, method, gene, true_beta,
        estimated_beta, pvalue, ci_lo, ci_hi
    """
    methods = methods or ["sctrial_did", "wilcoxon", "pseudobulk_ols"]
    rng = np.random.default_rng(seed)
    all_rows = []

    for it in range(n_iterations):
        it_seed = int(rng.integers(0, 2**31))
        sim = simulate_did_data(
            n_participants=n_participants,
            n_genes=n_genes,
            n_cells_per_participant=n_cells_per_participant,
            effect_sizes=effect_sizes,
            noise_sd=noise_sd,
            seed=it_seed,
            **sim_kwargs,
        )
        pb = sim["pseudobulk"]
        truth = sim["truth"]
        gene_cols = [c for c in pb.columns if c.startswith("gene_")]

        for method in methods:
            if method == "sctrial_did":
                results = _run_sctrial_did(pb, gene_cols)
            elif method == "wilcoxon":
                results = _run_wilcoxon(pb, gene_cols)
            elif method == "pseudobulk_ols":
                results = _run_pseudobulk_ols(pb, gene_cols)
            else:
                raise ValueError(f"Unknown method: {method}")

            for g in gene_cols:
                r = results.get(g, {})
                all_rows.append({
                    "iteration": it,
                    "method": method,
                    "gene": g,
                    "true_beta": truth[g],
                    "estimated_beta": r.get("beta", np.nan),
                    "pvalue": r.get("pvalue", np.nan),
                    "ci_lo": r.get("ci_lo", np.nan),
                    "ci_hi": r.get("ci_hi", np.nan),
                })

    return pd.DataFrame(all_rows)


def _run_sctrial_did(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """Run sctrial did_table on pseudobulk DataFrame."""
    import warnings

    import anndata as ad

    from ..design import TrialDesign
    from .did import did_table

    # Build minimal AnnData from pseudobulk
    obs = pb[["participant", "visit", "arm"]].copy().reset_index(drop=True)
    X = pb[gene_cols].values.astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=gene_cols),
    )
    design = TrialDesign(
        participant_col="participant",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_table(
            adata,
            gene_cols,
            design,
            visits=("Pre", "Post"),
            aggregate="participant_visit",
            standardize=False,
        )
    out = {}
    for _, row in res.iterrows():
        out[row["feature"]] = {
            "beta": row["beta_DiD"],
            "pvalue": row["p_DiD"],
            "ci_lo": row.get("ci_lo_DiD", np.nan),
            "ci_hi": row.get("ci_hi_DiD", np.nan),
        }
    return out


def _run_wilcoxon(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """Naive cross-sectional Wilcoxon on post-treatment treated vs control.

    This deliberately ignores pre-treatment data to demonstrate the cost
    of not accounting for baseline differences (participant random intercepts).
    It serves as a negative control: inflated type I error is expected.
    """
    from scipy.stats import mannwhitneyu

    post = pb[pb["visit"] == "Post"]
    treated = post[post["arm"] == "Treated"]
    control = post[post["arm"] == "Control"]

    out = {}
    n_failures = 0
    for g in gene_cols:
        t_vals = treated[g].values
        c_vals = control[g].values
        try:
            _, pval = mannwhitneyu(t_vals, c_vals, alternative="two-sided")
            out[g] = {
                "beta": t_vals.mean() - c_vals.mean(),
                "pvalue": pval,
            }
        except (ValueError, TypeError) as exc:
            logger.debug("Wilcoxon failed for %s: %s", g, exc)
            n_failures += 1
            out[g] = {}
    if n_failures:
        logger.warning("Wilcoxon: %d/%d genes failed", n_failures, len(gene_cols))
    return out


def _run_pseudobulk_ols(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """OLS DiD without participant fixed effects.

    Intentionally omits participant FE to demonstrate the anti-conservative
    bias of naive OLS when participant random intercepts are present.
    CIs and p-values will be too narrow/small because residual correlation
    within participants inflates effective sample size.
    """
    import statsmodels.formula.api as smf

    pb = pb.copy()
    pb["arm_bin"] = (pb["arm"] == "Treated").astype(float)
    pb["visit_bin"] = (pb["visit"] == "Post").astype(float)
    pb["interaction"] = pb["arm_bin"] * pb["visit_bin"]

    out = {}
    n_failures = 0
    for g in gene_cols:
        try:
            # Backtick-quote gene names for formula safety
            fit = smf.ols(f"Q('{g}') ~ arm_bin + visit_bin + interaction",
                          data=pb).fit()
            beta = fit.params["interaction"]
            pval = fit.pvalues["interaction"]
            ci = fit.conf_int().loc["interaction"]
            out[g] = {
                "beta": beta,
                "pvalue": pval,
                "ci_lo": ci.iloc[0],
                "ci_hi": ci.iloc[1],
            }
        except (ValueError, np.linalg.LinAlgError, KeyError) as exc:
            logger.debug("Pseudobulk OLS failed for %s: %s", g, exc)
            n_failures += 1
            out[g] = {}
    if n_failures:
        logger.warning(
            "Pseudobulk OLS: %d/%d genes failed", n_failures, len(gene_cols)
        )
    return out
