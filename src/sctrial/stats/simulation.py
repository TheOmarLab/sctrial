"""Monte Carlo simulation engine for DiD method comparison.

Generates synthetic **cell-level** scRNA-seq-like data with known ground-truth
DiD effects, enabling controlled comparison of statistical methods under
realistic conditions (variable cell counts, participant heterogeneity,
cell-level noise).
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
    """Generate synthetic cell-level DiD data with known ground-truth effects.

    Creates cell-level expression data for a two-arm (Treated vs Control),
    two-timepoint (Pre vs Post) design with participant random intercepts,
    variable cell counts per participant-visit, and cell-level noise.

    The data-generating process mirrors real scRNA-seq clinical trial data:

    .. math::

        Y_{ijk} = \\mu + \\alpha_i + \\beta_1 \\text{Post}_j
                  + \\beta_2 (\\text{Treat}_i \\times \\text{Post}_j)
                  + \\epsilon_{ijk}

    where *i* indexes participants, *j* indexes visits, *k* indexes cells,
    :math:`\\alpha_i \\sim N(0, \\sigma_{\\text{participant}}^2)`, and
    :math:`\\epsilon_{ijk} \\sim N(0, \\sigma_{\\text{noise}}^2)`.

    Parameters
    ----------
    n_participants : int
        Total participants (split equally between arms).
    n_genes : int
        Number of genes to simulate.
    n_cells_per_participant : int
        Mean cells per participant-visit.  Actual counts are drawn from
        Poisson(n_cells_per_participant) with a floor of 20.
    effect_sizes : dict
        Mapping of gene_name -> true DiD effect (beta_DiD).
        Genes not in this dict have effect = 0 (null).
    noise_sd : float
        Cell-level residual standard deviation.
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
        "adata" : AnnData  — cell-level expression matrix with obs columns
            ``participant``, ``visit``, ``arm``
        "pseudobulk" : DataFrame  — participant-visit means with ``n_cells``
        "truth" : dict mapping gene_name -> true_beta_DiD
        "params" : dict of simulation parameters
    """
    import anndata as ad

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

    obs_rows: list[dict] = []
    X_rows: list[np.ndarray] = []
    pb_rows: list[dict] = []  # pseudobulk aggregation

    for arm in ["Control", "Treated"]:
        for p in range(n_per_arm):
            pid = f"{arm[0]}{p}"
            # Participant random intercept (per-gene)
            participant_effect = rng.normal(0, participant_sd, size=n_genes)

            for visit_idx, visit in enumerate(["Pre", "Post"]):
                # Variable cell count (Poisson with floor)
                n_cells = max(20, rng.poisson(n_cells_per_participant))

                # Per-cell expression
                cell_X = np.empty((n_cells, n_genes), dtype=np.float64)
                for c in range(n_cells):
                    y = np.full(n_genes, baseline_mean, dtype=np.float64)
                    y += participant_effect
                    y += time_effect * visit_idx

                    # DiD interaction: only Treated x Post
                    if arm == "Treated" and visit == "Post":
                        for gi, g in enumerate(gene_names):
                            y[gi] += truth[g]

                    # Cell-level noise
                    y += rng.normal(0, noise_sd, size=n_genes)
                    cell_X[c] = y

                # Store cell-level data
                for c in range(n_cells):
                    obs_rows.append(
                        {
                            "participant": pid,
                            "visit": visit,
                            "arm": arm,
                        }
                    )
                X_rows.append(cell_X)

                # Pre-compute pseudobulk (mean per participant-visit)
                pb_mean = cell_X.mean(axis=0)
                pb_row = {
                    "participant": pid,
                    "visit": visit,
                    "arm": arm,
                    "n_cells": n_cells,
                }
                for gi, g in enumerate(gene_names):
                    pb_row[g] = pb_mean[gi]
                pb_rows.append(pb_row)

    # Build AnnData
    obs = pd.DataFrame(obs_rows)
    X = np.vstack(X_rows).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=obs.reset_index(drop=True),
        var=pd.DataFrame(index=gene_names),
    )

    pb = pd.DataFrame(pb_rows)

    return {
        "adata": adata,
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


def _run_single_iteration(args: tuple) -> list[dict]:
    """Run all methods on a single simulated dataset (for parallel dispatch)."""
    (
        it,
        it_seed,
        n_participants,
        n_genes,
        n_cells_per_participant,
        effect_sizes,
        noise_sd,
        methods,
        sim_kwargs,
    ) = args

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _run_single_iteration_inner(
            it,
            it_seed,
            n_participants,
            n_genes,
            n_cells_per_participant,
            effect_sizes,
            noise_sd,
            methods,
            sim_kwargs,
        )


def _run_single_iteration_inner(
    it,
    it_seed,
    n_participants,
    n_genes,
    n_cells_per_participant,
    effect_sizes,
    noise_sd,
    methods,
    sim_kwargs,
) -> list[dict]:
    """Inner implementation — called inside a ``catch_warnings`` context."""
    sim = simulate_did_data(
        n_participants=n_participants,
        n_genes=n_genes,
        n_cells_per_participant=n_cells_per_participant,
        effect_sizes=effect_sizes,
        noise_sd=noise_sd,
        seed=it_seed,
        **sim_kwargs,
    )
    adata = sim["adata"]
    pb = sim["pseudobulk"]
    truth = sim["truth"]
    gene_cols = [c for c in pb.columns if c.startswith("gene_")]

    rows = []
    for method in methods:
        if method == "sctrial_did":
            results = _run_sctrial_did(adata, gene_cols)
        elif method == "mixed_did":
            results = _run_mixed_did(adata, gene_cols)
        elif method == "wilcoxon":
            results = _run_wilcoxon(pb, gene_cols)
        elif method == "pseudobulk_ols":
            results = _run_pseudobulk_ols(pb, gene_cols)
        else:
            raise ValueError(f"Unknown method: {method}")

        for g in gene_cols:
            r = results.get(g, {})
            rows.append(
                {
                    "iteration": it,
                    "method": method,
                    "gene": g,
                    "true_beta": truth[g],
                    "estimated_beta": r.get("beta", np.nan),
                    "pvalue": r.get("pvalue", np.nan),
                    "ci_lo": r.get("ci_lo", np.nan),
                    "ci_hi": r.get("ci_hi", np.nan),
                }
            )
    return rows


def run_method_comparison(
    n_participants: int = 20,
    n_genes: int = 50,
    effect_sizes: dict[str, float] | None = None,
    noise_sd: float = 1.0,
    n_iterations: int = 200,
    methods: list[str] | None = None,
    seed: int = 42,
    n_cells_per_participant: int = 100,
    n_jobs: int | None = None,
    **sim_kwargs,
) -> pd.DataFrame:
    """Run Monte Carlo comparison of DiD methods on simulated data.

    Generates cell-level scRNA-seq data with known ground truth and runs
    each method on the same datasets.  Methods that operate on cell-level
    data (sctrial DiD, mixed DiD) receive the full AnnData so their
    internal aggregation + cluster-robust SE pipeline is exercised
    faithfully.  Methods that expect pseudobulk (OLS, Wilcoxon) receive
    the pre-aggregated participant-visit means.

    Iterations are parallelised across CPU cores for speed.

    Parameters
    ----------
    methods : list of str
        Methods to compare. Options: ``"sctrial_did"``, ``"mixed_did"``,
        ``"pseudobulk_ols"``, ``"wilcoxon"``.  Default: all four.
    n_iterations : int
        Number of simulation repetitions.
    n_jobs : int, optional
        Number of parallel workers.  Defaults to ``min(cpu_count, 8)``.
        Set to 1 to disable parallelisation.

    Returns
    -------
    DataFrame with columns: iteration, method, gene, true_beta,
        estimated_beta, pvalue, ci_lo, ci_hi
    """
    import multiprocessing as mp

    methods = methods or ["sctrial_did", "mixed_did", "pseudobulk_ols", "wilcoxon"]
    rng = np.random.default_rng(seed)

    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), 8)

    # Pre-generate seeds for reproducibility
    it_seeds = [int(rng.integers(0, 2**31)) for _ in range(n_iterations)]

    task_args = [
        (
            it,
            it_seeds[it],
            n_participants,
            n_genes,
            n_cells_per_participant,
            effect_sizes,
            noise_sd,
            methods,
            sim_kwargs,
        )
        for it in range(n_iterations)
    ]

    if n_jobs == 1:
        # Sequential fallback
        all_rows = []
        for args in task_args:
            all_rows.extend(_run_single_iteration(args))
    else:
        all_rows = []
        with mp.Pool(n_jobs) as pool:
            for batch in pool.imap(_run_single_iteration, task_args):
                all_rows.extend(batch)
                # Progress callback (if caller wraps with tqdm etc.)
                done = len(all_rows) // (len(methods) * n_genes)
                if done % 10 == 0:
                    logger.info("  %d/%d iterations complete", done, n_iterations)

    return pd.DataFrame(all_rows)


def _run_sctrial_did(adata, gene_cols: list[str]) -> dict:
    """Run sctrial did_table on cell-level AnnData.

    Mirrors the real user workflow: cell-level data → did_table with
    ``aggregate="cell"`` so that cluster-robust standard errors have
    many observations per cluster (participant), producing reliable
    inference.
    """
    import warnings

    from ..design import TrialDesign
    from .did import did_table

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
            aggregate="cell",
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


def _run_mixed_did(adata, gene_cols: list[str]) -> dict:
    """Run mixed-effects DiD with participant as random intercept.

    Receives cell-level AnnData.  Uses ``aggregate="participant_visit"``
    because the mixed model's random intercept absorbs participant
    heterogeneity and only needs pseudobulk-level data.
    """
    import warnings

    from ..design import TrialDesign
    from .mixed_effects import did_table_mixed

    design = TrialDesign(
        participant_col="participant",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = did_table_mixed(
            adata,
            gene_cols,
            design,
            visits=("Pre", "Post"),
            aggregate="participant_visit",
            standardize=False,
        )
    out = {}
    for _, row in res.iterrows():
        # Skip non-converged fits — their coefficients are unreliable
        if not row.get("converged", True):
            logger.debug("Mixed DiD did not converge for %s", row["feature"])
            continue
        out[row["feature"]] = {
            "beta": row["beta_DiD"],
            "pvalue": row["p_DiD"],
            "ci_lo": row.get("ci_lower", np.nan),
            "ci_hi": row.get("ci_upper", np.nan),
        }
    return out


def _run_wilcoxon(pb: pd.DataFrame, gene_cols: list[str]) -> dict:
    """Naive cross-sectional Wilcoxon on post-treatment treated vs control.

    This deliberately ignores pre-treatment data to demonstrate the cost
    of not accounting for baseline differences (participant random intercepts).
    It serves as a comparator that discards longitudinal information.
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

    Intentionally omits participant FE.  Without accounting for within-
    participant correlation the resulting standard errors may be mis-sized
    (conservative or anti-conservative depending on the variance structure).
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
            fit = smf.ols(f"Q('{g}') ~ arm_bin + visit_bin + interaction", data=pb).fit()
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
        logger.warning("Pseudobulk OLS: %d/%d genes failed", n_failures, len(gene_cols))
    return out
