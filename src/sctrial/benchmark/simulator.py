"""Hierarchical gamma-Poisson simulator for scRNA-seq clinical trial data.

Generates realistic cell-level count data for benchmarking, inspired by
rescueSim's generative framework (Crowell et al., 2020) but implemented
natively in Python for computational efficiency.

The simulator produces data within a single cell type — matching how
edgeR, dreamlet, NEBULA, and sctrial are actually applied in practice.

Design families
---------------
- **Two-arm longitudinal**: Treated/Control × Pre/Post, β₂ = interaction
- **Single-arm paired**: All participants same arm, Pre/Post, β₁ = time

Generative model (per gene g, participant i, visit j, cell k)::

    Library size:         L_ijk ~ LogNormal(log(target_lib), σ_lib)
    Participant RE:       α_ig  ~ N(0, σ²_participant)
    Log-mean:             log(μ_igk) = β₀g + α_ig + β₁·Post_j
                                       + β₂·(Treat_i × Post_j) + log(L_ijk)
    Overdispersion:       Y_igk ~ NegBin(μ_igk, θ_g)
    Cell counts:          n_cells_ij ~ Poisson(λ) or LogNormal (imbalanced)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run.

    Parameters
    ----------
    design : {"two_arm", "single_arm"}
        Trial design family.
    n_per_arm : int
        Participants per arm. For two-arm designs with equal allocation,
        total participants = 2 × n_per_arm. For single-arm, this IS the
        total participant count.
    n_genes : int
        Total genes to simulate.
    effects : dict[str, float]
        Gene-name → true effect size. Genes not listed have β₂=0 (null).
    mean_cells_per_visit : int
        Average cells per participant-visit.
    cell_count_mode : {"poisson", "lognormal"}
        How cell counts vary across participant-visits.
    cell_count_cv : float
        Coefficient of variation for lognormal cell counts (ignored if poisson).
    arm_ratio : tuple[int, int] | None
        Ratio of treated:control (e.g., (3, 7) for unbalanced). Only for two-arm.
        If None, uses equal allocation.
    missing_rate : float
        Fraction of participants missing their Post visit (0.0 = no attrition).
    dispersion_mode : {"calibrated", "fixed", "extreme"}
        How gene-specific θ_g is drawn.
    dispersion_fixed : float
        Fixed θ for all genes (only if dispersion_mode="fixed").
    participant_sd : float
        SD of participant random intercept (σ_participant).
    baseline_mean : float
        Mean of β₀g across genes (log-scale).
    baseline_sd : float
        SD of β₀g across genes.
    time_effect : float
        Global time effect β₁ (applied to all genes).
    target_library_size : int
        Target library size per cell.
    library_size_sd : float
        SD of log-library size.
    seed : int
        Random seed.
    """

    design: Literal["two_arm", "single_arm"] = "two_arm"
    n_per_arm: int = 20
    n_genes: int = 50
    effects: dict[str, float] = field(default_factory=dict)
    mean_cells_per_visit: int = 500
    cell_count_mode: Literal["poisson", "lognormal"] = "poisson"
    cell_count_cv: float = 0.5
    arm_ratio: tuple[int, int] | None = None
    missing_rate: float = 0.0
    dispersion_mode: Literal["calibrated", "fixed", "extreme"] = "calibrated"
    dispersion_fixed: float = 10.0
    participant_sd: float = 0.3
    baseline_mean: float = 2.0
    baseline_sd: float = 1.0
    time_effect: float = 0.1
    target_library_size: int = 10000
    library_size_sd: float = 0.3
    seed: int = 42


def simulate_trial(cfg: SimulationConfig) -> dict:
    """Generate a complete simulated trial dataset.

    Returns
    -------
    dict with keys:
        "adata"           : AnnData — cell-level counts (X = raw counts, obs has metadata)
        "pseudobulk_means": DataFrame — participant-visit mean expression (for sctrial/Wilcoxon)
        "pseudobulk_counts": DataFrame — participant-visit summed counts (for edgeR/limma/dreamlet)
        "truth"           : dict[str, float] — gene → true interaction effect
        "config"          : SimulationConfig
    """
    rng = np.random.default_rng(cfg.seed)

    # --- Participant structure ---
    # n_per_arm = participants PER ARM (not total)
    if cfg.design == "two_arm":
        if cfg.arm_ratio is not None:
            # arm_ratio is e.g. (3, 7) meaning 3 treated : 7 control
            n_treat = cfg.arm_ratio[0]
            n_ctrl = cfg.arm_ratio[1]
        else:
            n_treat = cfg.n_per_arm
            n_ctrl = cfg.n_per_arm
        arms = ["Treated"] * n_treat + ["Control"] * n_ctrl
    else:
        n_treat = cfg.n_per_arm
        n_ctrl = 0
        arms = ["Treated"] * n_treat

    n_total = len(arms)
    participant_ids = [f"P{i:03d}" for i in range(n_total)]
    visits = ["Pre", "Post"]

    # --- Gene parameters ---
    gene_names = [f"gene_{i}" for i in range(cfg.n_genes)]

    # Baseline expression: log-rate = log(gene_mean / library_size)
    # In real scRNA-seq, this spans ~5 orders of magnitude (most genes barely
    # detected, a few highly expressed). baseline_mean and baseline_sd define
    # the distribution of log-rates BEFORE the library-size offset is added.
    # Typical values from TNBC: mean=-12.4, sd=2.7 (on log scale).
    beta0 = rng.normal(cfg.baseline_mean, cfg.baseline_sd, size=cfg.n_genes)
    # No clipping — extreme low values produce sparse genes (realistic)

    # Gene-specific dispersion θ_g
    # Real scRNA-seq has very low θ (high overdispersion): median ~0.1-0.3
    theta: np.ndarray
    if cfg.dispersion_mode == "fixed":
        theta = np.full(cfg.n_genes, cfg.dispersion_fixed)
    elif cfg.dispersion_mode == "extreme":
        # Very low dispersion = extremely high variance
        theta = np.asarray(rng.uniform(0.01, 0.5, size=cfg.n_genes))
    else:
        # Calibrated: match real scRNA-seq (θ typically 0.01-2.0, median ~0.15)
        theta = np.asarray(rng.lognormal(np.log(0.15), 1.0, size=cfg.n_genes))
        theta = np.asarray(np.clip(theta, 0.01, 50.0))

    # True effects
    true_effects = {}
    for g, name in enumerate(gene_names):
        true_effects[name] = cfg.effects.get(name, 0.0)

    # Participant random intercepts (per gene)
    alpha = rng.normal(0, cfg.participant_sd, size=(n_total, cfg.n_genes))

    # --- Determine which participant-visits exist (attrition) ---
    missing_post = set()
    if cfg.missing_rate > 0:
        n_missing = max(1, int(n_total * cfg.missing_rate))
        missing_post = set(rng.choice(n_total, size=n_missing, replace=False))

    # --- Generate cells ---
    all_obs = []
    all_counts = []

    for i, (pid, arm) in enumerate(zip(participant_ids, arms)):
        for visit in visits:
            if visit == "Post" and i in missing_post:
                continue

            # Cell count for this participant-visit
            if cfg.cell_count_mode == "lognormal":
                mu_cells = cfg.mean_cells_per_visit
                sigma_cells = np.sqrt(np.log(1 + cfg.cell_count_cv**2))
                n_cells = int(
                    rng.lognormal(
                        np.log(mu_cells) - sigma_cells**2 / 2,
                        sigma_cells,
                    )
                )
            else:
                n_cells = rng.poisson(cfg.mean_cells_per_visit)

            n_cells = max(10, min(n_cells, 20000))  # safety bounds

            # Indicators
            is_post = 1.0 if visit == "Post" else 0.0
            is_treated = 1.0 if arm == "Treated" else 0.0

            # Library sizes for each cell
            log_lib = rng.normal(
                np.log(cfg.target_library_size),
                cfg.library_size_sd,
                size=n_cells,
            )
            lib_sizes = np.exp(log_lib)

            # Gene expression: log(μ) = β₀ + α_i + β₁·Post + β₂·(Treat×Post) + log(L)
            # Shape: (n_cells, n_genes)
            log_mu = (
                beta0[np.newaxis, :]  # (1, G)
                + alpha[i, :][np.newaxis, :]  # (1, G)
                + cfg.time_effect * is_post  # scalar
                + log_lib[:, np.newaxis]  # (C, 1)
            )

            # Add treatment × post interaction for signal genes
            for g, name in enumerate(gene_names):
                effect = true_effects[name]
                if effect != 0.0:
                    log_mu[:, g] += effect * is_post * is_treated

            mu = np.exp(log_mu)

            # Draw counts via Gamma-Poisson (equivalent to NegBin but handles
            # non-integer theta, which is essential for realistic scRNA-seq
            # where theta can be as low as 0.01)
            #
            # Step 1: rate ~ Gamma(shape=theta, scale=mu/theta)
            # Step 2: count ~ Poisson(rate)
            #
            # This is the standard compound distribution for NB counts.
            theta_broad = theta[np.newaxis, :]  # (1, G)
            rate = rng.gamma(
                shape=theta_broad,
                scale=mu / theta_broad,
            )
            counts = rng.poisson(rate).astype(np.int32)

            # Store obs metadata
            for k in range(n_cells):
                all_obs.append(
                    {
                        "participant": pid,
                        "arm": arm,
                        "visit": visit,
                        "library_size": lib_sizes[k],
                    }
                )
            all_counts.append(counts)

    # --- Assemble AnnData ---
    obs_df = pd.DataFrame(all_obs)
    X = np.vstack(all_counts)

    adata = ad.AnnData(
        X=sparse.csr_matrix(X),
        obs=obs_df,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]

    # --- Pseudobulk aggregation ---
    # Two outputs: means (for sctrial/Wilcoxon) and summed counts (for edgeR/limma/dreamlet)
    pb_mean_rows = []
    pb_count_rows = []
    for (pid, visit), grp in obs_df.groupby(["participant", "visit"]):
        idx = grp.index.values
        means = X[idx].mean(axis=0)
        sums = X[idx].sum(axis=0)
        meta = {"participant": pid, "visit": visit, "arm": grp["arm"].iloc[0]}
        mean_row = dict(meta)
        count_row = dict(meta)
        for g, name in enumerate(gene_names):
            mean_row[name] = means[g]
            count_row[name] = int(sums[g])
        pb_mean_rows.append(mean_row)
        pb_count_rows.append(count_row)
    pseudobulk_means = pd.DataFrame(pb_mean_rows)
    pseudobulk_counts = pd.DataFrame(pb_count_rows)

    return {
        "adata": adata,
        "pseudobulk_means": pseudobulk_means,
        "pseudobulk_counts": pseudobulk_counts,
        "truth": true_effects,
        "config": cfg,
    }


def calibrate_from_real_data(
    adata_real,
    layer: str | None = None,
    count_layer: str | None = None,
    participant_col: str = "participant",
    visit_col: str = "visit",
) -> dict:
    """Extract distributional parameters from real scRNA-seq data.

    Uses raw counts for dispersion/library-size calibration (critical for
    the NB generative model), and normalized expression for ICC estimation.

    Parameters
    ----------
    adata_real : AnnData
        Real dataset.
    layer : str, optional
        Normalized expression layer (for ICC). If None, uses .X.
    count_layer : str, optional
        Raw count layer for dispersion and library size calibration.
        If None, auto-detects: checks for integer .X, then "counts" layer.
        If no raw counts available, estimates from normalized data.
    participant_col : str
        Column name for participant IDs.
    visit_col : str
        Column name for visit/timepoint.

    Returns
    -------
    dict with keys:
        "mean_cells_per_visit" : float
        "cell_count_cv" : float
        "participant_icc" : float  (median across genes)
        "dispersion_median" : float
        "baseline_mean" : float  (log-scale, on raw counts)
        "baseline_sd" : float
        "library_size_mean" : float  (log-scale, on raw counts)
        "library_size_sd" : float
        "has_raw_counts" : bool
    """
    import warnings

    obs = adata_real.obs

    # --- Find raw counts ---
    has_raw_counts = False
    X_counts = None

    if count_layer is not None:
        X_counts = adata_real.layers[count_layer]
        has_raw_counts = True
    else:
        # Auto-detect: check if .X is integer counts
        X_test = adata_real.X
        if sparse.issparse(X_test):
            X_test_dense = X_test[:100].toarray()
        else:
            X_test_dense = X_test[:100]
        if np.allclose(X_test_dense, X_test_dense.astype(int)):
            X_counts = adata_real.X
            has_raw_counts = True
        elif "counts" in (adata_real.layers or {}):
            X_counts = adata_real.layers["counts"]
            has_raw_counts = True

    if X_counts is not None:
        if sparse.issparse(X_counts):
            X_counts = X_counts.toarray()
        X_counts = np.asarray(X_counts, dtype=float)

    # --- Normalized expression (for ICC) ---
    X_norm = adata_real.layers[layer] if layer else adata_real.X
    if sparse.issparse(X_norm):
        X_norm = X_norm.toarray()
    X_norm = np.asarray(X_norm, dtype=float)

    # --- Cell counts per participant-visit ---
    cell_counts = obs.groupby([participant_col, visit_col]).size()
    mean_cells = cell_counts.mean()
    cv_cells = cell_counts.std() / mean_cells if mean_cells > 0 else 0.5

    # --- Count-based statistics ---
    if has_raw_counts and X_counts is not None:
        # Library sizes from raw counts
        lib_sizes = X_counts.sum(axis=1)
        log_lib = np.log(lib_sizes + 1)
        avg_lib = lib_sizes.mean()

        # Gene-level stats on counts
        gene_means = X_counts.mean(axis=0)
        gene_vars = X_counts.var(axis=0)

        # Log-rate: log(gene_mean / avg_library_size)
        # This is the baseline parameter β₀g in the generative model
        # BEFORE library-size offset. Spans ~5 orders of magnitude.
        mask_expressed = gene_means > 0
        log_rates = np.log(gene_means / avg_lib + 1e-10)
        log_rates_expr = log_rates[mask_expressed]
        log_means = log_rates_expr  # used for baseline_mean/sd below

        # Dispersion: method of moments on counts
        # Var = mu + mu^2/theta => theta = mu^2 / (var - mu)
        # Real scRNA-seq has very low theta (0.01-2.0), do NOT clip at 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denom = gene_vars - gene_means
            denom = np.where(denom > 0, denom, 1e-6)
            theta_est = gene_means**2 / denom
            theta_est = np.clip(theta_est, 0.01, 1000)
    else:
        # No raw counts — estimate from normalized data
        # Use typical scRNA-seq defaults calibrated to the sparsity pattern
        zero_frac = (X_norm == 0).mean()
        # Typical library size for scRNA-seq: ~2000-10000
        # Estimate from expression scale
        lib_sizes_norm = X_norm.sum(axis=1)
        log_lib = np.log(np.clip(lib_sizes_norm, 1, None))

        gene_means = X_norm.mean(axis=0)
        gene_vars = X_norm.var(axis=0)
        log_means = np.log(gene_means + 1)

        # For normalized data, dispersion estimate is less reliable
        # Use conservative defaults based on zero fraction
        # More zeros → lower dispersion (higher overdispersion)
        theta_est = np.full_like(gene_means, 5.0)  # moderate default
        if zero_frac > 0.9:
            theta_est[:] = 2.0  # high sparsity → high overdispersion

    # --- ICC from normalized expression ---
    iccs = []
    n_genes_sample = min(50, adata_real.n_vars)
    gene_idx = np.random.choice(adata_real.n_vars, n_genes_sample, replace=False)
    for g in gene_idx:
        expr = X_norm[:, g]
        groups = obs[participant_col].values
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3:
            continue
        grand_mean = expr.mean()
        ss_between = sum(
            np.sum(groups == grp) * (expr[groups == grp].mean() - grand_mean) ** 2
            for grp in unique_groups
        )
        ss_total = np.sum((expr - grand_mean) ** 2)
        if ss_total > 0:
            iccs.append(ss_between / ss_total)

    return {
        "mean_cells_per_visit": float(mean_cells),
        "cell_count_cv": float(cv_cells),
        "participant_icc": float(np.median(iccs)) if iccs else 0.1,
        "dispersion_median": float(np.median(theta_est)),
        "baseline_mean": float(log_means.mean()),
        "baseline_sd": float(log_means.std()),
        "library_size_mean": float(log_lib.mean()),
        "library_size_sd": float(log_lib.std()),
        "has_raw_counts": has_raw_counts,
    }


def validate_simulator(
    cfg: SimulationConfig,
    adata_real,
    layer: str | None = None,
    participant_col: str = "participant",
    visit_col: str = "visit",
) -> dict:
    """Compare simulated data properties against real data.

    Compares on a common scale: if real data is normalized (no raw counts),
    the simulated raw counts are log1p-CPM-normalized before comparison.
    If real data has raw counts, compares counts directly.

    Returns
    -------
    dict with keys:
        "cell_level" : dict — mean-variance, zero fraction, library size
        "pseudobulk_level" : dict — ICC, cell-count distribution, mean distribution
    """
    sim = simulate_trial(cfg)
    sim_adata = sim["adata"]
    sim_X_raw = sim_adata.X.toarray() if sparse.issparse(sim_adata.X) else sim_adata.X

    real_X = adata_real.layers[layer] if layer else adata_real.X
    if sparse.issparse(real_X):
        real_X = real_X.toarray()
    real_X = np.asarray(real_X, dtype=float)

    # Check if real data is normalized (non-integer) — if so, normalize sim too
    real_is_counts = np.allclose(real_X[:100], real_X[:100].astype(int))
    if real_is_counts:
        sim_X = sim_X_raw.astype(float)
    else:
        # Normalize simulated counts to log1p-CPM to match real normalized data
        lib = sim_X_raw.sum(axis=1, keepdims=True).astype(float)
        lib = np.where(lib > 0, lib, 1.0)
        sim_X = np.log1p(sim_X_raw / lib * 1e6)

    # --- Cell-level comparisons ---
    # Mean-variance
    sim_means = sim_X.mean(axis=0)
    sim_vars = sim_X.var(axis=0)
    real_means = real_X.mean(axis=0)
    real_vars = real_X.var(axis=0)

    # Zero fraction per gene
    sim_zeros = (sim_X == 0).mean(axis=0)
    real_zeros = (real_X == 0).mean(axis=0)

    # Library size: compare on MATCHED gene count
    # When simulator uses fewer genes than the real dataset (e.g. 50 vs 20k),
    # whole-transcriptome library sizes are not comparable. Instead, compare
    # total counts across the SAME number of genes.
    n_sim_genes = sim_X.shape[1]
    n_real_genes = real_X.shape[1]
    if n_sim_genes < n_real_genes:
        # Subsample real data to same number of genes (pick random subset
        # with similar mean expression distribution)
        real_gene_means_sorted = np.argsort(real_X.mean(axis=0))
        # Sample evenly across the expression range
        idx = np.linspace(0, n_real_genes - 1, n_sim_genes, dtype=int)
        real_X_subset = real_X[:, real_gene_means_sorted[idx]]
        real_lib = real_X_subset.sum(axis=1)
    else:
        real_lib = real_X.sum(axis=1)
    sim_lib = sim_X.sum(axis=1)

    # --- Pseudobulk-level comparisons ---
    # Cell counts per participant-visit
    sim_cell_counts = sim_adata.obs.groupby(["participant", "visit"]).size().values
    real_cell_counts = adata_real.obs.groupby([participant_col, visit_col]).size().values

    # Pseudobulk means
    sim_pb = sim["pseudobulk_means"]
    gene_cols = [c for c in sim_pb.columns if c.startswith("gene_")]
    sim_pb_means = sim_pb[gene_cols].values.flatten()

    return {
        "cell_level": {
            "sim_gene_means": sim_means,
            "sim_gene_vars": sim_vars,
            "real_gene_means": real_means,
            "real_gene_vars": real_vars,
            "sim_zero_fraction": sim_zeros,
            "real_zero_fraction": real_zeros,
            "sim_library_sizes": sim_lib,
            "real_library_sizes": real_lib,
        },
        "pseudobulk_level": {
            "sim_cell_counts": sim_cell_counts,
            "real_cell_counts": real_cell_counts,
            "sim_pb_means": sim_pb_means,
        },
    }
