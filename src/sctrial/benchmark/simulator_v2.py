"""Transcriptome-scale hierarchical simulator for the sctrial benchmark.

Replaces the panel-only generative model, which had three structural defects that
each corrupted published benchmark numbers:

1. **No transcriptome.** Only the tested panel (50-2000 genes) was simulated, so a
   library size could only be formed by summing the panel. That sum moves with the
   signal genes, so CPM "normalisation" partly divides out the effect being
   measured. Measured consequence: +25% overshoot under a one-directional design.
   The same defect is why NEBULA's offset (``log(colSums(panel))``) was wrong.
2. **No participant x visit level.** A single participant intercept cancels exactly
   in the difference-in-differences contrast, so ``participant_sd`` was inert and
   the repeated-measures premise had no simulated counterpart.
3. **Uncalibrated scale.** Defaults produced ~2.3e7 UMIs per cell against a TNBC
   median of 2,113, and effect recovery only worked *because* of that density
   (log1p(mu) ~ log(mu) requires mu >> 1). At realistic depth the same code
   attenuated a nominal 0.5 to 0.047.

Generative model, per cell ``c`` of participant ``i`` at visit ``t``, gene ``g``::

    log mu_icgt = log L_ic + alpha_g + b_ig + u_igt + gamma_g * Post_t
                  + beta_g * (T_i * Post_t)
    Y_icgt ~ NegBinomial(mu_icgt, phi_g)          var = mu + phi_g * mu^2

with

* ``L_ic``    per-cell library size, lognormal, calibrated to the empirical
              TNBC distribution;
* ``alpha_g`` baseline log-rate relative to library size;
* ``b_ig``    participant random effect (level 1);
* ``u_igt``   participant x visit deviation (level 2) - sets the pre/post
              correlation, and is what makes the DiD contrast non-degenerate;
* ``gamma_g`` common time effect (cancels in DiD, present so single-arm designs
              are not trivially null);
* ``beta_g``  the treatment x time interaction: the estimand.

The full transcriptome is simulated; analysis panels are drawn from it as **nested**
subsets so panel-size effects are separable from gene-identity effects, and
normalisation uses the whole transcriptome exactly as a real workflow would.

``simulate_trial_v2`` returns the observed data a real analysis would have
(cell-level counts, full-transcriptome pseudobulk) plus the latent quantities
(``true_library_size``, ``b_ig``, ``u_igt``) which are for **validation only** and
must never be used as an analysis input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "TranscriptomeSimConfig",
    "simulate_trial_v2",
    "load_tnbc_targets",
    "make_signal",
    "nested_panels",
]

SignalArch = Literal["balanced", "heterogeneous", "one_directional"]


def load_tnbc_targets(path: str | Path | None = None) -> dict:
    """Empirical TNBC targets measured from the processed h5ad.

    These are the quantities a defensible calibration must reproduce. Measured by
    ``scripts/regen/tnbc_targets.py`` from the v5 TNBC object (141,553 cells x
    20,284 genes, 12 paired participants, 6 v 6 arms).
    """
    if path is None:
        path = (
            Path(__file__).resolve().parents[3]
            / "manuscript" / "benchmark" / "validation" / "tnbc_sim_targets.json"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TNBC simulation targets not found at {path}. Run "
            "scripts/regen/tnbc_targets.py first — the simulator must not fall back "
            "to uncalibrated defaults (that is how the previous benchmark produced "
            "2.3e7 UMIs per cell)."
        )
    with open(path) as fh:
        return json.load(fh)


@dataclass
class TranscriptomeSimConfig:
    """Configuration for the transcriptome-scale simulator.

    Defaults are the TNBC-calibrated values. Every one is an empirical quantity, so
    a run with defaults is a calibrated run — the previous design's defaults were
    arbitrary and silently used whenever calibration was not threaded through.
    """

    # --- design ---
    n_per_arm: int = 6
    design: Literal["two_arm", "single_arm"] = "two_arm"
    n_genes_transcriptome: int = 12000
    arm_ratio: tuple[int, int] | None = None

    # --- cells per participant-visit (distribution, not a fixed number) ---
    cells_per_pv_mean: float = 5898.0
    cells_per_pv_cv: float = 0.911
    cells_per_pv_min: int = 147
    cells_per_pv_max: int = 27653
    cells_scale: float = 1.0  # shrink for tractable benchmarking; 1.0 = TNBC scale

    # --- library size (per cell), lognormal on the log scale ---
    lib_log_mean: float = 7.7333
    lib_log_sd: float = 0.8000

    # --- baseline gene rates: alpha_g ~ N(mean, sd), relative to library size ---
    gene_rate_log_mean: float = -4.5338
    gene_rate_log_sd: float = 2.6428

    # --- NB dispersion: var = mu + phi * mu^2, phi_g lognormal ---
    # PARAMETERISATION (stated explicitly - this is a documented trap):
    #   THIS SIMULATOR uses NB2:            Var(Y) = mu + phi * mu^2
    #     implemented as gamma-Poisson with Gamma(shape=1/phi, scale=mu*phi),
    #     so `dispersion_median` IS the NB2 alpha.
    #   NEBULA parameterises the cell-level term as mu^2/phi_nebula, i.e. its
    #     "dispersion" is the RECIPROCAL of this one. Do not pass one for the other.
    #   The naive estimator (Var(Y) - E(Y)) / E(Y)^2 computed ACROSS ALL CELLS does
    #     NOT estimate this parameter once hierarchical terms exist: it absorbs
    #     b_ig, u_igt and library-size heterogeneity. Calibration therefore targets
    #     the CONDITIONAL cell-level dispersion (within participant-visit, offset by
    #     library size), and the simulator is validated on the observable
    #     mean-variance RELATIONSHIP rather than on a single latent scalar.
    # Conditional cell-level NB2 alpha measured WITHIN participant-visit with a
    # library-size offset (scripts/regen/disp_cond.py): 0.788. The marginal
    # estimator across all cells gives 2.837 because it also absorbs b_ig, u_igt and
    # library heterogeneity - a 3.6x difference, which is why the marginal value
    # must never be used as the generating parameter.
    dispersion_median: float = 0.7881
    dispersion_log_sd: float = 1.4647

    # --- hierarchy (levels 1 and 2), from between-participant SD + pre/post corr ---
    between_participant_sd: float = 0.9994
    prepost_corr: float = 0.4656

    # --- effects ---
    time_effect: float = 0.0            # gamma_g; cancels in DiD
    effects: dict[str, float] = field(default_factory=dict)  # beta_g by gene name

    seed: int = 0

    # ---- derived hierarchy variances -------------------------------------
    @property
    def participant_sd(self) -> float:
        """sigma_b: the participant component of the total between-participant SD."""
        return float(np.sqrt(max(self.prepost_corr, 0.0)) * self.between_participant_sd)

    @property
    def participant_visit_sd(self) -> float:
        """sigma_u: the participant x visit component.

        corr(pre, post) = sigma_b^2 / (sigma_b^2 + sigma_u^2), so
        sigma_u = sd * sqrt(1 - corr). This is the level whose absence made
        ``participant_sd`` inert in the old simulator.
        """
        return float(np.sqrt(max(1.0 - self.prepost_corr, 0.0)) * self.between_participant_sd)

    @classmethod
    def from_targets(cls, targets: dict | None = None, **overrides) -> TranscriptomeSimConfig:
        """Build a config directly from measured TNBC targets."""
        t = targets if targets is not None else load_tnbc_targets()
        kw = dict(
            cells_per_pv_mean=t["cells_per_pv_mean"],
            cells_per_pv_cv=t["cells_per_pv_cv"],
            cells_per_pv_min=t["cells_per_pv_min"],
            cells_per_pv_max=t["cells_per_pv_max"],
            lib_log_mean=t["lib_log_mean"],
            lib_log_sd=t["lib_log_sd"],
            gene_rate_log_mean=t["gene_mean_log_mean"],
            gene_rate_log_sd=t["gene_mean_log_sd"],
            dispersion_median=t["dispersion_cell_level"],
            dispersion_log_sd=t["dispersion_cell_level_log_sd"],
            between_participant_sd=t["between_participant_sd"],
            prepost_corr=t["prepost_corr"],
        )
        kw.update(overrides)
        return cls(**kw)


def nested_panels(
    n_transcriptome: int,
    sizes: tuple[int, ...] = (50, 200, 500, 2000),
    rng: np.random.Generator | None = None,
) -> dict[int, list[int]]:
    """Nested analysis panels: each larger panel CONTAINS every smaller one.

    Nesting is what lets a panel-size effect be separated from a gene-identity
    effect. With independently drawn panels the two are confounded, and the
    previous benchmark's "progressive miscalibration with panel size" could not be
    distinguished from a change in which genes were tested.
    """
    rng = rng or np.random.default_rng(0)
    order = rng.permutation(n_transcriptome)
    panels: dict[int, list[int]] = {}
    for s in sorted(sizes):
        if s > n_transcriptome:
            raise ValueError(f"panel size {s} exceeds transcriptome {n_transcriptome}")
        panels[s] = sorted(order[:s].tolist())
    return panels


def make_signal(
    panel_genes: list[str],
    signal_fraction: float,
    architecture: SignalArch = "balanced",
    magnitude: float = 0.5,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Effect sizes for the tested panel under one of three signal architectures.

    ``balanced`` (primary)
        Half +magnitude, half -magnitude. No net directional shift, so the library
        size is unmoved and there is no compositional artifact.
    ``heterogeneous`` (primary)
        Symmetric mixture of weak/moderate/large effects — the most realistic.
    ``one_directional`` (stress test)
        All +magnitude. This is the previous design. Retain it, but label it a
        COMPOSITION-STRESS scenario: the coordinated shift moves the library-size
        reference, and roughly two thirds of the dreamlet inflation previously
        attributed to empirical-Bayes moderation is attributable to it.
    """
    rng = rng or np.random.default_rng(0)
    n_sig = int(round(len(panel_genes) * signal_fraction))
    if n_sig == 0:
        return {}
    chosen = rng.choice(np.asarray(panel_genes, dtype=object), size=n_sig, replace=False)

    if architecture == "one_directional":
        vals = np.full(n_sig, magnitude)
    elif architecture == "balanced":
        vals = np.full(n_sig, magnitude)
        vals[: n_sig // 2] *= -1.0
        rng.shuffle(vals)
    elif architecture == "heterogeneous":
        # 50% weak, 30% moderate, 20% large; symmetric in sign
        tiers = rng.choice([0.2, 0.5, 1.0], size=n_sig, p=[0.5, 0.3, 0.2])
        signs = rng.choice([-1.0, 1.0], size=n_sig)
        vals = tiers * signs * (magnitude / 0.5)
    else:  # pragma: no cover
        raise ValueError(f"unknown architecture {architecture!r}")
    return {str(g): float(v) for g, v in zip(chosen, vals)}


def _draw_cells_per_pv(cfg: TranscriptomeSimConfig, n: int, rng) -> np.ndarray:
    """Cell counts per participant-visit from a calibrated lognormal, not a constant."""
    mean = cfg.cells_per_pv_mean * cfg.cells_scale
    cv = cfg.cells_per_pv_cv
    sigma = float(np.sqrt(np.log1p(cv**2)))
    mu = float(np.log(mean) - sigma**2 / 2)
    draws = rng.lognormal(mu, sigma, size=n)
    lo = max(1, int(cfg.cells_per_pv_min * cfg.cells_scale))
    hi = max(lo + 1, int(cfg.cells_per_pv_max * cfg.cells_scale))
    return np.clip(np.round(draws), lo, hi).astype(int)


def simulate_trial_v2(cfg: TranscriptomeSimConfig) -> dict:
    """Simulate a full transcriptome under the three-level hierarchical NB model.

    Returns
    -------
    dict with
        ``adata``                cell-level raw counts (sparse), obs has
                                 participant/visit/arm and the OBSERVED library total
        ``pseudobulk_counts``    participant x visit summed counts, full transcriptome
        ``pseudobulk_means``     participant x visit mean counts, full transcriptome
        ``gene_names``           transcriptome gene names
        ``panels``               nested panel -> gene names
        ``truth``                beta_g by gene (the estimand)
        ``latent``               VALIDATION ONLY: true_library_size, b_ig, u_igt,
                                 alpha_g, phi_g. Never use as an analysis input.
        ``config``               the config used
    """
    rng = np.random.default_rng(cfg.seed)
    G = cfg.n_genes_transcriptome

    # --- participants and arms ---
    if cfg.design == "single_arm":
        arms = ["Treated"] * cfg.n_per_arm
    elif cfg.arm_ratio is not None:
        nt, nc = cfg.arm_ratio
        arms = ["Treated"] * nt + ["Control"] * nc
    else:
        arms = ["Treated"] * cfg.n_per_arm + ["Control"] * cfg.n_per_arm
    participants = [f"P{i:03d}" for i in range(len(arms))]
    visits = ["Pre", "Post"]

    # --- gene-level parameters ---
    # alpha_g is the log RATE relative to library size, so that
    #   E[Y_icg] = L_ic * exp(alpha_g)  and  sum_g exp(alpha_g) = 1,
    # i.e. a cell's counts total its library size. The empirical target
    # `gene_rate_log_mean` is the log MEAN COUNT PER CELL, which is larger than the
    # rate by E[L]; using it directly inflated every rate by ~3,147x and produced
    # 15.8M UMIs per cell against a TNBC median of 2,113. Renormalising the
    # simplex makes this exact and independent of n_genes_transcriptome (the
    # simulated transcriptome is smaller than the real one, so a fixed offset
    # would not transfer).
    alpha = rng.normal(cfg.gene_rate_log_mean, cfg.gene_rate_log_sd, size=G)
    alpha -= float(np.log(np.exp(alpha).sum()))  # sum_g exp(alpha_g) == 1
    phi = rng.lognormal(np.log(cfg.dispersion_median), cfg.dispersion_log_sd, size=G)
    phi = np.clip(phi, 1e-3, 1e3)
    gene_names = [f"gene_{i}" for i in range(G)]
    gidx = {g: i for i, g in enumerate(gene_names)}

    beta = np.zeros(G)
    for g, b in cfg.effects.items():
        if g in gidx:
            beta[gidx[g]] = b
    gamma = np.full(G, cfg.time_effect)

    # --- level 1 and level 2 random effects ---
    sd_b, sd_u = cfg.participant_sd, cfg.participant_visit_sd
    b_ig = rng.normal(0.0, sd_b, size=(len(participants), G)) if sd_b > 0 else np.zeros((len(participants), G))
    u_igt = (
        rng.normal(0.0, sd_u, size=(len(participants), len(visits), G))
        if sd_u > 0
        else np.zeros((len(participants), len(visits), G))
    )

    # --- cells ---
    n_pv = len(participants) * len(visits)
    cells_pv = _draw_cells_per_pv(cfg, n_pv, rng)

    blocks, obs_rows = [], []
    pb_sum_rows, pb_mean_rows = [], []
    k = 0
    for pi, (pid, arm) in enumerate(zip(participants, arms)):
        for ti, visit in enumerate(visits):
            n_cells = int(cells_pv[k])
            k += 1
            is_post = 1.0 if visit == "Post" else 0.0
            is_treated = 1.0 if arm == "Treated" else 0.0

            # Draw L so the simulated TOTAL counts match the empirical MEDIAN
            # library size. A cell's total has mean E[L] = exp(mu + sd^2/2); with
            # many genes the total is near-symmetric, so its median tracks E[L]
            # rather than exp(mu). Subtracting the Jensen term makes E[L] equal the
            # calibration target instead of overshooting it by exp(sd^2/2) = 1.4x.
            L = rng.lognormal(
                cfg.lib_log_mean - cfg.lib_log_sd**2 / 2.0, cfg.lib_log_sd, size=n_cells
            )
            # log mu = log L + alpha + b + u + gamma*Post + beta*(T*Post)
            log_rate = (
                alpha
                + b_ig[pi]
                + u_igt[pi, ti]
                + gamma * is_post
                + beta * is_treated * is_post
            )
            mu = np.exp(log_rate)[None, :] * L[:, None]
            # NB via gamma-Poisson: var = mu + phi*mu^2  => shape = 1/phi
            shape = 1.0 / phi
            lam = rng.gamma(shape[None, :], mu / shape[None, :])
            counts = rng.poisson(lam).astype(np.int32)

            blocks.append(sp.csr_matrix(counts))
            obs_rows.append(
                pd.DataFrame(
                    {
                        "participant": pid,
                        "visit": visit,
                        "arm": arm,
                        "true_library_size": L,
                    }
                )
            )
            s = counts.sum(axis=0)
            pb_sum_rows.append({"participant": pid, "visit": visit, "arm": arm,
                                "n_cells": n_cells,
                                **{g: int(v) for g, v in zip(gene_names, s)}})
            pb_mean_rows.append({"participant": pid, "visit": visit, "arm": arm,
                                 "n_cells": n_cells,
                                 **{g: float(v) for g, v in zip(gene_names, s / n_cells)}})

    X = sp.vstack(blocks, format="csr")
    obs = pd.concat(obs_rows, ignore_index=True)
    obs["observed_library_size"] = np.asarray(X.sum(axis=1)).ravel()
    obs.index = [f"cell_{i}" for i in range(X.shape[0])]

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_names

    panels = {
        size: [gene_names[i] for i in idx]
        for size, idx in nested_panels(G, rng=np.random.default_rng(cfg.seed + 1)).items()
    }

    return {
        "adata": adata,
        "pseudobulk_counts": pd.DataFrame(pb_sum_rows),
        "pseudobulk_means": pd.DataFrame(pb_mean_rows),
        "gene_names": gene_names,
        "panels": panels,
        "truth": {g: float(beta[gidx[g]]) for g in cfg.effects if g in gidx},
        "latent": {
            "alpha_g": alpha,
            "phi_g": phi,
            "b_ig": b_ig,
            "u_igt": u_igt,
            "participants": participants,
            "arms": arms,
        },
        "config": cfg,
    }
