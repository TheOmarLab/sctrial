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
    "load_empirical",
    "build_params",
    "gene_baseline_rates",
    "expected_counts_per_cell",
    "eligible_panel_genes",
    "iter_pv_blocks",
    "oracle_estimands",
]

SignalArch = Literal["balanced", "heterogeneous", "one_directional"]

# Genes generated per pass inside a participant-visit block. Bounds the float64
# gamma/Poisson transient; see the note in ``iter_pv_blocks``.
_GENE_CHUNK = 2000

# Gauss-Hermite nodes for the oracle expectation over the random effects.
_QUAD_NODES = 64

# Offset separating the gene-rate stream from every other draw. Any fixed odd
# value works; it exists so the two streams cannot silently coincide.
_SUBSTREAM_OFFSET = 7919

# Rate bins used to impute dispersion for genes with too few counts to estimate
# one. Enough bins to follow the mean-dependence, few enough that each holds a
# usable empirical distribution.
_RATE_BINS = 50


def _validation_dir() -> Path:
    """Locate the calibration outputs without guessing a parent depth.

    ``parents[N]`` is the defect that resolved outside the project on HPC and
    silently blanked four figure panels while every script exited 0. The layouts
    genuinely differ: ``manuscript/`` sits inside the project root on the cluster
    and beside the repo locally. Check, do not assume.
    """
    import os

    env = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
    if env:
        return Path(env) / "benchmark" / "validation"
    here = Path(__file__).resolve()
    repo = here.parents[3]  # <repo>/src/sctrial/benchmark/simulator_v2.py
    for base in (repo / "manuscript", repo.parent.parent / "manuscript"):
        if base.is_dir():
            return base / "benchmark" / "validation"
    return repo / "manuscript" / "benchmark" / "validation"


def load_tnbc_targets(path: str | Path | None = None) -> dict:
    """Empirical TNBC targets measured from the processed h5ad.

    These are the quantities a defensible calibration must reproduce. Measured by
    ``scripts/calibrate_simulator.py targets`` from the v5 TNBC object
    (141,553 cells x 20,284 genes, 12 paired participants, 6 v 6 arms).
    """
    path = Path(path) if path is not None else _validation_dir() / "tnbc_sim_targets.json"
    if not path.exists():
        raise FileNotFoundError(
            f"TNBC simulation targets not found at {path}. Run "
            "`python scripts/calibrate_simulator.py targets` first — the simulator "
            "must not fall back to uncalibrated defaults, which is how the previous "
            "benchmark produced 2.3e7 UMIs per cell."
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
    # Matches the TNBC transcriptome (20,284 genes). Genes DETECTED per cell is an
    # observable property in its own right, not merely a percentage of the feature
    # universe: at 12,000 genes the simulation detected 629/cell against TNBC's
    # 1,080, which changes dropout, count occupancy and effective normalisation even
    # though the per-gene detection rate matched. Statistical tests still run only on
    # the nested 50/200/500/2000 panels; the remaining genes exist to give realistic
    # count allocation, library normalisation, NEBULA offsets and dreamlet norm factors.
    n_genes_transcriptome: int = 20284
    arm_ratio: tuple[int, int] | None = None
    # Minimum expected counts per cell for a gene to be ELIGIBLE for the tested
    # panel. The transcriptome is generated in full -- that is the whole point,
    # since normalisation and offsets must see it -- but a panel drawn uniformly
    # from it is dominated by genes no assay would report on. With TNBC's rate
    # distribution a random 50-gene panel contains ~9 detectable genes, and
    # filterByExpr drops the rest: measured 30-54% finite p-values for dreamlet
    # and 42-60% for NEBULA. A real workflow filters to detected genes and then
    # tests; so does this. 0.05 counts/cell leaves ~3,300 eligible genes, enough
    # for the 2,000-gene panel.
    panel_min_mean_count: float = 0.05
    # Tested panel sizes, stated in the config rather than defaulted inside the
    # panel builder so a configuration that cannot supply them fails loudly at
    # construction instead of somewhere downstream.
    panel_sizes: tuple[int, ...] = (50, 200, 500, 2000)

    # --- cells per participant-visit (distribution, not a fixed number) ---
    cells_per_pv_mean: float = 5898.0
    cells_per_pv_cv: float = 0.911
    cells_per_pv_min: int = 147
    cells_per_pv_max: int = 27653
    cells_scale: float = 1.0  # shrink for tractable benchmarking; 1.0 = TNBC scale
    # SCENARIO parameter, not a calibration: when set, every participant-visit
    # gets exactly this many cells. Used by the "varying cells" scenarios, where
    # cell yield is the thing being varied and must therefore be controlled
    # rather than resampled. Reported in the scenario name.
    cells_per_pv_fixed: int | None = None
    # SCENARIO parameter: fraction of participants whose Post visit is missing.
    # Tests robustness to incomplete follow-up, which is the norm in real trials.
    missing_rate: float = 0.0

    # --- library size (per cell) ---
    # Prefer EMPIRICAL RESAMPLING: the lognormal fit reproduced the median poorly
    # (+34%) and Q75 badly (+53%). There is no reason to fit a parametric family to
    # a nuisance distribution we already possess; resampling reproduces the skew,
    # both tails and the heteroscedasticity exactly. The lognormal parameters remain
    # as a fallback when the empirical file is absent.
    use_empirical_library: bool = True
    use_empirical_cells_per_pv: bool = True
    empirical_path: str | None = None
    lib_log_mean: float = 7.7333
    lib_log_sd: float = 0.8000

    # --- baseline gene rates ---
    # Prefer RESAMPLING the empirical gene-proportion vector. Fitting a lognormal
    # to the mean and SD of log gene expression fails badly: TNBC's log
    # proportions have mean -12.685 and SD 2.643, but the distribution has a
    # LIGHTER upper tail than lognormal, so E[exp(x)] is 0.171 rather than the
    # 0.353 those two moments imply. A lognormal with the right two moments
    # therefore implies 7,157 counts per cell against an actual mean library of
    # 3,467 -- a 2.06x inconsistency. Renormalising onto the simplex then divides
    # every gene down by that factor, and the simulation detected 783 genes per
    # cell against TNBC's 1,087 with a zero fraction of 0.954 against 0.940.
    #
    # The empirical proportions sum to 1 by construction, so resampling them
    # satisfies the simplex constraint exactly AND reproduces the tail. Same
    # reasoning as for library sizes and cells per participant-visit: there is no
    # reason to fit a parametric family to a nuisance distribution we possess.
    use_empirical_gene_rates: bool = True
    gene_rate_log_mean: float = -4.5338
    gene_rate_log_sd: float = 2.6428

    # --- NB dispersion: var = mu + phi * mu^2, phi_g lognormal ---
    # CALIBRATED WITHIN A HOMOGENEOUS POPULATION, BY GAMMA-POISSON MLE.
    #
    # The simulator generates ONE cell population, so its generating dispersion must
    # be the WITHIN-cell-type value. Estimating it on cell-type-pooled TNBC absorbs
    # between-cell-type mean differences into alpha: conditioning on participant x
    # visit alone gives 0.774, adding cell type gives 0.275 (0.35x) - i.e. 65% of the
    # apparent "cell-level" dispersion was unmodelled cell-type heterogeneity. The
    # implausible alpha = 10-30 at low expression seen in the MARGINAL curve is
    # absent from both conditional curves; the true cell-level relationship is
    # nearly flat. This is the same reason reference-based simulators (muscat,
    # scDesign3) estimate parameters within subpopulations rather than pooling.
    #
    # ESTIMATOR: Cox-Reid adjusted profile-likelihood Gamma-Poisson MLE per gene with
    # empirical-Bayes shrinkage toward a mean-dependent trend
    # (``sctrial.benchmark.calibration.conditional_dispersion``), NOT a raw
    # method-of-moments estimate. Strata are participant x visit x cell type and are
    # therefore small; MoM is badly biased there, and the Cox-Reid term is what
    # removes the bias from estimating one mean per stratum.
    #
    # CONSEQUENCE FOR VALIDATION: the simulated MARGINAL mean-variance curve is not
    # expected to match cell-type-pooled TNBC, because the simulation has no cell
    # types to pool. The acceptance criterion is agreement of the CONDITIONAL curve.
    # Matching the pooled marginal by tuning a latent parameter would be
    # unidentifiable - many hierarchy/dispersion combinations reproduce the same
    # marginal, so such a loop can silently compensate a miscalibrated hierarchy.
    #
    # PARAMETERISATION (stated explicitly - this is a documented trap):
    #   THIS SIMULATOR uses NB2:            Var(Y) = mu + phi * mu^2
    #     implemented as gamma-Poisson with Gamma(shape=1/phi, scale=mu*phi),
    #     so `dispersion_median` IS the NB2 alpha.
    #   NEBULA parameterises the cell-level term as mu^2/phi_nebula, i.e. its
    #     "dispersion" is the RECIPROCAL of this one. Do not pass one for the other.
    #   The naive estimator (Var(Y) - E(Y)) / E(Y)^2 computed ACROSS ALL CELLS does
    #     NOT estimate this parameter once hierarchical terms exist: it absorbs
    #     b_ig, u_igt and library-size heterogeneity, giving 2.837 against a
    #     conditional 0.275 - which is why the marginal value must never be used as
    #     the generating parameter.
    # Prefer RESAMPLING the empirical per-gene dispersion, paired with the gene
    # rate. The scalars below are a fallback for when that pool is unavailable.
    use_empirical_dispersion: bool = True
    # Cox-Reid APL Gamma-Poisson MLE with EB shrinkage, over the ESTIMABLE genes.
    dispersion_median: float = 0.4416
    # Mean-dependence of the CONDITIONAL curve: log phi_g declines slowly with the
    # gene's log rate. Fitted on the same conditional estimates, so it is on the same
    # footing as `dispersion_median`. 0.0 restores flat behaviour.
    dispersion_mean_slope: float = -0.1753
    # The log-rate the trend is anchored at. It MUST be the median over the same
    # population `dispersion_median` was computed on (the ESTIMABLE genes).
    # Anchoring at the median over ALL genes instead shifted every dispersion by
    # exp(slope * 1.6983) = 0.74 and was the measured cause of the Gate C failure.
    dispersion_anchor: float = -10.5486
    # RESIDUAL sd about the trend, NOT the total sd of log alpha. Exporting the
    # total and consuming it as the residual double-counts the trend variance.
    dispersion_residual_sd: float = 1.3224

    # --- hierarchy (levels 1 and 2), from between-participant SD + pre/post corr ---
    # LATENT, split-half corrected, measured on TNBC. These are pooled over cell
    # types; see the conditioning question recorded in tasks/MASTER_PLAN.md.
    between_participant_sd: float = 0.5969
    prepost_corr: float = 0.3987

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

    # `from_targets` was DELETED. It read keys (`dispersion_cell_level`,
    # `dispersion_cell_level_log_sd`) that no longer exist in the targets file, so
    # it would have raised on first use -- and a second, dead config constructor is
    # exactly how a reparameterisation silently diverges from the live one. The
    # single supported path is scripts/calibrate_simulator.py.


def load_empirical(path: str | Path | None = None) -> dict | None:
    """Empirical TNBC nuisance pools (library sizes, cells per participant-visit).

    Returns None if absent, so the simulator falls back to the parametric fits
    rather than failing — but the fits are known to reproduce the library-size
    distribution poorly, so the empirical file should normally be present.
    """
    path = Path(path) if path is not None else _validation_dir() / "tnbc_empirical.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def gene_baseline_rates(cfg: TranscriptomeSimConfig) -> np.ndarray:
    """``alpha_g``: log gene proportions, summing to 1 on the exponential scale.

    Reproduced exactly as :func:`build_params` draws it -- it is the FIRST draw
    from ``default_rng(cfg.seed)``, so a fresh generator reproduces it bit for
    bit. That lets the analysis panel be chosen before the effects are defined
    (which the orchestrator needs, since the signal is injected on the tested
    genes) without simulating anything. ``tests/test_benchmark.py`` asserts the
    two agree.

    A gene with zero empirical counts gets ``alpha = -inf``, i.e. it is never
    expressed. That is faithful rather than a degenerate case, and callers must
    treat ``exp(alpha)`` as the rate rather than assuming ``alpha`` is finite.
    """
    G = cfg.n_genes_transcriptome
    emp = load_empirical(cfg.empirical_path)
    if emp is not None and cfg.use_empirical_gene_rates and "gene_mean_count" in emp:
        pool = np.asarray(emp["gene_mean_count"], dtype=float)
        pool = pool[np.isfinite(pool) & (pool >= 0)]
        order = _gene_rate_permutation(cfg)
        props = pool[order] if pool.size == G else pool[order % max(pool.size, 1)]
        total = props.sum()
        if total <= 0:
            raise ValueError("empirical gene means sum to zero")
        props = props / total
        with np.errstate(divide="ignore"):
            return np.log(props)
    rng = np.random.default_rng(cfg.seed)
    alpha = rng.normal(cfg.gene_rate_log_mean, cfg.gene_rate_log_sd, size=G)
    return alpha - float(np.log(np.exp(alpha).sum()))


def _gene_rate_permutation(cfg: TranscriptomeSimConfig) -> np.ndarray:
    """The gene ordering shared by the rate and dispersion pools.

    Both are resampled with THIS permutation so each simulated gene keeps a real
    gene's (rate, dispersion) pair. Drawing them independently destroys the
    mean-dependence of dispersion: measured slope -0.0165 against the empirical
    -0.1753.
    """
    return np.random.default_rng(cfg.seed).permutation(cfg.n_genes_transcriptome)


def expected_counts_per_cell(cfg: TranscriptomeSimConfig) -> np.ndarray:
    """Expected counts per cell for each gene: ``E[L] * exp(alpha_g)``."""
    emp = load_empirical(cfg.empirical_path)
    if emp is not None and cfg.use_empirical_library and "library_sizes" in emp:
        mean_lib = float(np.mean(emp["library_sizes"]))
    else:
        # L ~ lognormal(mu - sd^2/2, sd), so E[L] = exp(mu).
        mean_lib = float(np.exp(cfg.lib_log_mean))
    return np.exp(gene_baseline_rates(cfg)) * mean_lib


def eligible_panel_genes(cfg: TranscriptomeSimConfig) -> np.ndarray:
    """Indices of genes a real analysis would carry into testing."""
    exp_counts = expected_counts_per_cell(cfg)
    idx = np.flatnonzero(exp_counts >= cfg.panel_min_mean_count)
    if idx.size == 0:
        raise ValueError(
            f"no gene reaches {cfg.panel_min_mean_count} expected counts per cell; "
            "the rate distribution or the library size is miscalibrated"
        )
    return idx


def nested_panels(
    cfg: TranscriptomeSimConfig,
    sizes: tuple[int, ...] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[int, list[int]]:
    """Nested analysis panels drawn from the DETECTABLE genes.

    Nesting is what lets a panel-size effect be separated from a gene-identity
    effect. With independently drawn panels the two are confounded, and the
    previous benchmark's "progressive miscalibration with panel size" could not be
    distinguished from a change in which genes were tested.

    Panels are drawn from :func:`eligible_panel_genes`, not from the whole
    transcriptome. The transcriptome is still simulated in full and still supplies
    every normalisation denominator and offset; it is only the TESTED set that is
    restricted, exactly as a real pipeline restricts it.
    """
    rng = rng or np.random.default_rng(0)
    sizes = tuple(sizes) if sizes is not None else tuple(cfg.panel_sizes)
    eligible = eligible_panel_genes(cfg)
    order = rng.permutation(eligible)
    panels: dict[int, list[int]] = {}
    for s in sorted(sizes):
        if s > eligible.size:
            raise ValueError(
                f"panel size {s} exceeds the {eligible.size} genes reaching "
                f"{cfg.panel_min_mean_count} expected counts per cell. Lower "
                "panel_min_mean_count or raise the simulated depth; do NOT pad the "
                "panel with undetectable genes, which is what produced 30-60% "
                "non-finite p-values."
            )
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


def _draw_cells_per_pv(cfg: TranscriptomeSimConfig, n: int, rng, emp=None) -> np.ndarray:
    """Cell counts per participant-visit.

    Resampled from the empirical TNBC pool when available (the parametric fit
    mismatched the lower tail badly: Q05 ratio 2.63). Unequal cell yield per
    biosample is exactly the phenomenon pseudobulk precision weighting addresses,
    so its distribution must be right rather than merely its median.
    """
    if cfg.cells_per_pv_fixed is not None:
        return np.full(n, max(int(cfg.cells_per_pv_fixed), 1), dtype=int)
    if emp is not None and cfg.use_empirical_cells_per_pv and "cells_per_pv" in emp:
        pool = np.asarray(emp["cells_per_pv"], dtype=float)
        draws = rng.choice(pool, size=n, replace=True)
        draws = draws * cfg.cells_scale
        return np.maximum(np.round(draws), 1).astype(int)
    mean = cfg.cells_per_pv_mean * cfg.cells_scale
    cv = cfg.cells_per_pv_cv
    sigma = float(np.sqrt(np.log1p(cv**2)))
    mu = float(np.log(mean) - sigma**2 / 2)
    draws = rng.lognormal(mu, sigma, size=n)
    lo = max(1, int(cfg.cells_per_pv_min * cfg.cells_scale))
    hi = max(lo + 1, int(cfg.cells_per_pv_max * cfg.cells_scale))
    return np.clip(np.round(draws), lo, hi).astype(int)


def build_params(cfg: TranscriptomeSimConfig) -> dict:
    """Draw every gene-level and participant-level latent parameter.

    Separated from cell generation so that the full simulation
    (:func:`simulate_trial_v2`) and the calibration gates
    (:mod:`sctrial.benchmark.gates`, which only need summary statistics and must
    not materialise 141k x 20k count matrices) consume **one** implementation of
    the generative model. Two implementations of a generative model is how a
    calibration silently stops describing the thing it calibrates.
    """
    # TWO explicitly independent streams. ``gene_baseline_rates`` consumes
    # ``default_rng(cfg.seed)`` so the panel can be chosen without simulating;
    # everything else draws from a separately seeded generator. Sharing one
    # stream would make the dispersion noise a deterministic function of the
    # gene rates, which is a dependence the model does not contain.
    rng = np.random.default_rng(cfg.seed + _SUBSTREAM_OFFSET)
    emp = load_empirical(cfg.empirical_path)
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
    # simplex makes this exact and independent of n_genes_transcriptome.
    # One implementation, shared with the panel selector, so the panel can never
    # be chosen from a different gene-rate vector than the data are generated
    # from. Consumes exactly the same first draw from this seed.
    alpha = gene_baseline_rates(cfg)

    # Centre the random effects MULTIPLICATIVELY. With log mu = log L + alpha + b + u
    # and sum_g exp(alpha_g) = 1, the per-cell total is L * E[exp(b+u)] =
    # L * exp((sd_b^2 + sd_u^2)/2) -- a 1.65x inflation that made every simulated
    # library 1.7x the empirical one. Subtracting the Jensen term restores
    # E[exp(b+u)] = 1 so totals track the resampled library sizes, without changing
    # the variance structure that sets the pre/post correlation.
    re_jensen = (cfg.participant_sd**2 + cfg.participant_visit_sd**2) / 2.0

    # Dispersion: RESAMPLE the empirical per-gene conditional alpha, PAIRED with
    # the gene rate. Same principle as library sizes and gene proportions -- do not
    # fit a parametric family to a distribution already in hand.
    #
    # The parametric handoff it replaces carried three defects that together
    # explained the entire Gate C failure:
    #  (a) ANCHOR MISMATCH. `dispersion_median` is the median over the 9,727
    #      ESTIMABLE genes, but the trend was re-anchored at the median over all
    #      20,284 -- an offset of 1.6983 log units, multiplying every estimable
    #      gene's phi by exp(-0.1753 * 1.6983) = 0.7425. The observed ratio of
    #      simulated to calibrated dispersion was 0.32371/0.44164 = 0.7330, a 1.3%
    #      match. This was the cause, not a contributor.
    #  (b) `dispersion_log_sd` is the TOTAL sd of log alpha but was consumed as the
    #      RESIDUAL sd on top of the trend, double-counting 0.0415 of variance.
    #  (c) `trend_intercept` was computed and stored but never exported, which is
    #      why (a) was possible at all.
    # None of the three can recur, because no anchor, slope or residual sd is used.
    #
    # PAIRING IS NOT OPTIONAL: resampling alpha independently of expression
    # destroys the mean-dependence (measured slope -0.0165 against -0.1753). The
    # SAME permutation that drew the gene rates is reused so each gene keeps its
    # own (rate, dispersion) pair.
    finite = np.isfinite(alpha)
    phi = None
    if emp is not None and cfg.use_empirical_dispersion and "cond_alpha" in emp:
        pool = np.asarray(emp["cond_alpha"], dtype=float)
        rate_pool = np.asarray(emp.get("gene_mean_count", []), dtype=float)
        if pool.size == rate_pool.size == G:
            order = _gene_rate_permutation(cfg)
            phi = pool[order]
            rates = rate_pool[order]
            ok = np.isfinite(phi) & (phi > 0)
            if ok.sum() < 100:
                phi = None
            else:
                # 10,557 of 20,284 genes (52%) have too few counts to identify a
                # dispersion. They are NOT irrelevant, and an earlier note here
                # claiming otherwise was wrong: they carry 1.15% of transcriptome
                # counts and contribute ~24 detected genes per cell at the median
                # library, about 30% of the Gate A genes-detected gap. They also
                # enter full-transcriptome normalisation, TMM factors, sparsity and
                # the CPM denominator.
                #
                # Assigning them all one value would impose a spike on the
                # background transcriptome. Instead draw each from the empirical
                # conditional dispersion distribution of estimable genes in its own
                # LOCAL RATE BIN, so the background keeps realistic dispersion
                # spread rather than a point mass.
                phi = phi.copy()
                miss = np.flatnonzero(~ok)
                if miss.size:
                    est_rates = rates[ok]
                    est_phi = phi[ok]
                    edges = np.quantile(est_rates, np.linspace(0, 1, _RATE_BINS + 1))
                    edges[0], edges[-1] = -np.inf, np.inf
                    bin_of_missing = np.clip(
                        np.searchsorted(edges, rates[miss], side="right") - 1,
                        0, _RATE_BINS - 1,
                    )
                    est_bin = np.clip(
                        np.searchsorted(edges, est_rates, side="right") - 1,
                        0, _RATE_BINS - 1,
                    )
                    draw = np.random.default_rng(cfg.seed + 104729)
                    for b in np.unique(bin_of_missing):
                        donors = est_phi[est_bin == b]
                        if donors.size == 0:
                            donors = est_phi
                        tgt = miss[bin_of_missing == b]
                        phi[tgt] = draw.choice(donors, size=tgt.size, replace=True)
    if phi is None:
        alpha_c = np.where(finite, alpha, np.min(alpha[finite]) if finite.any() else 0.0)
        log_alpha = np.log(cfg.dispersion_median) + cfg.dispersion_mean_slope * (
            alpha_c - cfg.dispersion_anchor
        )
        phi = np.exp(log_alpha + rng.normal(0.0, cfg.dispersion_residual_sd, size=G))
    phi = np.clip(np.asarray(phi, dtype=float), 1e-3, 1e3)

    gene_names = [f"gene_{i}" for i in range(G)]
    gidx = {g: i for i, g in enumerate(gene_names)}

    beta = np.zeros(G)
    for g, b in cfg.effects.items():
        if g in gidx:
            beta[gidx[g]] = b
    gamma = np.full(G, cfg.time_effect)

    # --- level 1 and level 2 random effects ---
    sd_b, sd_u = cfg.participant_sd, cfg.participant_visit_sd
    shape_b = (len(participants), G)
    shape_u = (len(participants), len(visits), G)
    b_ig = rng.normal(0.0, sd_b, size=shape_b) if sd_b > 0 else np.zeros(shape_b)
    u_igt = rng.normal(0.0, sd_u, size=shape_u) if sd_u > 0 else np.zeros(shape_u)

    # --- cells ---
    n_pv = len(participants) * len(visits)
    cells_pv = _draw_cells_per_pv(cfg, n_pv, rng, emp=emp)

    # Missing Post visits. Drawn here rather than in the block loop so the set is
    # part of the same seeded draw as everything else and a replicate is fully
    # reproducible from its seed.
    dropped: set[tuple[str, str]] = set()
    if cfg.missing_rate > 0:
        n_drop = int(round(len(participants) * cfg.missing_rate))
        if n_drop:
            for i in rng.choice(len(participants), size=n_drop, replace=False):
                dropped.add((participants[int(i)], "Post"))

    return {
        "dropped": dropped,
        "rng": rng,
        "emp": emp,
        "G": G,
        "participants": participants,
        "arms": arms,
        "visits": visits,
        "gene_names": gene_names,
        "gene_index": gidx,
        "alpha": alpha,
        "phi": phi,
        "beta": beta,
        "gamma": gamma,
        "b_ig": b_ig,
        "u_igt": u_igt,
        "cells_pv": cells_pv,
        "re_jensen": re_jensen,
        "config": cfg,
    }


def iter_pv_blocks(cfg: TranscriptomeSimConfig, params: dict | None = None):
    """Yield one participant-visit block of cell-level counts at a time.

    Yielding rather than returning is what makes full-TNBC-scale Monte Carlo
    calibration tractable: a single replicate is 141k cells x 20,284 genes, which
    never has to exist at once if the consumer only needs summary statistics.

    Yields
    ------
    dict with ``participant``, ``visit``, ``arm``, ``counts`` (n_cells x G int32),
    ``library`` (the drawn latent L per cell), ``pi``, ``ti``.
    """
    p = params if params is not None else build_params(cfg)
    rng, emp = p["rng"], p["emp"]
    alpha, phi, beta, gamma = p["alpha"], p["phi"], p["beta"], p["gamma"]
    b_ig, u_igt, cells_pv = p["b_ig"], p["u_igt"], p["cells_pv"]
    re_jensen = p["re_jensen"]
    shape = 1.0 / phi

    k = 0
    for pi, (pid, arm) in enumerate(zip(p["participants"], p["arms"])):
        for ti, visit in enumerate(p["visits"]):
            n_cells = int(cells_pv[k])
            k += 1
            if (pid, visit) in p["dropped"]:
                continue
            is_post = 1.0 if visit == "Post" else 0.0
            is_treated = 1.0 if arm == "Treated" else 0.0

            if emp is not None and cfg.use_empirical_library and "library_sizes" in emp:
                # Resample the observed depth distribution directly: the lognormal
                # fit reproduced the median +34% and Q75 +53% off.
                L = rng.choice(
                    np.asarray(emp["library_sizes"], dtype=float), size=n_cells, replace=True
                )
            else:
                L = rng.lognormal(
                    cfg.lib_log_mean - cfg.lib_log_sd**2 / 2.0, cfg.lib_log_sd, size=n_cells
                )

            log_rate = (
                alpha
                + b_ig[pi]
                + u_igt[pi, ti]
                + gamma * is_post
                + beta * is_treated * is_post
                - re_jensen
            )
            # Generate in GENE CHUNKS. At full TNBC scale a single block is up to
            # 27,653 cells x 20,284 genes, so the float64 gamma and Poisson
            # intermediates would peak near 11 GB per worker and make Monte Carlo
            # calibration impossible to parallelise. Chunking bounds the transient
            # to the chunk while leaving the output identical in distribution.
            rate = np.exp(log_rate)
            counts = np.empty((n_cells, len(alpha)), dtype=np.int32)
            for c0 in range(0, len(alpha), _GENE_CHUNK):
                sl = slice(c0, min(c0 + _GENE_CHUNK, len(alpha)))
                mu = rate[None, sl] * L[:, None]
                # NB via gamma-Poisson: var = mu + phi*mu^2  => shape = 1/phi
                lam = rng.gamma(shape[None, sl], mu / shape[None, sl])
                counts[:, sl] = rng.poisson(lam)

            yield {
                "participant": pid,
                "visit": visit,
                "arm": arm,
                "counts": counts,
                "library": L,
                "pi": pi,
                "ti": ti,
            }


def oracle_estimands(params: dict) -> dict[str, np.ndarray]:
    """Per-gene POPULATION truth on each METHOD CLASS's own estimand scale.

    Different methods do not estimate the same functional, so scoring them all
    against the injected ``beta`` silently penalises whichever method's estimand
    differs most from it. Two published conclusions have already been produced
    that way (log2-versus-natural-log, then this).

    ``count_link``
        The log-link coefficient: exactly ``beta_g``. This is the estimand of
        NEBULA (NB log link) and of log-CPM pseudobulk models (dreamlet,
        limma-voom, edgeR): with full-transcriptome normalisation the library
        reference does not move with a panel-restricted effect, so
        ``DiD[log CPM_g] = beta_g``.

    ``log1p_cpm``
        The estimand targeted by sctrial and the Wilcoxon change score: the
        difference-in-differences of ``log(1 + CPM)`` at participant level.
        Because ``d/dx log(1+x) = 1/(1+x)`` this equals ``beta_g`` only when
        CPM >> 1, and is attenuated for low-expression genes at realistic depth.
        The attenuation is a real property of the estimand and is reported, not
        corrected away.

    MARGINALISED, NOT CONDITIONED. The expectation is taken over the participant
    and participant x visit random effects rather than evaluated at their
    realised values. Conditioning on the realised draw makes the "truth" a random
    quantity: under a true null it returns values like +0.15 and -0.86 instead of
    zero, so a perfectly calibrated method would be scored as biased and a null
    scenario would have a non-null target. The random effects are exactly the
    variability the standard error is meant to cover.

    The expectation ``E[log(1 + exp(m + eps))]``, ``eps ~ N(0, sigma_b^2 +
    sigma_u^2)``, is evaluated by Gauss-Hermite quadrature.
    """
    cfg = params["config"]
    alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
    re_jensen = params["re_jensen"]
    sigma2 = cfg.participant_sd**2 + cfg.participant_visit_sd**2

    nodes, weights = np.polynomial.hermite_e.hermegauss(_QUAD_NODES)
    weights = weights / np.sqrt(2.0 * np.pi)
    eps = np.sqrt(sigma2) * nodes

    def _e_log1p_cpm(shift: np.ndarray) -> np.ndarray:
        """E[log(1 + CPM)] with the log-mean shifted by ``shift`` per gene."""
        m = np.log(1e6) + alpha + shift - re_jensen
        # logaddexp(0, z) == log1p(exp(z)) without overflowing for large z.
        return (np.logaddexp(0.0, m[:, None] + eps[None, :]) * weights[None, :]).sum(axis=1)

    zero = np.zeros_like(alpha)
    # Treated: (post - pre); Control: (post - pre). gamma is common and cancels,
    # but it is carried explicitly so a future non-common time effect is handled.
    treated_delta = _e_log1p_cpm(gamma + beta) - _e_log1p_cpm(zero)
    control_delta = _e_log1p_cpm(gamma) - _e_log1p_cpm(zero)

    if cfg.design == "single_arm":
        log1p_cpm = treated_delta
    else:
        log1p_cpm = treated_delta - control_delta

    return {"count_link": beta.copy(), "log1p_cpm": log1p_cpm}


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
        ``truth``                beta_g by gene (the injected effect)
        ``oracle``               per-gene truth on each method class's own estimand
                                 scale (see :func:`oracle_estimands`)
        ``latent``               VALIDATION ONLY: b_ig, u_igt, alpha_g, phi_g.
                                 Never use as an analysis input.
        ``config``               the config used
    """
    params = build_params(cfg)
    gene_names = params["gene_names"]
    gidx = params["gene_index"]

    blocks, obs_rows = [], []
    pb_sum_rows, pb_mean_rows = [], []
    for blk in iter_pv_blocks(cfg, params=params):
        counts = blk["counts"]
        n_cells = counts.shape[0]
        blocks.append(sp.csr_matrix(counts))
        obs_rows.append(
            pd.DataFrame(
                {
                    "participant": blk["participant"],
                    "visit": blk["visit"],
                    "arm": blk["arm"],
                    "true_library_size": blk["library"],
                }
            )
        )
        s = counts.sum(axis=0)
        meta = {
            "participant": blk["participant"],
            "visit": blk["visit"],
            "arm": blk["arm"],
            "n_cells": n_cells,
        }
        pb_sum_rows.append({**meta, **{g: int(v) for g, v in zip(gene_names, s)}})
        pb_mean_rows.append({**meta, **{g: float(v) for g, v in zip(gene_names, s / n_cells)}})

    X = sp.vstack(blocks, format="csr")
    obs = pd.concat(obs_rows, ignore_index=True)
    obs["observed_library_size"] = np.asarray(X.sum(axis=1)).ravel()
    obs.index = [f"cell_{i}" for i in range(X.shape[0])]

    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_names

    panels = {
        size: [gene_names[i] for i in idx]
        for size, idx in nested_panels(
            cfg, rng=np.random.default_rng(cfg.seed + 1)
        ).items()
    }

    oracle = oracle_estimands(params)
    return {
        "adata": adata,
        "pseudobulk_counts": pd.DataFrame(pb_sum_rows),
        "pseudobulk_means": pd.DataFrame(pb_mean_rows),
        "gene_names": gene_names,
        "panels": panels,
        "truth": {g: float(params["beta"][gidx[g]]) for g in cfg.effects if g in gidx},
        "oracle": {
            scale: {g: float(vals[gidx[g]]) for g in gene_names}
            for scale, vals in oracle.items()
        },
        "latent": {
            "alpha_g": params["alpha"],
            "phi_g": params["phi"],
            "b_ig": params["b_ig"],
            "u_igt": params["u_igt"],
            "participants": params["participants"],
            "arms": params["arms"],
        },
        "config": cfg,
    }
