"""The calibration estimators must recover parameters they themselves generated.

Every number the simulator is configured from comes out of
``sctrial.benchmark.calibration``. If those estimators are biased, the simulator
is miscalibrated in a way no downstream check can detect -- the benchmark would
simply be measuring methods against the wrong data-generating process while every
validation figure looked fine.

The only way to test an estimator without a second, independent estimator is to
run it on data whose parameters are known by construction. That is what these
tests do: generate at a known value, estimate, and require recovery.

Two specific errors are pinned here because both have already happened:

1. **Marginal used as if conditional.** The dispersion estimator must condition on
   the homogeneity stratum. Pooling heterogeneous populations inflates it (TNBC:
   0.774 pooled versus 0.275 within cell type), and generating at the pooled value
   overshoots the observable mean-variance curve 4.6-9.2x.
2. **Observable used as if latent.** The pre/post correlation of ``log(1+CPM)`` is
   attenuated by pseudobulk sampling noise. Setting the generating parameter equal
   to the observable under-disperses the hierarchy (0.348 observable for a 0.466
   generating value).
"""
from __future__ import annotations

import numpy as np
import pytest

from sctrial.benchmark.calibration import summarize_simulation
from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _cfg(**kw) -> TranscriptomeSimConfig:
    base = dict(
        n_per_arm=6,
        n_genes_transcriptome=3000,
        cells_per_pv_mean=400,
        cells_per_pv_cv=0.5,
        cells_per_pv_min=100,
        cells_per_pv_max=900,
        # Every empirical resampling path is OFF. These tests validate an
        # ESTIMATOR by generating at a known parameter and requiring recovery, so
        # the generating value has to be a parameter, not a resampled pool.
        use_empirical_library=False,
        use_empirical_cells_per_pv=False,
        use_empirical_gene_rates=False,
        use_empirical_dispersion=False,
        dispersion_median=0.30,
        dispersion_mean_slope=0.0,
        dispersion_anchor=0.0,
        dispersion_residual_sd=0.5,
        between_participant_sd=1.0,
        prepost_corr=0.5,
        seed=1,
    )
    base.update(kw)
    return TranscriptomeSimConfig(**base)


@pytest.mark.parametrize("alpha", [0.10, 0.30, 0.80])
def test_conditional_dispersion_recovers_generating_value(alpha):
    """The pooled within-stratum moment estimator must be unbiased for alpha."""
    stats = summarize_simulation(_cfg(dispersion_median=alpha)).statistics()
    est = stats["cond_alpha_median"]
    assert est == pytest.approx(alpha, rel=0.20), (
        f"generating alpha={alpha} recovered as {est:.4f}. A biased dispersion "
        "estimator miscalibrates the simulator invisibly."
    )


def test_variance_components_recover_latent_hierarchy():
    """sigma_b and sigma_u must be recovered, not their noise-attenuated shadow."""
    cfg = _cfg(between_participant_sd=1.0, prepost_corr=0.5)
    stats = summarize_simulation(cfg).statistics()

    assert stats["between_participant_sd_latent"] == pytest.approx(1.0, rel=0.15)
    assert stats["prepost_corr_latent"] == pytest.approx(0.5, abs=0.10)
    assert stats["sigma_e_pseudobulk"] > 0, (
        "split-half estimate of pseudobulk sampling noise collapsed to zero; "
        "without it sigma_u and sigma_e are not separately identifiable"
    )


def test_observable_correlation_is_attenuated_relative_to_latent():
    """Pin the distinction that caused the error, so it cannot silently reverse.

    The observable pre/post correlation MUST be lower than the latent one. If a
    future change makes them equal, the split-half noise correction has stopped
    working and the config would again be parameterised from the observable.
    """
    stats = summarize_simulation(_cfg(prepost_corr=0.5)).statistics()
    assert stats["prepost_corr_pooled"] < stats["prepost_corr_latent"], (
        "observable correlation is not attenuated relative to the latent value; "
        "the sampling-noise correction is inert"
    )


def test_dispersion_estimator_is_inflated_by_pooling_heterogeneous_groups():
    """Conditioning on the homogeneity stratum is not optional.

    Two populations with different mean profiles, pooled, must produce a larger
    apparent dispersion than either alone. This is the mechanism behind TNBC's
    0.774-versus-0.275, and it is why the generating parameter is the
    within-cell-type value.
    """
    from sctrial.benchmark.calibration import SummaryAccumulator

    rng = np.random.default_rng(0)
    n_genes, n_cells = 200, 400
    # Two "cell types" with genuinely different mean profiles, each Poisson (so
    # any measured overdispersion is pooling, not cell-level dispersion).
    mu_a = rng.lognormal(2.0, 1.0, size=n_genes)
    mu_b = mu_a * np.exp(rng.normal(0.0, 1.5, size=n_genes))
    ca = rng.poisson(mu_a[None, :], size=(n_cells, n_genes))
    cb = rng.poisson(mu_b[None, :], size=(n_cells, n_genes))

    split = SummaryAccumulator(n_genes=n_genes, min_strata_detected=1)
    split.add_block(ca, "P1", "Pre", stratum="P1|Pre|A")
    split.add_block(cb, "P1", "Pre", stratum="P1|Pre|B")
    pooled = SummaryAccumulator(n_genes=n_genes, min_strata_detected=1)
    pooled.add_block(np.vstack([ca, cb]), "P1", "Pre", stratum="P1|Pre")

    a_split = np.nanmedian(split.conditional_alpha()[0])
    a_pooled = np.nanmedian(pooled.conditional_alpha()[0])
    assert a_pooled > a_split, (
        f"pooling two distinct populations gave alpha={a_pooled:.3f}, not larger "
        f"than the stratified {a_split:.3f} -- the conditioning has no effect, so "
        "the estimator would not have detected the TNBC cell-type artifact"
    )


def test_variance_components_are_blind_to_design_effects():
    """A real treatment effect must NOT inflate the nuisance hierarchy.

    If arm, visit and arm-by-visit effects were absorbed into sigma_b or sigma_u,
    a genuine therapy-associated shift would be simulated as random temporal
    variability, and the simulated null would be harder than the real nuisance
    process is. Every method's Type I error would then be measured against a
    conservative straw man.

    Injecting a large arm-by-visit effect must leave the estimated components
    unchanged. The estimator centres the change score and the participant mean
    WITHIN arm, which removes the visit effect, the arm effect and their
    interaction together.
    """
    null = summarize_simulation(_cfg(effects={})).statistics()

    # A large interaction on 20% of a well-expressed subset, plus a common time
    # effect: exactly the design structure the estimator must be blind to.
    treated = _cfg(
        effects={f"gene_{i}": 1.5 for i in range(0, 600, 3)},
        time_effect=0.8,
    )
    with_effect = summarize_simulation(treated).statistics()

    for key in ("sigma_b_latent", "sigma_u_latent", "between_participant_sd_latent"):
        a, b = null[key], with_effect[key]
        assert b == pytest.approx(a, rel=0.15), (
            f"{key} moved from {a:.4f} to {b:.4f} when a treatment effect was "
            "injected; design effects are leaking into the nuisance hierarchy"
        )


def test_distributional_gate_is_calibrated_and_sensitive():
    """The distributional gate must pass identical distributions and catch shifts.

    A gate calibrated too tightly rejects a correct simulator; too loosely it
    certifies a wrong one. Both failure modes occurred here. The reference was
    built from distances to a CENTROID while the observed distance was measured
    the same way -- but every realisation contributes to the centroid and the
    observed one does not, so two independent samples from the IDENTICAL
    distribution scored just outside the floor (0.0099 against 0.0084). With a
    bootstrap reference the asymmetry ran the other way and biased toward PASS.

    Both sides are now pairwise distances between independent realisations.
    """
    from sctrial.benchmark.gates import _distribution_gate

    rng = np.random.default_rng(0)

    def grid(x):
        return np.percentile(x, np.linspace(1, 99, 99)).tolist()

    sims = [{"_corr_quantiles": grid(rng.normal(0, 0.3, 4000))} for _ in range(30)]

    same = _distribution_gate({"corr_quantiles": grid(rng.normal(0, 0.3, 4000))}, sims)[0]
    assert same["verdict"] == "PASS", (
        f"identical distributions scored {same['verdict']} (W1 {same['observed']:.4f} "
        f"against a floor of {same['sim_hi95']:.4f}); the gate would reject a "
        "correctly calibrated simulator"
    )

    shifted = _distribution_gate({"corr_quantiles": grid(rng.normal(0.05, 0.3, 4000))}, sims)[0]
    assert shifted["verdict"] == "FAIL", (
        "a 0.05 location shift was not detected; the gate is too permissive"
    )
