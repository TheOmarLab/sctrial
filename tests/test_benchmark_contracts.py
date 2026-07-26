"""Gate F/G: every method gets the input its estimand assumes, and is scored on it.

These are seam tests. Each defect they pin was invisible in isolation -- both
halves of the seam were individually correct -- and each one reached the
manuscript:

* every method normalised against the 50-2000 gene **panel** rather than the
  transcriptome, so a coordinated signal moved the denominator it was being
  measured against. About two thirds of dreamlet's "empirical-Bayes inflation"
  was that artifact;
* NEBULA received ``log(colSums(panel))`` for an argument that is logged
  internally, double-logging the offset and attenuating library-size adjustment;
* every method was scored against one shared ``true_beta`` although
  ``log(1+CPM)`` models and log-link models estimate different functionals.

A seam has no natural owner, so it needs an explicit test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sctrial.benchmark.contracts import (
    METHOD_ESTIMAND,
    METHOD_INPUT,
    participant_log1p_cpm,
    prepare_inputs,
)
from sctrial.benchmark.orchestrator import CORE_METHODS
from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig, simulate_trial_v2

RUNNERS = Path(__file__).resolve().parent.parent / "src" / "sctrial" / "benchmark" / "runners"
_TEST_GENES = 2500


def _cfg(**kw) -> TranscriptomeSimConfig:
    base = dict(
        n_per_arm=6,
        n_genes_transcriptome=_TEST_GENES,
        # A 2,500-gene test transcriptome yields ~1,100 detectable genes,
        # so the production 2,000-gene panel does not fit. Stated here
        # rather than silently skipped.
        panel_sizes=(50, 200, 500),
        cells_per_pv_fixed=60,
        use_empirical_library=False,
        use_empirical_cells_per_pv=False,
        seed=7,
    )
    base.update(kw)
    return TranscriptomeSimConfig(**base)


@pytest.fixture(scope="module")
def sim():
    return simulate_trial_v2(_cfg())


@pytest.fixture(scope="module")
def inputs(sim):
    return prepare_inputs(sim, sim["panels"][50])


# ---------------------------------------------------------------------------
# Normalisation scope
# ---------------------------------------------------------------------------


def test_cpm_denominator_is_the_transcriptome_not_the_panel(sim):
    """The CPM denominator must be the full transcriptome.

    With a panel denominator the reference moves with the signal: under a
    coordinated effect the panel total shifts and every null gene in the panel
    acquires an offsetting apparent effect.
    """
    panel = sim["panels"][50]
    pb = sim["pseudobulk_counts"]
    out = participant_log1p_cpm(pb, panel, gene_cols=sim["gene_names"])

    mat = pb[sim["gene_names"]].to_numpy(dtype=float)
    expected = np.log1p(mat / mat.sum(axis=1, keepdims=True) * 1e6)
    pos = {g: i for i, g in enumerate(sim["gene_names"])}
    for g in panel[:10]:
        np.testing.assert_allclose(out[g].to_numpy(), expected[:, pos[g]], rtol=1e-10)

    # And it is NOT the panel denominator. Compared on genes that actually carry
    # counts: an all-zero gene is log1p(0) = 0 under any denominator, so it
    # cannot distinguish the two and would make this test pass vacuously.
    panel_mat = pb[panel].to_numpy(dtype=float)
    panel_norm = np.log1p(panel_mat / panel_mat.sum(axis=1, keepdims=True) * 1e6)
    expressed = np.flatnonzero(panel_mat.sum(axis=0) > 0)
    assert expressed.size >= 5, "too few expressed panel genes to test the denominator"
    for j in expressed[:5]:
        assert not np.allclose(out[panel[j]].to_numpy(), panel_norm[:, j]), (
            f"{panel[j]}: normalisation used the panel total; that is the artifact "
            "this contract exists to prevent"
        )


def test_panel_selection_happens_after_normalisation(sim):
    """Selecting the panel must not change any gene's normalised value."""
    pb = sim["pseudobulk_counts"]
    small = participant_log1p_cpm(pb, sim["panels"][50], gene_cols=sim["gene_names"])
    large = participant_log1p_cpm(pb, sim["panels"][500], gene_cols=sim["gene_names"])
    for g in sim["panels"][50][:10]:
        np.testing.assert_allclose(small[g].to_numpy(), large[g].to_numpy(), rtol=1e-12)


def test_lib_size_is_the_full_transcriptome_total(sim, inputs):
    expected = sim["pseudobulk_counts"][sim["gene_names"]].to_numpy(dtype=float).sum(axis=1)
    np.testing.assert_allclose(inputs["lib_size"], expected)
    panel_total = sim["pseudobulk_counts"][sim["panels"][50]].to_numpy(dtype=float).sum(axis=1)
    assert not np.allclose(inputs["lib_size"], panel_total)


def test_cell_lib_size_is_the_full_transcriptome_total(sim, inputs):
    expected = np.asarray(sim["adata"].X.sum(axis=1)).ravel()
    np.testing.assert_allclose(inputs["cell_lib_size"], expected)
    assert inputs["cell_counts"].n_vars == 50, "NEBULA must still be TESTED on the panel"
    assert len(inputs["cell_lib_size"]) == sim["adata"].n_obs


# ---------------------------------------------------------------------------
# sctrial and Wilcoxon must receive the identical outcome
# ---------------------------------------------------------------------------


def test_sctrial_and_wilcoxon_share_one_outcome(inputs):
    """Any difference between them must be the test, not the data.

    They are contrasted in the paper as "the DiD framework versus a rank test on
    the same change scores". If they silently received differently prepared
    outcomes, that contrast would not be about the tests at all.
    """
    assert METHOD_INPUT["sctrial_did"] == METHOD_INPUT["wilcoxon_paired"]
    assert METHOD_ESTIMAND["sctrial_did"] == METHOD_ESTIMAND["wilcoxon_paired"]


# ---------------------------------------------------------------------------
# Count-based runners must refuse a panel-derived library size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", ["dreamlet_runner", "limma_voom", "edger_qlf"])
def test_count_runners_require_explicit_lib_size(module, inputs):
    """Silence is the failure mode here: a default would be a panel library size."""
    import importlib

    mod = importlib.import_module(f"sctrial.benchmark.runners.{module}")
    with pytest.raises(ValueError, match="lib_size"):
        mod.run(inputs["pseudobulk_counts"], inputs["panel_genes"])


def test_nebula_requires_explicit_lib_size(inputs):
    from sctrial.benchmark.runners import nebula_runner

    with pytest.raises(ValueError, match="lib_size"):
        nebula_runner.run(inputs["cell_counts"], inputs["panel_genes"])


@pytest.mark.parametrize("fname", ["dreamlet_runner.py", "limma_voom.py", "edger_qlf.py"])
def test_count_runners_do_not_recompute_lib_size_from_the_panel(fname):
    """``keep.lib.sizes=FALSE`` after filtering silently restores the panel total."""
    src = (RUNNERS / fname).read_text(encoding="utf-8")
    assert "y$samples$lib.size <- meta$lib_size" in src, (
        f"{fname} never sets the supplied library size on the DGEList"
    )
    assert "keep.lib.sizes=FALSE" not in src, (
        f"{fname} recomputes lib.size from the filtered PANEL, discarding the "
        "full-transcriptome value it was given"
    )


def test_nebula_offset_is_linear_scale_and_not_the_panel_sum():
    """nebula logs the offset internally; a logged input is logged twice."""
    src = (RUNNERS / "nebula_runner.py").read_text(encoding="utf-8")
    assert "offset = log(colSums" not in src, (
        "NEBULA offset is double-logged AND derived from the panel"
    )
    assert src.count("offset = meta$lib_size") >= 1
    assert "log(colSums(counts))" not in src


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def test_every_core_method_has_an_estimand_and_an_input():
    for m in CORE_METHODS:
        assert m in METHOD_ESTIMAND, f"{m} has no declared estimand"
        assert m in METHOD_INPUT, f"{m} has no declared input representation"


def test_oracle_is_zero_under_a_true_null():
    """A null scenario must have a null target on BOTH estimand scales."""
    sim = simulate_trial_v2(_cfg(effects={}))
    for scale, table in sim["oracle"].items():
        vals = np.array(list(table.values()))
        assert np.abs(vals).max() < 1e-9, f"oracle {scale} is non-zero under the null"


def test_count_link_oracle_equals_the_injected_effect():
    sim = simulate_trial_v2(_cfg(effects={"gene_0": 0.5, "gene_1": -0.8}))
    assert sim["oracle"]["count_link"]["gene_0"] == pytest.approx(0.5)
    assert sim["oracle"]["count_link"]["gene_1"] == pytest.approx(-0.8)


def test_log1p_cpm_oracle_is_attenuated_and_same_sign():
    """Pin the attenuation, which is the reason two oracles exist.

    ``log(1+CPM)`` shrinks a log-link effect toward zero for low-expression
    genes. Scoring sctrial against the injected beta would report that shrinkage
    as method bias.
    """
    sim = simulate_trial_v2(_cfg(effects={"gene_0": 0.5, "gene_1": -0.8}))
    o = sim["oracle"]["log1p_cpm"]
    for g, injected in (("gene_0", 0.5), ("gene_1", -0.8)):
        assert np.sign(o[g]) == np.sign(injected), f"{g}: oracle flipped sign"
        assert abs(o[g]) <= abs(injected) + 1e-9, (
            f"{g}: log1p oracle {o[g]:.4f} exceeds the injected {injected}; the "
            "transform can only shrink"
        )


def test_log1p_cpm_oracle_approaches_beta_at_high_expression():
    """The two estimands must coincide when CPM >> 1, or the oracle is wrong."""
    # Parametric rates on purpose: this test constructs a high-expression regime
    # to check the quadrature limit, which the empirical proportion vector (drawn
    # from real, mostly low-expression genes) cannot produce.
    cfg = _cfg(
        effects={"gene_0": 0.5},
        use_empirical_gene_rates=False,
        gene_rate_log_mean=-2.0,
        gene_rate_log_sd=0.2,
    )
    sim = simulate_trial_v2(cfg)
    o = sim["oracle"]["log1p_cpm"]["gene_0"]
    assert o == pytest.approx(0.5, rel=0.05), (
        f"at high expression the log1p oracle is {o:.4f}, not ~0.5; the quadrature "
        "or the transform is wrong"
    )


# ---------------------------------------------------------------------------
# Gate G: the conventional pseudobulk comparator
# ---------------------------------------------------------------------------


def test_conventional_pseudobulk_comparator_is_included():
    assert "limma_voom" in CORE_METHODS, (
        "Gate G requires a conventional pseudobulk comparator in the reported set"
    )


def test_limma_models_the_repeated_measure():
    """An unpaired ~arm*visit would treat a participant's two visits as independent."""
    src = (RUNNERS / "limma_voom.py").read_text(encoding="utf-8")
    assert "duplicateCorrelation" in src
    assert "block=meta$participant" in src


def test_limma_does_not_silently_fall_back_to_an_unpaired_model():
    """A fallback would report a different model under the same method name."""
    src = (RUNNERS / "limma_voom.py").read_text(encoding="utf-8")
    two_arm = src.split("_R_SCRIPT_SINGLE_ARM")[0]
    assert "tryCatch" not in two_arm, (
        "the two-arm limma script swallows a duplicateCorrelation failure; the "
        "failure rate is a reported result, not something to paper over"
    )


# ---------------------------------------------------------------------------
# The real-data path must be the SAME path
# ---------------------------------------------------------------------------


def test_real_data_path_reproduces_the_simulated_path_exactly(sim):
    """Permutation and subsampling must hand methods identical representations.

    Those analyses run the same methods on real cohorts. They previously built
    their own pseudobulk and normalised inside the tested panel, so the
    real-data results characterised these methods under a different
    normalisation scope from the simulation used to characterise them -- and
    nothing compared the two. Feeding the same data through both paths and
    requiring bit-identical output is the only check that keeps them together.
    """
    from sctrial.benchmark.contracts import prepare_inputs_from_adata

    panel = sim["panels"][50]
    adata = sim["adata"].copy()
    # Deliberately non-canonical column names: the real cohorts use
    # participant_id / timepoint / response, so the renaming must be exercised.
    adata.obs = adata.obs.rename(
        columns={"participant": "pid", "visit": "tp", "arm": "grp"}
    )

    via_sim = prepare_inputs(sim, panel)
    via_real = prepare_inputs_from_adata(
        adata, panel, participant_col="pid", visit_col="tp", arm_col="grp"
    )

    def _index(df):
        import pandas as pd

        return pd.MultiIndex.from_arrays([df["participant"], df["visit"]])

    a = via_sim["participant_log1p_cpm"].copy()
    a.index = _index(a)
    b = via_real["participant_log1p_cpm"].copy()
    b.index = _index(b)
    b = b.loc[a.index]

    np.testing.assert_allclose(a[panel].to_numpy(), b[panel].to_numpy(), atol=0, rtol=0)

    import pandas as pd

    lib_sim = pd.Series(via_sim["lib_size"], index=a.index)
    lib_real = pd.Series(
        via_real["lib_size"], index=_index(via_real["participant_log1p_cpm"])
    ).loc[a.index]
    np.testing.assert_allclose(lib_sim.to_numpy(), lib_real.to_numpy(), atol=0, rtol=0)
    np.testing.assert_allclose(
        np.sort(via_sim["cell_lib_size"]), np.sort(via_real["cell_lib_size"])
    )


# ---------------------------------------------------------------------------
# The calibration must actually reach the simulator
# ---------------------------------------------------------------------------


def test_frozen_calibration_reaches_the_simulator(monkeypatch):
    """The single most consequential wiring in the benchmark.

    The previous benchmark's headline defect was not a wrong calibration: it was
    a calibration that existed, was described in the Methods, and was never
    threaded through, so every published simulation ran on dataclass defaults at
    2.3e7 UMIs per cell. Nothing failed, nothing warned, and the numbers looked
    plausible.

    This asserts the merge order directly: the frozen configuration is the floor,
    and a scenario may override only the knobs it is explicitly varying.
    """
    from sctrial.benchmark import orchestrator

    captured = {}

    def _spy(cfg):
        captured["cfg"] = cfg
        raise RuntimeError("stop after config construction")

    monkeypatch.setattr(orchestrator, "simulate_trial_v2", _spy)

    base = {
        "n_genes_transcriptome": _TEST_GENES,
        "panel_sizes": (50, 200, 500),
        "use_empirical_library": False,
        "use_empirical_cells_per_pv": False,
        # Deliberately unlike the dataclass default, so "the default happened to
        # match" cannot make this pass.
        "dispersion_median": 0.9137,
        "between_participant_sd": 1.234,
        "prepost_corr": 0.321,
    }
    scenario = {
        "name": "t",
        "description": "t",
        "panel_size": 50,
        "signal_fraction": 0.0,
        "architecture": "balanced",
        "magnitude": 0.5,
        # The scenario varies participant SD; everything else must come from base.
        "config_kwargs": {
            "design": "two_arm",
            "n_per_arm": 6,
            "cells_per_pv_fixed": 40,
            "between_participant_sd": 2.5,
        },
    }

    with pytest.raises(RuntimeError, match="stop after config construction"):
        orchestrator._run_single_iteration(("t", 0, 5, scenario, ["sctrial_did"], base))

    cfg = captured["cfg"]
    assert cfg.dispersion_median == pytest.approx(0.9137), (
        "the frozen calibration did not reach the simulator; this is exactly the "
        "defect that produced 2.3e7 UMIs per cell under a Methods section "
        "describing calibrated parameters"
    )
    assert cfg.prepost_corr == pytest.approx(0.321)
    assert cfg.between_participant_sd == pytest.approx(2.5), (
        "the scenario must win for the knob it is explicitly varying"
    )
    assert cfg.cells_per_pv_fixed == 40


def test_no_base_config_falls_back_to_dataclass_defaults():
    """Document the fallback so the guard above is unambiguous.

    ``base_config=None`` is legitimate only for tests. The production driver
    refuses to start without a frozen configuration, which is asserted here so
    that guarantee is not quietly removed later.
    """
    driver = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_benchmark.py"
    ).read_text(encoding="utf-8")
    assert "_load_frozen_config()" in driver
    assert "There is deliberately no default fallback" in driver
    assert driver.count("base_config=frozen") >= 2, (
        "a benchmark phase runs without the frozen calibration"
    )


def test_every_reported_method_has_a_plotting_style():
    """A method added to CORE_METHODS must not vanish from the figures.

    The style dicts were hand-maintained copies in Figure 3 AND Supp Fig 5, so a
    new method appeared in neither, or in one and not the other, with both files
    looking internally consistent. Figure 3 now derives the list from
    CORE_METHODS and raises on a missing style; Supp Fig 5 imports it.
    """
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parent.parent
    fig3 = (root / "manuscript_figures" / "main"
            / "figure3_robustness_benchmarking.py").read_text(encoding="utf-8")
    supp = (root / "manuscript_figures" / "supp"
            / "supp_fig5_sensitivity_robustness.py").read_text(encoding="utf-8")

    assert "from sctrial.benchmark.orchestrator import CORE_METHODS" in fig3, (
        "Figure 3 restates the method list instead of deriving it"
    )
    assert "_BENCH_METHODS = [" not in supp, (
        "Supp Fig 5 keeps its own copy of the method list; it must import Figure 3's"
    )
    for m in CORE_METHODS:
        assert f'"{m}"' in fig3, f"{m} has no plotting style in Figure 3"
    sys.modules.pop("manuscript_figures", None)


# ---------------------------------------------------------------------------
# Panel eligibility must not depend on the injected signal
# ---------------------------------------------------------------------------


def test_eligibility_is_independent_of_the_injected_signal():
    """A gene must not become testable BECAUSE it is differential.

    If eligibility were computed after injecting the effect, signal genes would be
    preferentially admitted at the margin and every power estimate would be
    optimistic by an amount nobody could see.
    """
    from sctrial.benchmark.simulator_v2 import eligible_panel_genes, nested_panels

    base = _cfg(effects={})
    strong = _cfg(effects={f"gene_{i}": 3.0 for i in range(200)})

    np.testing.assert_array_equal(
        eligible_panel_genes(base), eligible_panel_genes(strong)
    )
    p0 = nested_panels(base, rng=np.random.default_rng(1))
    p1 = nested_panels(strong, rng=np.random.default_rng(1))
    for size in p0:
        assert p0[size] == p1[size], f"panel {size} moved when a signal was injected"


def test_all_methods_receive_the_same_panel():
    """Comparability requires an identical denominator across methods."""
    from sctrial.benchmark.simulator_v2 import simulate_trial_v2

    sim = simulate_trial_v2(_cfg())
    panel = sim["panels"][50]
    inputs = prepare_inputs(sim, panel)
    assert inputs["panel_genes"] == panel
    assert list(inputs["pseudobulk_counts"].columns[3:]) == panel
    assert list(inputs["cell_counts"].var_names) == panel
    outcome_genes = [c for c in inputs["participant_log1p_cpm"].columns
                     if c not in ("participant", "visit", "arm")]
    assert outcome_genes == panel


def test_evaluability_is_recorded_not_silently_dropped():
    """A method's own filtering must show up as a rate, not shrink the denominator."""
    from sctrial.benchmark.orchestrator import _run_single_iteration

    scenario = {
        "name": "t", "description": "t", "panel_size": 50, "signal_fraction": 0.0,
        "architecture": "balanced", "magnitude": 0.5,
        "config_kwargs": {
            "design": "two_arm", "n_per_arm": 6, "cells_per_pv_fixed": 40,
            "n_genes_transcriptome": _TEST_GENES, "panel_sizes": (50, 200, 500),
            "use_empirical_library": False, "use_empirical_cells_per_pv": False,
            "use_empirical_gene_rates": False, "use_empirical_dispersion": False,
        },
    }
    rows = _run_single_iteration(("t", 0, 3, scenario, ["sctrial_did"], None))
    assert len(rows) == 50, "the denominator must be the full panel"
    assert all("evaluable" in r for r in rows)


# ---------------------------------------------------------------------------
# Freeze-level: normalisation must not follow the panel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel_size", [50, 200, 500])
def test_library_size_is_invariant_to_panel_size(sim, panel_size):
    """The normalisation reference must be identical for every nested panel.

    This is the artifact Gate D proved: with a panel-scoped denominator the bias
    swings from +0.094 to -0.076 purely with signal architecture, while a
    full-transcriptome denominator holds at ~-0.01 throughout. If TMM or voom
    recomputed norm factors after the matrix were cut down to the tested panel,
    that artifact would come straight back and no downstream check would see it.

    Freeze-level assertion: lib_size is computed BEFORE panel selection and does
    not move as the panel grows from 50 to 2,000 genes.
    """
    ref = prepare_inputs(sim, sim["panels"][50])["lib_size"]
    got = prepare_inputs(sim, sim["panels"][panel_size])["lib_size"]
    np.testing.assert_allclose(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("panel_size", [50, 200, 500])
def test_cell_library_size_is_invariant_to_panel_size(sim, panel_size):
    """Same guarantee for NEBULA's per-cell offset."""
    ref = prepare_inputs(sim, sim["panels"][50])["cell_lib_size"]
    got = prepare_inputs(sim, sim["panels"][panel_size])["cell_lib_size"]
    np.testing.assert_allclose(got, ref, atol=0, rtol=0)


def test_count_runners_keep_the_supplied_library_size_through_filtering(sim):
    """filterByExpr must not be allowed to recompute lib.size from the panel."""
    for fname in ("dreamlet_runner.py", "limma_voom.py", "edger_qlf.py"):
        src = (RUNNERS / fname).read_text(encoding="utf-8")
        assert "y$samples$lib.size <- meta$lib_size" in src
        assert "keep.lib.sizes=TRUE" in src
        assert "keep.lib.sizes=FALSE" not in src, (
            f"{fname} lets filtering recompute the library size from the tested "
            "panel, which reinstates the normalisation-scope artifact"
        )


def test_limma_follows_the_canonical_repeated_measures_order():
    """The voom / duplicateCorrelation sequence must be in the prescribed order.

    The limma workflow for repeated measures is: normalise, voom, estimate the
    within-block correlation, re-voom WITH that correlation, then fit with it.
    Skipping the second voom, or fitting without the block, silently reverts to
    treating a participant's two visits as independent -- which is the
    pseudoreplication this paper is about, committed by the comparator.

    Also guards two mixups the limma maintainers call out: combining a logCPM /
    limma-trend pipeline with voom, and putting participant in the design as a
    fixed effect while ALSO blocking on it.
    """
    src = (RUNNERS / "limma_voom.py").read_text(encoding="utf-8")
    two_arm = src.split("_R_SCRIPT_SINGLE_ARM")[0]

    order = [
        "y$samples$lib.size <- meta$lib_size",
        "keep.lib.sizes=TRUE",
        "calcNormFactors(y)",
        "v0 <- voom(y, design)",
        "corfit <- duplicateCorrelation(v0, design, block=meta$participant)",
        "v <- voom(y, design, block=meta$participant, correlation=corfit$consensus)",
        "fit <- lmFit(v, design, block=meta$participant, correlation=corfit$consensus)",
        "eBayes(fit)",
    ]
    pos = -1
    for step in order:
        i = two_arm.find(step)
        assert i > pos, f"limma step out of order or missing: {step!r}"
        pos = i

    assert "trend=TRUE" not in two_arm, "voom and limma-trend must not be combined"
    assert "~arm * visit" in two_arm
    assert "participant +" not in two_arm, (
        "participant appears as a fixed effect AND as a duplicateCorrelation block"
    )
