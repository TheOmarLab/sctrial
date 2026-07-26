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
    cfg = _cfg(effects={"gene_0": 0.5}, gene_rate_log_mean=-2.0, gene_rate_log_sd=0.2)
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
