"""A real cohort reaches the benchmark runners in the vocabulary they read.

The runners read the columns ``participant``, ``visit`` and ``arm`` and the labels
``Pre``/``Post`` and ``Treated``/``Control``. The real-data path renamed the columns
of a cohort and carried its labels, so on a cohort labeled ``Responder`` and
``Non-responder`` no row matched either arm and every method failed for every gene.
It also read ``adata.X`` when the named counts layer was absent, which hands a
normalised matrix to the count models of a cohort without counts.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sctrial.benchmark.contracts import (
    CANONICAL_ARMS,
    CANONICAL_VISITS,
    canonical_cohort,
    prepare_inputs,
    prepare_inputs_from_adata,
    require_raw_counts,
)
from sctrial.benchmark.permutation import _permute_visits, run_permutation_test
from sctrial.benchmark.simulator_v2 import TranscriptomeSimConfig, simulate_trial_v2
from sctrial.benchmark.subsample import run_subsampling

PYTHON_METHODS = ["sctrial_did", "wilcoxon_paired"]
REAL = dict(participant_col="participant_id", visit_col="timepoint", visits=("T0", "T1"),
            arm_col="response", arm_treated="Responder", arm_control="Non-responder")
N_GENES = 40


def _cohort(arms=("Responder", "Non-responder"), visits=("T0", "T1"), *, n_per_arm=5, n_cells=20,
            seed=0) -> ad.AnnData:
    """Counts from a seeded Poisson; a unit-test fixture with a cohort's own labels."""
    rng = np.random.default_rng(seed)
    rows, counts = [], []
    for a, arm in enumerate(arms):
        for i in range(n_per_arm):
            base = rng.gamma(2.0, 2.0, size=N_GENES)
            for visit in visits:
                for _ in range(n_cells):
                    rows.append({"participant_id": f"a{a}p{i}", "response": arm, "timepoint": visit})
                    counts.append(rng.poisson(base))
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    x = np.asarray(counts, dtype=float)
    adata = ad.AnnData(X=np.log1p(x), obs=obs)
    adata.var_names = [f"g{j}" for j in range(N_GENES)]
    adata.layers["counts"] = x
    return adata


def _relabel(adata: ad.AnnData, column: str, mapping: dict) -> ad.AnnData:
    out = adata.copy()
    out.obs[column] = out.obs[column].map(mapping)
    return out


# ── canonical_cohort ──


def test_canonical_cohort_maps_columns_labels_and_counts():
    adata = _cohort(visits=("T0", "T1", "T2"))
    cohort = canonical_cohort(adata, **REAL)

    assert list(cohort.obs.columns) == ["participant", "visit", "arm"]
    kept = adata.obs["timepoint"].isin(["T0", "T1"]).to_numpy()
    assert cohort.n_obs == int(kept.sum()) and list(cohort.obs_names) == list(adata.obs_names[kept])
    assert list(cohort.var_names) == list(adata.var_names)
    source = adata.obs[kept]
    assert cohort.obs["visit"].tolist() == source["timepoint"].map({"T0": "Pre", "T1": "Post"}).tolist()
    assert cohort.obs["arm"].tolist() == source["response"].map(
        {"Responder": "Treated", "Non-responder": "Control"}).tolist()
    assert cohort.obs["participant"].tolist() == source["participant_id"].tolist()
    np.testing.assert_array_equal(np.asarray(cohort.X), adata.layers["counts"][kept])
    assert set(cohort.obs["visit"]) == set(CANONICAL_VISITS)
    assert set(cohort.obs["arm"]) == set(CANONICAL_ARMS)


def test_canonical_cohort_leaves_out_a_third_arm_with_one_warning():
    adata = _cohort(arms=("Responder", "Non-responder", "Unknown"))
    with pytest.warns(UserWarning, match="excluded from the two-arm analysis") as caught:
        cohort = canonical_cohort(adata, **REAL)
    assert len(caught) == 1 and "'Unknown' (5)" in str(caught[0].message)
    assert cohort.obs["participant"].nunique() == 10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reference = canonical_cohort(adata[adata.obs["response"] != "Unknown"].copy(), **REAL)
    pd.testing.assert_frame_equal(cohort.obs, reference.obs)
    np.testing.assert_array_equal(np.asarray(cohort.X), np.asarray(reference.X))


def test_canonical_cohort_single_arm_has_one_arm_label():
    adata = _cohort()
    single = {k: v for k, v in REAL.items() if not k.startswith("arm")}
    cohort = canonical_cohort(adata, **single)
    assert set(cohort.obs["arm"]) == {CANONICAL_ARMS[0]} and cohort.n_obs == adata.n_obs


@pytest.mark.parametrize("change,error,match", [
    (dict(visits=("T0", "Week4")), ValueError, r"No rows of visit 'Week4'.*labels present: \['T0', 'T1'\]"),
    (dict(visits=("T0", "T0")), ValueError, "two different labels"),
    (dict(arm_treated=None), ValueError, "needs arm_treated and arm_control"),
    (dict(arm_control="Control"), ValueError, r"No rows of arm 'Control'.*'Non-responder'"),
    (dict(arm_control="Responder"), ValueError, "two distinct arm labels"),
    (dict(counts_layer="raw"), KeyError, r"counts layer 'raw' not found \(layers: \['counts'\]\)"),
    (dict(arm_col="group"), KeyError, "group"),
])
def test_canonical_cohort_refuses_a_cohort_it_cannot_map(change, error, match):
    with pytest.raises(error, match=match):
        canonical_cohort(_cohort(), **{**REAL, **change})


def test_canonical_cohort_refuses_a_cohort_with_one_arm():
    adata = _cohort(arms=("Responder",))
    with pytest.raises(ValueError, match="No rows of arm 'Non-responder'"):
        canonical_cohort(adata, **REAL)


def test_canonical_cohort_reads_sparse_counts_and_categorical_columns():
    """The usual form of a real AnnData: sparse counts and Categorical obs columns."""
    from scipy import sparse

    dense = _cohort(arms=("Responder", "Non-responder", "Unknown"), visits=("T0", "T1", "T2"))
    real = dense.copy()
    real.layers["counts"] = sparse.csr_matrix(real.layers["counts"])
    for col in ("participant_id", "timepoint"):
        real.obs[col] = pd.Categorical(real.obs[col])
    real.obs["response"] = pd.Categorical(
        real.obs["response"], categories=["Non-responder", "Responder", "Unknown", "Not evaluable"])

    with pytest.warns(UserWarning, match=r"'Unknown' \(5\)") as caught:
        cohort = canonical_cohort(real, **REAL)
    assert "Not evaluable" not in str(caught[0].message)
    with pytest.warns(UserWarning, match="excluded"):
        reference = canonical_cohort(dense, **REAL)
    pd.testing.assert_frame_equal(cohort.obs, reference.obs)
    assert sparse.issparse(cohort.X)
    np.testing.assert_array_equal(cohort.X.toarray(), np.asarray(reference.X))

    require_raw_counts(["dreamlet", "nebula"], cohort)
    non_counts = cohort.copy()
    non_counts.X = non_counts.X * 0.5
    with pytest.raises(ValueError, match="model raw counts"):
        require_raw_counts(["dreamlet"], non_counts)

    genes = list(real.var_names[:10])
    from_sparse = prepare_inputs_from_adata(cohort, genes, counts_layer=None)
    from_dense = prepare_inputs_from_adata(reference, genes, counts_layer=None)
    pd.testing.assert_frame_equal(from_sparse["participant_log1p_cpm"], from_dense["participant_log1p_cpm"])
    np.testing.assert_array_equal(from_sparse["lib_size"], from_dense["lib_size"])


# ── the per-iteration contract refuses what the runners cannot read ──


def test_prepare_inputs_refuses_labels_the_runners_do_not_read():
    adata = _cohort(visits=("Pre", "Post"))
    genes = list(adata.var_names[:10])
    kw = dict(participant_col="participant_id", visit_col="timepoint", arm_col="response")
    with pytest.raises(ValueError, match=r"labels \['Non-responder', 'Responder'\].*canonical_cohort"):
        prepare_inputs_from_adata(adata, genes, **kw)

    canonical = _relabel(adata, "response", {"Responder": "Treated", "Non-responder": "Control"})
    with pytest.raises(ValueError, match=r"obs\['timepoint'\] holds labels \['T0', 'T1'\]"):
        prepare_inputs_from_adata(_relabel(canonical, "timepoint", {"Pre": "T0", "Post": "T1"}),
                                  genes, **kw)
    unlabeled = canonical.copy()
    unlabeled.obs.loc[unlabeled.obs.index[:3], "response"] = None
    with pytest.raises(ValueError, match="3 missing labels"):
        prepare_inputs_from_adata(unlabeled, genes, **kw)

    out = prepare_inputs_from_adata(canonical, genes, **kw)
    assert set(out["participant_log1p_cpm"]["arm"]) == set(CANONICAL_ARMS)


def test_prepare_inputs_does_not_fall_back_to_x_for_a_missing_counts_layer():
    adata = _cohort(arms=CANONICAL_ARMS, visits=CANONICAL_VISITS)
    genes = list(adata.var_names[:10])
    kw = dict(participant_col="participant_id", visit_col="timepoint", arm_col="response")
    with pytest.raises(KeyError, match="counts layer 'umi' not found"):
        prepare_inputs_from_adata(adata, genes, counts_layer="umi", **kw)

    from_layer = prepare_inputs_from_adata(adata, genes, counts_layer="counts", **kw)
    in_x = adata.copy()
    in_x.X = in_x.layers["counts"].copy()
    del in_x.layers["counts"]
    from_x = prepare_inputs_from_adata(in_x, genes, counts_layer=None, **kw)
    pd.testing.assert_frame_equal(from_layer["participant_log1p_cpm"], from_x["participant_log1p_cpm"])
    np.testing.assert_array_equal(from_layer["lib_size"], from_x["lib_size"])


def test_count_models_refuse_a_matrix_that_does_not_hold_counts():
    adata = _cohort()
    adata.layers["tpm"] = adata.layers["counts"] / adata.layers["counts"].sum(axis=1, keepdims=True) * 1e6
    tpm = canonical_cohort(adata, **REAL, counts_layer="tpm")
    counts = canonical_cohort(adata, **REAL)

    require_raw_counts(PYTHON_METHODS, tpm)
    require_raw_counts(["dreamlet", "limma_voom", "nebula", *PYTHON_METHODS], counts)
    for method in ("dreamlet", "limma_voom", "edger_qlf", "nebula"):
        with pytest.raises(ValueError, match="model raw counts"):
            require_raw_counts([method], tpm)

    genes = list(adata.var_names[:5])
    with pytest.raises(ValueError, match=r"\['nebula'\] model raw counts"):
        run_permutation_test(adata, genes, methods=["sctrial_did", "nebula"], n_permutations=1,
                             counts_layer="tpm", **REAL)
    with pytest.raises(ValueError, match=r"\['dreamlet'\] model raw counts"):
        run_subsampling(adata, genes, methods=["dreamlet"], n_resamples=1, counts_layer="tpm", **REAL)


# ── end to end on the runners that need no R ──


def test_permutation_on_real_labels_equals_the_run_on_canonical_labels():
    adata = _cohort()
    genes = list(adata.var_names[:12])
    result = run_permutation_test(adata, genes, methods=PYTHON_METHODS, n_permutations=4, seed=3,
                                  **REAL)
    assert len(result) == 4 * len(PYTHON_METHODS) * len(genes)
    assert np.isfinite(result["pvalue"]).all()

    canonical = _relabel(_relabel(adata, "response", {"Responder": "Treated", "Non-responder": "Control"}),
                         "timepoint", {"T0": "Pre", "T1": "Post"})
    reference = run_permutation_test(
        canonical, genes, methods=PYTHON_METHODS, n_permutations=4, seed=3,
        participant_col="participant_id", visit_col="timepoint", visits=("Pre", "Post"),
        arm_col="response", arm_treated="Treated", arm_control="Control")
    pd.testing.assert_frame_equal(result, reference)


def test_permutation_names_the_arms_explicitly():
    adata = _cohort()
    genes = list(adata.var_names[:5])
    without_arms = {k: v for k, v in REAL.items() if k not in ("arm_treated", "arm_control")}
    with pytest.raises(ValueError, match="needs arm_treated and arm_control"):
        run_permutation_test(adata, genes, methods=PYTHON_METHODS, n_permutations=1, **without_arms)
    with pytest.raises(TypeError, match="visits"):
        run_permutation_test(adata, genes, methods=PYTHON_METHODS, n_permutations=1,
                             participant_col="participant_id", visit_col="timepoint",
                             arm_col="response", arm_treated="Responder",
                             arm_control="Non-responder")
    with pytest.raises(ValueError, match="design_type"):
        run_permutation_test(adata, genes, methods=PYTHON_METHODS, n_permutations=1,
                             design_type="paired", **REAL)


def test_single_arm_permutation_ignores_the_arm_column():
    adata = _cohort()
    genes = list(adata.var_names[:8])
    result = run_permutation_test(adata, genes, design_type="single_arm", methods=PYTHON_METHODS,
                                  n_permutations=3, seed=1, participant_col="participant_id",
                                  visit_col="timepoint", visits=("T0", "T1"), arm_col="response")
    assert np.isfinite(result["pvalue"]).all()
    assert len(result) == 3 * len(PYTHON_METHODS) * len(genes)


def test_visit_permutation_keeps_every_sample_whole():
    cohort = canonical_cohort(_cohort(), **REAL)
    sample = cohort.obs["participant"].astype(str) + "/" + cohort.obs["visit"].astype(str)
    kept, exchanged = set(), set()
    for seed in range(40):
        permuted = _permute_visits(cohort, "participant", "visit", np.random.default_rng(seed))
        assert permuted.obs["visit"].dtype == cohort.obs["visit"].dtype
        labels = permuted.obs["visit"].astype(str)
        # one label per original sample, and the two samples of a participant keep two labels
        assert labels.groupby(sample.to_numpy()).nunique().eq(1).all()
        assert labels.groupby(cohort.obs["participant"].to_numpy()).nunique().eq(2).all()
        same = (labels == cohort.obs["visit"].astype(str)).groupby(cohort.obs["participant"].to_numpy()).all()
        kept.update(same.index[same])
        exchanged.update(same.index[~same])
    participants = set(cohort.obs["participant"])
    assert kept == participants and exchanged == participants


def test_visit_permutation_leaves_a_missing_visit_label_missing():
    cohort = canonical_cohort(_cohort(), **REAL)
    unlabeled = cohort.obs["participant"] == cohort.obs["participant"].iloc[0]
    cohort.obs["visit"] = cohort.obs["visit"].astype(object)
    cohort.obs.loc[unlabeled, "visit"] = np.nan
    for seed in range(8):
        permuted = _permute_visits(cohort, "participant", "visit", np.random.default_rng(seed))
        assert permuted.obs.loc[unlabeled, "visit"].isna().all()
        assert permuted.obs.loc[~unlabeled, "visit"].notna().all()


def test_visit_permutation_needs_two_visit_labels():
    cohort = canonical_cohort(_cohort(), **REAL)
    one_visit = cohort[cohort.obs["visit"] == "Pre"].copy()
    with pytest.raises(ValueError, match="two visit labels"):
        _permute_visits(one_visit, "participant", "visit", np.random.default_rng(0))


def test_subsampling_on_real_labels_gives_results():
    adata = _cohort(n_per_arm=8)
    genes = list(adata.var_names[:30])
    result = run_subsampling(adata, genes, methods=PYTHON_METHODS, fractions=[0.75], n_resamples=4,
                             seed=2, **REAL)
    assert len(result) == 4 * len(PYTHON_METHODS)
    assert (result["n_participants"] == 12).all()
    assert np.isfinite(result["spearman_rho"]).all()
    assert result["spearman_rho"].between(-1, 1).all() and result["spearman_rho"].nunique() > 1
    assert np.isfinite(result["jaccard_top20"]).all()


def test_counts_in_x_are_named_with_counts_layer_none():
    adata = _cohort()
    genes = list(adata.var_names[:8])
    in_x = adata.copy()
    in_x.X = in_x.layers["counts"].copy()
    del in_x.layers["counts"]
    kw = dict(methods=PYTHON_METHODS, n_permutations=2, seed=5, **REAL)
    with pytest.raises(KeyError, match="counts layer 'counts' not found"):
        run_permutation_test(in_x, genes, **kw)
    pd.testing.assert_frame_equal(run_permutation_test(in_x, genes, counts_layer=None, **kw),
                                  run_permutation_test(adata, genes, **kw))


def test_a_resample_that_leaves_too_few_participants_in_an_arm_is_a_missing_result():
    adata = _cohort(n_per_arm=4)
    genes = list(adata.var_names[:30])
    result = run_subsampling(adata, genes, methods=["sctrial_did"], fractions=[0.5],
                             n_resamples=12, seed=0, **REAL)
    assert len(result) == 12 and (result["n_participants"] == 4).all()
    assert result["spearman_rho"].isna().any() and result["spearman_rho"].notna().any()


# ── the mapped cohort is the simulated path ──


def test_a_relabeled_simulated_cohort_gives_the_inputs_of_the_simulated_path():
    """Labels of a real cohort, mapped by canonical_cohort, change nothing downstream."""
    sim = simulate_trial_v2(TranscriptomeSimConfig(
        n_per_arm=6, n_genes_transcriptome=2500, panel_sizes=(50,), cells_per_pv_fixed=40,
        use_empirical_library=False, use_empirical_cells_per_pv=False, seed=11))
    panel = sim["panels"][50]
    adata = sim["adata"].copy()
    adata.obs = adata.obs.rename(columns={"participant": "pid", "visit": "tp", "arm": "grp"})
    adata.obs["grp"] = adata.obs["grp"].astype(str).map({"Treated": "Drug", "Control": "Placebo"})
    adata.obs["tp"] = adata.obs["tp"].astype(str).map({"Pre": "Baseline", "Post": "Week 12"})

    cohort = canonical_cohort(adata, participant_col="pid", visit_col="tp",
                              visits=("Baseline", "Week 12"), arm_col="grp", arm_treated="Drug",
                              arm_control="Placebo", counts_layer=None)
    via_real = prepare_inputs_from_adata(cohort, panel, counts_layer=None)
    via_sim = prepare_inputs(sim, panel)

    def _keyed(inputs, name):
        frame = inputs[name].copy()
        frame.index = pd.MultiIndex.from_arrays([frame["participant"].astype(str), frame["visit"]])
        return frame.sort_index()

    a, b = _keyed(via_sim, "participant_log1p_cpm"), _keyed(via_real, "participant_log1p_cpm")
    assert a.index.equals(b.index)
    assert a["arm"].astype(str).tolist() == b["arm"].astype(str).tolist()
    np.testing.assert_allclose(a[panel].to_numpy(), b[panel].to_numpy(), atol=0, rtol=0)
    counts_a, counts_b = _keyed(via_sim, "pseudobulk_counts"), _keyed(via_real, "pseudobulk_counts")
    np.testing.assert_array_equal(counts_a[panel].to_numpy(), counts_b[panel].to_numpy())
    np.testing.assert_allclose(np.sort(via_sim["lib_size"]), np.sort(via_real["lib_size"]),
                               atol=0, rtol=0)
    np.testing.assert_allclose(np.sort(via_sim["cell_lib_size"]), np.sort(via_real["cell_lib_size"]),
                               atol=0, rtol=0)
