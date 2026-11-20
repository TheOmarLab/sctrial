import types
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial.stats.gsea as gsea
from sctrial import TrialDesign


def _dummy_prerank(rnk, gene_sets, **kwargs):
    res = pd.DataFrame(
        {
            "gene_set": list(gene_sets.keys()) if isinstance(gene_sets, dict) else ["GS1"],
            "NES": [1.0],
            "FDR q-val": [0.1],
        }
    )
    return types.SimpleNamespace(res2d=res)


def _three_arm_adata() -> AnnData:
    """Six cells: two visits by three arms, so the wrappers have rows to leave out."""
    adata = AnnData(X=np.random.default_rng(0).random((6, 2)))
    adata.var_names = ["G1", "G2"]
    adata.obs["visit"] = ["V1", "V2"] * 3
    adata.obs["arm"] = ["Treated", "Treated", "Control", "Control", "Other", "Other"]
    adata.obs["participant_id"] = ["p1", "p1", "p2", "p2", "p3", "p3"]
    adata.obs["celltype"] = ["A", "B", "A", "B", "A", "B"]
    return adata


def test_run_gsea_did_multi(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    received = []

    def fake_did_table(adata, *args, **kwargs):
        received.append(adata)
        return pd.DataFrame(
            {
                "feature": ["G1", "G2"],
                "beta_DiD": [1.0, -1.0],
                "se_DiD": [0.5, 0.5],
                "p_DiD": [0.01, 0.2],
                "n_units": [5, 5],
            }
        )

    monkeypatch.setattr(gsea, "did_table", fake_did_table)

    with pytest.warns(UserWarning, match="excluded from the two-arm analysis") as caught:
        res = gsea.run_gsea_did_multi(
            _three_arm_adata(),
            gene_sets={"A": {"S1": ["G1"]}, "B": {"S2": ["G2"]}},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
        )
    assert set(res.keys()) == {"A", "B"}
    # one restriction and one gene-level table for the two collections
    assert len([w for w in caught if "excluded" in str(w.message)]) == 1
    assert len(received) == 1
    assert set(received[0].obs["arm"]) == {"Treated", "Control"} and received[0].n_obs == 4


def test_run_gsea_did_by_celltype(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    received = []

    def fake_run(adata, *args, **kwargs):
        received.append(adata)
        return pd.DataFrame({"gene_set": ["S1"], "NES": [1.0], "FDR q-val": [0.1]})

    monkeypatch.setattr(gsea, "run_gsea_did", fake_run)

    with pytest.warns(UserWarning, match="excluded from the two-arm analysis") as caught:
        res = gsea.run_gsea_did_by_celltype(
            _celltype_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=["A", "B"],
        )
    assert set(res.keys()) == {"A", "B"}
    assert len([w for w in caught if "excluded" in str(w.message)]) == 1
    assert not [w for w in caught if "skipped" in str(w.message)]
    assert all(set(a.obs["arm"]) == {"Treated", "Control"} for a in received)


def _celltype_adata() -> AnnData:
    """Three arms. A and B are sampled at both visits in both analysis arms; the other cell
    types are in the third arm only, in one analysis arm only, or at one visit only."""
    rows = []

    def add(celltype, arm, participants, visits=("V1", "V2")):
        for pid in participants:
            rows.extend({"celltype": celltype, "arm": arm, "participant_id": pid, "visit": v}
                        for v in visits)

    for celltype in ("A", "B"):
        add(celltype, "Treated", ["t1", "t2"])
        add(celltype, "Control", ["c1", "c2"])
        add(celltype, "Other", ["o1"])
    add("OnlyOther", "Other", ["o1", "o2"])
    add("OnlyTreated", "Treated", ["t1", "t2"])
    add("OneVisit", "Treated", ["t1", "t2"], visits=("V1",))
    add("OneVisit", "Control", ["c1", "c2"], visits=("V1",))
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    adata = AnnData(X=np.random.default_rng(0).random((len(obs), 2)), obs=obs)
    adata.var_names = ["G1", "G2"]
    return adata


@pytest.mark.parametrize("celltypes", [None, ["A", "OnlyOther", "OnlyTreated", "OneVisit", "B"]])
def test_run_gsea_did_by_celltype_skips_cell_types_without_a_did(monkeypatch, celltypes):
    # A cell type found only in the excluded arm used to stop the whole call with a KeyError.
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    seen = []

    def fake_run(adata, *args, celltype, **kwargs):
        seen.append(celltype)
        return pd.DataFrame({"gene_set": ["S1"], "NES": [1.0], "FDR q-val": [0.1]})

    monkeypatch.setattr(gsea, "run_gsea_did", fake_run)

    with pytest.warns(UserWarning) as caught:
        res = gsea.run_gsea_did_by_celltype(
            _celltype_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=celltypes,
        )
    assert list(res) == seen == ["A", "B"]
    skipped = [str(w.message) for w in caught if "skipped" in str(w.message)]
    assert len(skipped) == 1 and "OnlyTreated" in skipped[0] and "OneVisit" in skipped[0]
    # a cell type that is not in the two arms is skipped only when it was asked for
    assert ("OnlyOther" in skipped[0]) == (celltypes is not None)


def test_run_gsea_did_by_celltype_skips_a_cell_type_with_too_few_participants(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))

    def fake_did_table(adata, *args, celltype, **kwargs):
        return pd.DataFrame({"feature": ["G1", "G2"], "beta_DiD": [1.0, -1.0], "se_DiD": [0.5, 0.5],
                             "p_DiD": [0.01, 0.2], "n_units": [2 if celltype == "B" else 4] * 2})

    monkeypatch.setattr(gsea, "did_table", fake_did_table)

    with pytest.warns(UserWarning) as caught:
        res = gsea.run_gsea_did_by_celltype(
            _celltype_adata(),
            gene_sets={"S1": ["G1"]},
            design=TrialDesign(celltype_col="celltype"),
            visits=("V1", "V2"),
            celltypes=["A", "B"],
            min_units=4,
        )
    assert list(res) == ["A"]
    assert any("1 cell types were skipped" in str(w.message) and "B" in str(w.message) for w in caught)


def test_run_gsea_pseudobulk(monkeypatch):
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    monkeypatch.setattr(
        gsea,
        "pseudobulk_did",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "feature": ["G1", "G2"],
                "beta_DiD": [1.0, -1.0],
                "se_DiD": [0.5, 0.5],
                "p_DiD": [0.01, 0.2],
                "n_units": [5, 5],
            }
        ),
    )

    adata = AnnData(X=np.random.rand(4, 2))
    adata.var_names = ["G1", "G2"]
    design = types.SimpleNamespace()

    res = gsea.run_gsea_pseudobulk(
        adata,
        gene_sets={"S1": ["G1"]},
        design=design,
        visits=("V1", "V2"),
    )
    assert isinstance(res, pd.DataFrame)


@pytest.mark.parametrize(
    "rank_fn,beta,se,p",
    [
        ("_rank_did_results", "beta_DiD", "se_DiD", "p_DiD"),
        ("_rank_between_arm_results", "beta_arm", "se_arm", "p_arm"),
        ("_rank_within_arm_results", "beta_time", "se_time", "p_time"),
    ],
)
def test_ranking_leaves_out_genes_without_an_estimate(rank_fn, beta, se, p):
    # c and e could not be fitted. Ranked at zero they sat in the middle of the list and
    # counted against every gene set; d has an estimate and no standard error.
    res = pd.DataFrame({
        "feature": list("abcdef"),
        beta: [1.0, -2.0, np.nan, 0.5, np.nan, 0.0],
        se: [0.5, 1.0, np.nan, np.nan, np.nan, 0.4],
        p: [0.04, 0.03, np.nan, 0.5, np.nan, 1.0],
        "n_units": [11, 11, 11, 11, 3, 11],
    })
    rank = getattr(gsea, rank_fn)
    by_t = rank(res, "tstat", 4)
    assert dict(zip(by_t["feature"], by_t["rank"])) == pytest.approx({"a": 2.0, "f": 0.0, "b": -2.0})
    assert list(rank(res, "beta", 4)["feature"]) == ["a", "d", "f", "b"]
    assert list(rank(res, "signed_confidence", 4)["feature"]) == ["a", "d", "f", "b"]

    with pytest.raises(ValueError, match="No genes have an estimate"):
        rank(res.assign(**{beta: np.nan}), "tstat", 4)
    with pytest.raises(ValueError, match="Unknown rank_by"):
        rank(res, "other", 4)


def test_ranking_of_estimated_genes_is_unchanged_by_genes_without_an_estimate():
    rng = np.random.default_rng(0)
    res = pd.DataFrame({"feature": [f"g{i}" for i in range(50)], "beta_DiD": rng.normal(size=50),
                        "se_DiD": rng.uniform(0.2, 1.0, 50), "p_DiD": rng.uniform(size=50), "n_units": 9})
    padded = pd.concat([res, pd.DataFrame({"feature": [f"u{i}" for i in range(10)], "beta_DiD": np.nan,
                                           "se_DiD": np.nan, "p_DiD": np.nan, "n_units": 9})])
    for rank_by in ("tstat", "beta", "signed_confidence"):
        pd.testing.assert_frame_equal(
            gsea._rank_did_results(padded, rank_by, 4).reset_index(drop=True),
            gsea._rank_did_results(res, rank_by, 4).reset_index(drop=True),
        )


def test_run_gsea_did_by_celltype_counts_eligibility_after_crossover_exclusion(monkeypatch):
    # Cell type X has control participants only among the crossovers. Counted before their
    # exclusion it passed, every gene came back without an estimate, and a ranking of zeros
    # went to GSEA.
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=_dummy_prerank))
    rows = []

    def add(celltype, arm, participants, crossover=False):
        for pid in participants:
            rows.extend({"celltype": celltype, "arm": arm, "participant_id": pid, "visit": v,
                         "is_crossover": crossover} for v in ("V1", "V2") for _ in range(3))

    add("A", "Treated", ["t1", "t2", "t3"])
    add("A", "Control", ["c1", "c2", "c3"])
    add("X", "Treated", ["t1", "t2", "t3", "t4"])
    add("X", "Control", ["x1", "x2"], crossover=True)
    obs = pd.DataFrame(rows)
    obs.index = obs.index.astype(str)
    adata = AnnData(X=np.random.default_rng(0).poisson(3.0, size=(len(obs), 5)).astype(float), obs=obs)
    adata.var_names = [f"g{i}" for i in range(5)]
    design = TrialDesign(celltype_col="celltype", crossover_col="is_crossover")
    kw = dict(gene_sets={"S": ["g0", "g1"]}, design=design, visits=("V1", "V2"), min_units=4)

    with pytest.warns(UserWarning, match="1 cell types were skipped.*X"):
        excluded = gsea.run_gsea_did_by_celltype(adata, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = gsea.run_gsea_did_by_celltype(adata, exclude_crossovers=False, **kw)
    assert list(excluded) == ["A"] and list(kept) == ["A", "X"]


def test_ranking_is_the_same_whatever_the_row_order_and_the_last_bits():
    # Genes with identical data tie exactly, and equal statistics can differ in the last bits
    # between runs. Their place in the ranked list decides enrichment scores, so the ranking
    # must not depend on either.
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        "feature": [f"g{i:03d}" for i in range(60)],
        "beta_DiD": np.repeat(rng.normal(size=20), 3),
        "se_DiD": np.repeat(rng.uniform(0.3, 1.0, 20), 3),
        "p_DiD": np.repeat(rng.uniform(size=20), 3),
        "n_units": 9,
    })
    reference = gsea._rank_did_results(base, "tstat", 4)
    assert reference["rank"].duplicated().sum() == 40
    tied = reference.groupby("rank", sort=False)["feature"].apply(list)
    assert all(names == sorted(names) for names in tied)

    shuffled = base.sample(frac=1.0, random_state=1).reset_index(drop=True)
    noisy = shuffled.assign(beta_DiD=shuffled["beta_DiD"] * (1 + 1e-14 * rng.normal(size=60)))
    for other in (shuffled, noisy):
        again = gsea._rank_did_results(other, "tstat", 4)
        assert again["feature"].tolist() == reference["feature"].tolist()
        assert again["rank"].tolist() == reference["rank"].tolist()


def test_duplicated_gene_names_keep_the_strongest_rank():
    res = pd.DataFrame({"feature": ["a", "a", "b", "b"], "beta_DiD": [1.0, -3.0, 2.0, 2.0],
                        "se_DiD": 1.0, "p_DiD": 0.5, "n_units": 9})
    ranking = gsea._rank_did_results(res, "beta", 4)
    assert dict(zip(ranking["feature"], ranking["rank"])) == {"b": 2.0, "a": -3.0}


_VISITS = ("V1", "V2")
_LIBRARIES = {
    "first": {
        "up": [f"G{i}" for i in range(10)],
        "mid": [f"G{i}" for i in range(20, 40)],
        "wide": [f"G{i}" for i in range(5, 60, 2)],
    },
    "second": {
        "low": [f"G{i}" for i in range(60, 75)],
        "mixed": [f"G{i}" for i in (0, 3, 11, 29, 47, 68, 90)],
    },
    "third": {"tail": [f"G{i}" for i in range(80, 100)], "all_up": [f"G{i}" for i in range(5)]},
}
_PRERANK = dict(permutation_num=50, min_size=3, max_size=100, no_plot=True, outdir=None)


def _one_call_and_ranking(kind, adata, design):
    """The one-call GSEA of a design and the ranking it is built on."""
    if kind == "did":
        return (
            lambda sets: gsea.run_gsea_did(adata, sets, design, _VISITS, rank_by="tstat", **_PRERANK),
            lambda: gsea.rank_genes_did(adata, design, _VISITS, rank_by="tstat"),
        )
    if kind == "within_arm":
        return (
            lambda sets: gsea.run_gsea_within_arm(adata, sets, design, "Treated", _VISITS, **_PRERANK),
            lambda: gsea.rank_genes_within_arm(adata, design, "Treated", _VISITS),
        )
    return (
        lambda sets: gsea.run_gsea_cross_sectional(adata, sets, design, "V2", **_PRERANK),
        lambda: gsea.rank_genes_cross_sectional(adata, design, "V2"),
    )


@pytest.mark.parametrize("kind", ["did", "within_arm", "cross_sectional"])
def test_one_ranking_gives_each_library_the_result_of_its_own_call(sample_adata, trial_design, kind):
    # The gene-level table was computed again for every library. One ranking, reused, must
    # give each library exactly what a call of its own gives.
    pytest.importorskip("gseapy")
    one_call, rank_genes = _one_call_and_ranking(kind, sample_adata, trial_design)
    ranking = rank_genes()
    untouched = ranking.copy()
    assert list(ranking.columns) == ["feature", "rank"]
    assert ranking["rank"].is_monotonic_decreasing
    for sets in _LIBRARIES.values():
        pd.testing.assert_frame_equal(
            gsea.run_gsea_prerank(ranking, sets, **_PRERANK), one_call(sets), check_exact=True
        )
    pd.testing.assert_frame_equal(ranking, untouched)


def test_run_gsea_did_multi_equals_one_call_per_library(sample_adata, trial_design):
    pytest.importorskip("gseapy")
    kw = dict(rank_by="tstat", cov_type="nonrobust", min_units=5, **_PRERANK)
    multi = gsea.run_gsea_did_multi(sample_adata, _LIBRARIES, trial_design, _VISITS, **kw)
    assert list(multi) == list(_LIBRARIES)
    for label, sets in _LIBRARIES.items():
        alone = gsea.run_gsea_did(sample_adata, sets, trial_design, _VISITS, **kw)
        pd.testing.assert_frame_equal(multi[label], alone, check_exact=True)


def test_run_gsea_did_multi_ranks_the_genes_once(monkeypatch, sample_adata, trial_design):
    tables, rankings = [], []
    real_did_table = gsea.did_table

    def counting_did_table(*args, **kwargs):
        tables.append(kwargs["cov_type"])
        return real_did_table(*args, **kwargs)

    def recording_prerank(rnk, gene_sets, **kwargs):
        rankings.append(rnk)
        return types.SimpleNamespace(res2d=pd.DataFrame({"Term": list(gene_sets), "NES": 1.0}))

    monkeypatch.setattr(gsea, "did_table", counting_did_table)
    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=recording_prerank))
    res = gsea.run_gsea_did_multi(
        sample_adata, _LIBRARIES, trial_design, _VISITS, rank_by="tstat", cov_type="nonrobust"
    )
    assert tables == ["nonrobust"]
    assert [list(r["Term"]) for r in res.values()] == [list(sets) for sets in _LIBRARIES.values()]
    assert len(rankings) == 3 and len({id(r) for r in rankings}) == 3
    assert all(r.equals(rankings[0]) for r in rankings)

    objects = gsea.run_gsea_did_multi(sample_adata, _LIBRARIES, trial_design, _VISITS, return_obj=True)
    assert len(tables) == 2 and list(objects) == list(_LIBRARIES)
    assert all(hasattr(obj, "res2d") for obj in objects.values())


def test_run_gsea_prerank_keeps_the_ranking_it_is_given(monkeypatch):
    # gseapy renames, sorts and drops rows of a ranking in place. A ranking reused for the
    # next library must not carry what the previous run did to it.
    received = []

    def editing_prerank(rnk, gene_sets, **kwargs):
        received.append(rnk.copy())
        rnk.columns = ["gene_name", "prerank"]
        rnk.drop(rnk.index[:1], inplace=True)
        return types.SimpleNamespace(res2d=pd.DataFrame({"Term": ["S"], "NES": [1.0]}))

    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=editing_prerank))
    ranking = pd.DataFrame({"feature": ["a", "b", "c"], "rank": [2.0, 1.0, -1.0]})
    untouched = ranking.copy()
    for _ in range(2):
        gsea.run_gsea_prerank(ranking, {"S": ["a", "b"]})
    pd.testing.assert_frame_equal(ranking, untouched)
    pd.testing.assert_frame_equal(received[0], received[1])
    pd.testing.assert_frame_equal(received[0], untouched)


def test_run_gsea_prerank_prepares_a_ranking_made_elsewhere(monkeypatch):
    received = []

    def recording_prerank(rnk, gene_sets, **kwargs):
        received.append(rnk)
        return types.SimpleNamespace(res2d=pd.DataFrame({"Term": ["S"], "NES": [1.0]}))

    monkeypatch.setattr(gsea, "gp", types.SimpleNamespace(prerank=recording_prerank))
    ranking = pd.DataFrame({
        "feature": ["b", "a", "c", "c", "d", "e"],
        "rank": [1.0, 1.0, 0.5, -3.0, np.inf, np.nan],
        "note": "x",
    })
    gsea.run_gsea_prerank(ranking, {"S": ["a", "b"]})
    assert list(received[0].columns) == ["feature", "rank"]
    assert received[0]["feature"].tolist() == ["a", "b", "c"]
    assert received[0]["rank"].tolist() == [1.0, 1.0, -3.0]

    gsea.run_gsea_prerank(pd.DataFrame({"feature": ["a", "b"], "rank": [-1, 2]}), {"S": ["a", "b"]})
    assert received[1]["feature"].tolist() == ["b", "a"]
    assert received[1]["rank"].tolist() == [2.0, -1.0] and received[1]["rank"].dtype == float

    with pytest.raises(ValueError, match="needs the columns 'feature' and 'rank'; missing: rank"):
        gsea.run_gsea_prerank(pd.DataFrame({"feature": ["a"], "score": [1.0]}), {"S": ["a"]})


def test_a_finalized_ranking_is_unchanged_by_finalizing_it_again():
    # run_gsea_prerank finalizes every ranking, including one the rank_genes functions have
    # finalized already; the second pass must not move a gene or a bit.
    rng = np.random.default_rng(3)
    values = np.concatenate([
        rng.normal(size=200) / 3.0, np.repeat(rng.normal(size=20), 3),
        [1e-17, -1e-17, 0.0, 123456789.123456789, -2.0 / 3.0, 5e-324],
    ])
    raw = pd.DataFrame({"feature": [f"g{i:03d}" for i in range(len(values))], "rank": values})
    once = gsea._finalize_ranking(raw.sample(frac=1.0, random_state=0))
    twice = gsea._finalize_ranking(once)
    assert once["feature"].tolist() == twice["feature"].tolist()
    assert once["rank"].to_numpy().tobytes() == twice["rank"].to_numpy().tobytes()
    assert once.index.tolist() == twice.index.tolist()
