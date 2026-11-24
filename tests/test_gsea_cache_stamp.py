"""A DiD-ranked GSEA cache is reused only for the covariance that ranked its genes."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

pytest.importorskip("scanpy")
pytest.importorskip("gseapy")


@pytest.fixture
def sh(load_figure_module):
    load_figure_module("supp/supp_tables")
    return sys.modules["manuscript_figures._shared"]


def test_cache_built_with_another_covariance_is_stale(tmp_path, sh):
    cache = tmp_path / "results.csv"
    cache.write_text("Term,NES\nA,1.0\n")
    sh._gsea_fingerprint_path(cache).write_text(sh._gsea_did_stamp("cluster"))
    assert sh._gsea_cache_fresh(cache, sh._gsea_did_stamp("cluster"))
    assert not sh._gsea_cache_fresh(cache, sh._gsea_did_stamp("nonrobust"))
    sh._gsea_fingerprint_path(cache).write_text(sh.STATS_FINGERPRINT)
    assert not sh._gsea_cache_fresh(cache, sh._gsea_did_stamp("cluster"))


def _ranking(sign: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"feature": ["g1", "g2"], "rank": [sign, -sign]})


def _nes_of_first_gene(ranking, gene_sets, **kwargs):
    """Stands for gseapy: one pathway whose NES tells which ranking it was given."""
    return pd.DataFrame({"Term": ["T1"], "NES": [1.5 * float(ranking["rank"].iloc[0])]})


def test_loader_ranks_with_the_requested_covariance_and_stamps_it(tmp_path, monkeypatch, sh):
    import sctrial

    monkeypatch.setattr(sh, "GSEA_OUTPUT", tmp_path)
    monkeypatch.setattr(sh, "GSEA_LIBRARIES", [("Library_A", "A")])
    received = []

    def fake_rank_genes_did(adata, design, visits, **kwargs):
        received.append((kwargs["cov_type"], kwargs["rank_by"]))
        return _ranking(1.0 if kwargs["cov_type"] == "cluster" else -1.0)

    monkeypatch.setattr(sctrial, "rank_genes_did", fake_rank_genes_did)
    monkeypatch.setattr(sctrial, "run_gsea_prerank", _nes_of_first_gene)

    first = sh.load_or_run_gsea_did(None, None, ("Pre", "Post"), None, "Cohort")
    again = sh.load_or_run_gsea_did(None, None, ("Pre", "Post"), None, "Cohort")
    pooled = sh.load_or_run_gsea_did(None, None, ("Pre", "Post"), None, "Cohort", cov_type="nonrobust")
    assert received == [("cluster", "tstat"), ("nonrobust", "tstat")]
    assert first["NES"].tolist() == again["NES"].tolist() == [1.5]
    assert pooled["NES"].tolist() == [-1.5]
    stamp = sh._gsea_fingerprint_path(sh._gsea_cache_path("Cohort", "A", method="did"))
    assert stamp.read_text() == sh._gsea_did_stamp("nonrobust")


_LOADERS = {
    "did": ("load_or_run_gsea_did", "rank_genes_did", dict(visits=("Pre", "Post"))),
    "within_arm": (
        "load_or_run_gsea_within_arm", "rank_genes_within_arm",
        dict(arm="Treated", visits=("Pre", "Post")),
    ),
    "cross_sectional": ("load_or_run_gsea_cross_sectional", "rank_genes_cross_sectional", dict(visit="D7")),
}


def _counted_loader(sh, monkeypatch, tmp_path, method, libraries):
    """The loader of *method* over *libraries*, with the ranking and gseapy replaced by counters."""
    import sctrial

    loader, ranker, args = _LOADERS[method]
    monkeypatch.setattr(sh, "GSEA_OUTPUT", tmp_path)
    monkeypatch.setattr(sh, "GSEA_LIBRARIES", libraries)
    ranked, enriched = [], []

    def fake_rank(adata, design, *positional, **kwargs):
        ranked.append(kwargs["rank_by"])
        return _ranking(float(len(ranked)))

    def fake_prerank(ranking, gene_sets, **kwargs):
        enriched.append(gene_sets)
        return _nes_of_first_gene(ranking, gene_sets)

    monkeypatch.setattr(sctrial, ranker, fake_rank)
    monkeypatch.setattr(sctrial, "run_gsea_prerank", fake_prerank)

    def run(**extra):
        return getattr(sh, loader)(None, None, layer=None, dataset_name="Cohort", **args, **extra)

    return run, ranked, enriched


@pytest.mark.parametrize("method", ["within_arm", "cross_sectional"])
def test_within_arm_and_cross_sectional_caches_need_the_stamp_of_the_current_code(
    tmp_path, monkeypatch, sh, method
):
    # These caches were reused whenever the file existed, so a change to the ranking of the
    # genes kept serving results ranked by the previous code.
    run, ranked, _ = _counted_loader(sh, monkeypatch, tmp_path, method, [("Library_A", "A")])

    assert run()["NES"].tolist() == [1.5]
    assert run()["NES"].tolist() == [1.5] and ranked == ["tstat"]
    stamp = sh._gsea_fingerprint_path(sh._gsea_cache_path("Cohort", "A", method=method))
    assert stamp.read_text() == sh.STATS_FINGERPRINT

    stamp.write_text("built by other code")
    assert run()["NES"].tolist() == [3.0]
    stamp.unlink()
    assert run()["NES"].tolist() == [4.5]


_FIVE = [(f"Library_{name}", name) for name in "ABCDE"]


@pytest.mark.parametrize("method", sorted(_LOADERS))
def test_loader_ranks_the_genes_once_for_all_libraries(tmp_path, monkeypatch, sh, method):
    # The gene-level table, one model per gene, was computed again for each library.
    run, ranked, enriched = _counted_loader(sh, monkeypatch, tmp_path, method, _FIVE)

    first = run()
    assert len(ranked) == 1 and enriched == [name for name, _ in _FIVE]
    assert sorted(first["Library"]) == list("ABCDE") and set(first["NES"]) == {1.5}

    run()
    assert len(ranked) == 1 and len(enriched) == 5

    sh._gsea_fingerprint_path(sh._gsea_cache_path("Cohort", "C", method=method)).unlink()
    refreshed = run()
    assert len(ranked) == 2 and enriched[5:] == ["Library_C"]
    assert refreshed.set_index("Library")["NES"].to_dict() == {"A": 1.5, "B": 1.5, "C": 3.0, "D": 1.5, "E": 1.5}

    forced = run(force=True)
    assert len(ranked) == 3 and enriched[6:] == [name for name, _ in _FIVE]
    assert set(forced["NES"]) == {4.5}


def test_loader_does_not_repeat_a_ranking_that_failed(tmp_path, monkeypatch, sh, capsys):
    import sctrial

    run, _, enriched = _counted_loader(sh, monkeypatch, tmp_path, "within_arm", _FIVE)
    attempts = []

    def failing_rank(*args, **kwargs):
        attempts.append(1)
        raise ValueError("No genes have an estimate")

    monkeypatch.setattr(sctrial, "rank_genes_within_arm", failing_rank)
    assert run() is None
    assert len(attempts) == 1 and enriched == []
    assert capsys.readouterr().out.count("gene ranking failed: No genes have an estimate") == 1


def test_fingerprint_covers_every_statistics_module(sh):
    from pathlib import Path

    import sctrial

    pkg = Path(sctrial.__file__).resolve().parent
    hashed = {p.relative_to(pkg).as_posix() for p in sh._fingerprint_sources(pkg)}
    assert {p.relative_to(pkg).as_posix() for p in (pkg / "stats").glob("*.py")} <= hashed
    for module in (
        "stats/did.py",
        "stats/_utils.py",
        "stats/comparisons.py",
        "stats/gsea.py",
        "stats/abundance.py",
        "stats/power.py",
        "stats/effect_size.py",
        "utils.py",
        "scoring.py",
    ):
        assert module in hashed
    assert all(p.is_file() for p in sh._fingerprint_sources(pkg))


def test_fingerprint_changes_with_any_hashed_file(sh, tmp_path):
    import shutil
    from pathlib import Path

    import sctrial

    pkg = Path(sctrial.__file__).resolve().parent
    copy = tmp_path / "sctrial"
    for path in sh._fingerprint_sources(pkg):
        target = copy / path.relative_to(pkg)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    reference = sh._stats_fingerprint(copy)
    assert reference == sh._stats_fingerprint(pkg) == sh.STATS_FINGERPRINT
    for name in ("stats/abundance.py", "stats/power.py", "stats/did.py", "scoring.py"):
        original = (copy / name).read_bytes()
        (copy / name).write_bytes(original + b"\n# changed\n")
        assert sh._stats_fingerprint(copy) != reference, name
        (copy / name).write_bytes(original)
    assert sh._stats_fingerprint(copy) == reference


def test_tied_pathways_keep_one_order_whatever_order_they_arrive_in(sh):
    # Two gene sets with the same members have the same NES; gseapy does not return them in
    # the same order in every run, and a panel of top pathways must not depend on that.
    rows = pd.DataFrame({
        "Term": ["Set B", "Set A", "Set C", "Set D"],
        "NES": [2.5, 2.5, -3.0, 1.0],
        "Library": ["GO", "GO", "GO", "GO"],
    })
    first = sh._order_gsea(rows)
    second = sh._order_gsea(rows.iloc[[1, 0, 3, 2]].reset_index(drop=True))
    assert first["Term"].tolist() == second["Term"].tolist() == ["Set C", "Set A", "Set B", "Set D"]
