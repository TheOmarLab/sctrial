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
    assert sh._gsea_did_cache_fresh(cache, "cluster")
    assert not sh._gsea_did_cache_fresh(cache, "nonrobust")
    sh._gsea_fingerprint_path(cache).write_text(sh.STATS_FINGERPRINT)
    assert not sh._gsea_did_cache_fresh(cache, "cluster")


def test_loader_ranks_with_the_requested_covariance_and_stamps_it(tmp_path, monkeypatch, sh):
    import sctrial

    monkeypatch.setattr(sh, "GSEA_OUTPUT", tmp_path)
    monkeypatch.setattr(sh, "GSEA_LIBRARIES", [("Library_A", "A")])
    received = []

    def fake_run_gsea_did(adata, **kwargs):
        received.append(kwargs["cov_type"])
        return pd.DataFrame({"Term": ["T1"], "NES": [1.5 if kwargs["cov_type"] == "cluster" else -1.5]})

    monkeypatch.setattr(sctrial, "run_gsea_did", fake_run_gsea_did)

    first = sh.load_or_run_gsea_did(None, None, ("Pre", "Post"), None, "Cohort")
    again = sh.load_or_run_gsea_did(None, None, ("Pre", "Post"), None, "Cohort")
    pooled = sh.load_or_run_gsea_did(None, None, ("Pre", "Post"), None, "Cohort", cov_type="nonrobust")
    assert received == ["cluster", "nonrobust"]
    assert first["NES"].tolist() == again["NES"].tolist() == [1.5]
    assert pooled["NES"].tolist() == [-1.5]
    stamp = sh._gsea_fingerprint_path(sh._gsea_cache_path("Cohort", "A", method="did"))
    assert stamp.read_text() == sh._gsea_did_stamp("nonrobust")



@pytest.mark.parametrize("loader,runner", [
    ("load_or_run_gsea_within_arm", "run_gsea_within_arm"),
    ("load_or_run_gsea_cross_sectional", "run_gsea_cross_sectional"),
])
def test_within_arm_and_cross_sectional_caches_need_the_stamp_of_the_current_code(
    tmp_path, monkeypatch, sh, loader, runner
):
    # These caches were reused whenever the file existed, so a change to the ranking of the
    # genes kept serving results ranked by the previous code.
    import sctrial

    monkeypatch.setattr(sh, "GSEA_OUTPUT", tmp_path)
    monkeypatch.setattr(sh, "GSEA_LIBRARIES", [("Library_A", "A")])
    calls = []

    def fake_run(adata, **kwargs):
        calls.append(kwargs["rank_by"])
        return pd.DataFrame({"Term": ["T1"], "NES": [float(len(calls))]})

    monkeypatch.setattr(sctrial, runner, fake_run)
    args = (dict(arm="Treated", visits=("Pre", "Post")) if "within" in loader else dict(visit="D7"))
    run = lambda: getattr(sh, loader)(None, None, layer=None, dataset_name="Cohort", **args)  # noqa: E731

    assert run()["NES"].tolist() == [1.0]
    assert run()["NES"].tolist() == [1.0] and len(calls) == 1
    method = "within_arm" if "within" in loader else "cross_sectional"
    stamp = sh._gsea_fingerprint_path(sh._gsea_cache_path("Cohort", "A", method=method))
    assert stamp.read_text() == sh.STATS_FINGERPRINT

    stamp.write_text("built by other code")
    assert run()["NES"].tolist() == [2.0]
    stamp.unlink()
    assert run()["NES"].tolist() == [3.0]


def test_fingerprint_covers_the_estimators_and_the_gene_ranking(sh):
    import inspect

    source = inspect.getsource(sh._stats_fingerprint)
    for module in ("stats/did.py", "utils.py", "stats/_utils.py", "stats/comparisons.py", "stats/gsea.py"):
        assert module in source


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
