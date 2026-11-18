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

