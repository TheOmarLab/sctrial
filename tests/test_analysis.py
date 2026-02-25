"""Comprehensive tests for sctrial.analysis (DiDAnalyzer)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import sctrial as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toy_adata() -> AnnData:
    """Build a small balanced trial AnnData."""
    obs_rows = []
    for pid in ["P1", "P2", "P3", "P4"]:
        arm = "Treated" if pid in ("P1", "P2") else "Control"
        for visit in ["V1", "V2"]:
            obs_rows.append({
                "participant_id": pid,
                "visit": visit,
                "arm": arm,
                "celltype": "TypeA" if pid in ("P1", "P3") else "TypeB",
            })
    obs = pd.DataFrame(obs_rows)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(len(obs), 2))
    ad = AnnData(X=X, obs=obs)
    ad.var_names = ["G0", "G1"]
    ad.obs["score"] = rng.normal(size=len(obs))
    return ad


def _default_design() -> st.TrialDesign:
    return st.TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="arm",
        arm_treated="Treated",
        arm_control="Control",
        celltype_col="celltype",
    )


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:

    def test_basic_fit(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        res = analyzer.fit(features=["score"], visits=("V1", "V2"))
        assert isinstance(res, pd.DataFrame)
        assert not res.empty
        assert "beta_DiD" in res.columns

    def test_results_stored(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        assert analyzer.results_ is None
        analyzer.fit(features=["score"], visits=("V1", "V2"))
        assert analyzer.results_ is not None

    def test_returned_copy_not_alias(self):
        """P3 fix: returned DataFrame should be a copy, not an alias."""
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        returned = analyzer.fit(features=["score"], visits=("V1", "V2"))
        # Mutate the returned copy
        returned["beta_DiD"] = 999.0
        # Internal state should be unchanged
        assert (analyzer.results_["beta_DiD"] != 999.0).all()

    def test_fit_with_config(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        cfg = st.DiDConfig(standardize=False, use_bootstrap=False)
        res = analyzer.fit(features=["score"], visits=("V1", "V2"), config=cfg)
        assert not res.empty

    def test_fit_with_celltype(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        res = analyzer.fit(
            features=["score"], visits=("V1", "V2"), celltype="TypeA"
        )
        assert not res.empty

    def test_fit_overwrites_previous(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        analyzer.fit(features=["score"], visits=("V1", "V2"))
        analyzer.fit(features=["score"], visits=("V1", "V2"))
        # Both should succeed; results_ reflects the latest
        assert analyzer.results_ is not None


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------

class TestSummarize:

    def test_summarize_after_fit(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        analyzer.fit(features=["score"], visits=("V1", "V2"))
        summary = analyzer.summarize()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summarize_before_fit_raises(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        with pytest.raises(ValueError, match="No results available"):
            analyzer.summarize()


# ---------------------------------------------------------------------------
# plot_forest()
# ---------------------------------------------------------------------------

class TestPlotForest:

    def test_plot_forest_before_fit_raises(self):
        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        with pytest.raises(ValueError, match="No results available"):
            analyzer.plot_forest()

    def test_plot_forest_after_fit(self):
        import matplotlib
        matplotlib.use("Agg")

        ad = _toy_adata()
        analyzer = st.DiDAnalyzer(ad, _default_design())
        analyzer.fit(features=["score"], visits=("V1", "V2"))
        fig = analyzer.plot_forest()
        assert fig is not None
