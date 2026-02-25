"""Comprehensive tests for sctrial.scoring module.

Covers: score_gene_sets (zmean, mean, sparse, dense, validation,
dedup, non-finite, overwrite, min_genes warnings) and
score_gene_sets_aucell (import guard only — pyscenic not available).
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

import sctrial as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells: int = 10, n_genes: int = 5, *, sparse: bool = False) -> AnnData:
    """Small AnnData with known values for deterministic tests."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float64)
    if sparse:
        X = sp.csr_matrix(X)
    adata = AnnData(X=X)
    adata.var_names = [f"G{i}" for i in range(n_genes)]
    return adata


# ===================================================================
# Input Validation
# ===================================================================

class TestValidation:
    """Input validation for score_gene_sets."""

    def test_invalid_method(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="Unknown method"):
            st.score_gene_sets(adata, {"A": ["G0"]}, method="bad")

    def test_empty_gene_sets(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="non-empty dict"):
            st.score_gene_sets(adata, {})

    def test_non_dict_gene_sets(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="non-empty dict"):
            st.score_gene_sets(adata, [["G0"]])  # type: ignore[arg-type]

    def test_prefix_not_string(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="prefix must be a string"):
            st.score_gene_sets(adata, {"A": ["G0"]}, prefix=123)  # type: ignore[arg-type]

    def test_min_genes_zero(self):
        adata = _make_adata()
        with pytest.raises(ValueError, match="min_genes must be >= 1"):
            st.score_gene_sets(adata, {"A": ["G0"]}, min_genes=0)

    def test_missing_layer(self):
        adata = _make_adata()
        with pytest.raises(KeyError, match="not found"):
            st.score_gene_sets(adata, {"A": ["G0"]}, layer="missing")

    def test_string_gene_list_raises_type_error(self):
        """P2 #3: A bare string instead of list must raise TypeError."""
        adata = _make_adata()
        with pytest.raises(TypeError, match="must be a list"):
            st.score_gene_sets(adata, {"bad": "G0G1"})

    def test_int_gene_list_raises_type_error(self):
        adata = _make_adata()
        with pytest.raises(TypeError, match="must be a list"):
            st.score_gene_sets(adata, {"bad": 42})  # type: ignore[dict-item]

    def test_tuple_gene_list_accepted(self):
        """Tuples and sets should be accepted as gene collections."""
        adata = _make_adata()
        st.score_gene_sets(adata, {"A": ("G0", "G1", "G2")}, min_genes=1)
        assert "A" in adata.obs.columns
        assert not np.isnan(adata.obs["A"]).all()

    def test_set_gene_list_accepted(self):
        adata = _make_adata()
        st.score_gene_sets(adata, {"A": {"G0", "G1", "G2"}}, min_genes=1)
        assert "A" in adata.obs.columns


# ===================================================================
# Core Scoring — zmean
# ===================================================================

class TestZmean:
    """Tests for the zmean scoring method."""

    def test_basic_zmean(self):
        adata = _make_adata()
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]})
        assert "S" in adata.obs.columns
        scores = adata.obs["S"].values
        assert scores.shape == (10,)
        assert np.isfinite(scores).all()

    def test_zmean_mean_approximately_zero(self):
        """Z-scored values should average near zero across cells."""
        adata = _make_adata(n_cells=200)
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]})
        assert abs(adata.obs["S"].mean()) < 0.3

    def test_zmean_zero_variance_all_genes(self):
        """All-constant genes → NaN score."""
        X = np.ones((10, 3))
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]})
        assert np.isnan(adata.obs["S"]).all()

    def test_zmean_zero_variance_partial(self):
        """Mix of constant + varying genes: constant ones excluded."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4))
        X[:, 0] = 5.0  # constant gene
        adata = AnnData(X=X)
        adata.var_names = ["Gconst", "G1", "G2", "G3"]
        st.score_gene_sets(adata, {"S": ["Gconst", "G1", "G2", "G3"]})
        scores = adata.obs["S"].values
        assert np.isfinite(scores).all()

    def test_zmean_sparse(self):
        adata = _make_adata(sparse=True)
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]})
        assert np.isfinite(adata.obs["S"]).all()


# ===================================================================
# Core Scoring — mean
# ===================================================================

class TestMean:
    """Tests for the mean scoring method."""

    def test_mean_dense(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        np.testing.assert_allclose(adata.obs["S"].values, [2.0, 5.0])

    def test_mean_sparse(self):
        X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        np.testing.assert_allclose(adata.obs["S"].values, [2.0, 5.0])

    def test_mean_sparse_csc_conversion(self):
        """CSC matrix should be converted to CSR internally."""
        X = sp.csc_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        np.testing.assert_allclose(adata.obs["S"].values, [2.0, 5.0])


# ===================================================================
# Duplicate Gene Handling
# ===================================================================

class TestDuplicateGenes:
    """P2 #2: Duplicate genes should be deduplicated."""

    def test_duplicates_removed(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1"]
        st.score_gene_sets(adata, {"nodups": ["G0", "G1"]}, method="mean", min_genes=1)

        adata2 = AnnData(X=X.copy())
        adata2.var_names = ["G0", "G1"]
        st.score_gene_sets(adata2, {"dups": ["G0", "G1", "G1"]}, method="mean", min_genes=1)

        np.testing.assert_allclose(adata.obs["nodups"].values, adata2.obs["dups"].values)

    def test_duplicates_zmean(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((20, 3))
        adata1 = AnnData(X=X.copy())
        adata1.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata1, {"S": ["G0", "G1", "G2"]})

        adata2 = AnnData(X=X.copy())
        adata2.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata2, {"S": ["G0", "G1", "G2", "G0", "G1"]})

        np.testing.assert_allclose(adata1.obs["S"].values, adata2.obs["S"].values)

    def test_duplicate_count_uses_unique(self, caplog):
        """Log message should report unique gene count, not raw count."""
        adata = _make_adata()
        # 3 unique genes (G0, G1, MISSING), 5 total entries with duplicates
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(
                adata,
                {"S": ["G0", "G1", "MISSING", "G0", "G1"]},
                min_genes=3,
            )
        # Should report 2/3 unique genes, not 2/5 or 4/5
        assert "2/3 unique genes" in caplog.text.lower()


# ===================================================================
# min_genes Threshold & Warnings
# ===================================================================

class TestMinGenes:
    """P2 #1: min_genes threshold and warning logging."""

    def test_below_min_genes_yields_nan(self):
        adata = _make_adata()
        # Only G0, G1 present; min_genes=3 (default)
        st.score_gene_sets(adata, {"S": ["G0", "G1"]})
        assert np.isnan(adata.obs["S"]).all()

    def test_below_min_genes_logs_warning(self, caplog):
        adata = _make_adata()
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["G0", "G1"]})
        assert "only 2/2 unique genes found" in caplog.text.lower() or "only 2" in caplog.text

    def test_custom_min_genes(self):
        adata = _make_adata()
        st.score_gene_sets(adata, {"S": ["G0", "G1"]}, min_genes=1)
        assert not np.isnan(adata.obs["S"]).all()

    def test_no_genes_found(self, caplog):
        adata = _make_adata()
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["MISSING1", "MISSING2", "MISSING3"]})
        assert np.isnan(adata.obs["S"]).all()
        assert "0/" in caplog.text

    def test_partial_overlap_logs_info(self, caplog):
        """When some genes are missing but overlap >= min_genes, log INFO."""
        adata = _make_adata()
        with caplog.at_level(logging.INFO, logger="sctrial.scoring"):
            st.score_gene_sets(
                adata,
                {"S": ["G0", "G1", "G2", "MISSING1", "MISSING2"]},
                min_genes=3,
            )
        assert not np.isnan(adata.obs["S"]).all()
        assert "3/5 unique genes" in caplog.text.lower()


# ===================================================================
# Overwrite Behaviour
# ===================================================================

class TestOverwrite:
    """overwrite parameter."""

    def test_overwrite_true_replaces(self):
        adata = _make_adata()
        adata.obs["S"] = 999.0
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, overwrite=True)
        assert (adata.obs["S"] != 999.0).any()

    def test_overwrite_false_skips(self):
        adata = _make_adata()
        adata.obs["S"] = 999.0
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, overwrite=False)
        assert (adata.obs["S"] == 999.0).all()


# ===================================================================
# Prefix
# ===================================================================

class TestPrefix:
    """prefix parameter."""

    def test_prefix_applied(self):
        adata = _make_adata()
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, prefix="ms_")
        assert "ms_S" in adata.obs.columns
        assert "S" not in adata.obs.columns

    def test_empty_prefix(self):
        adata = _make_adata()
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, prefix="")
        assert "S" in adata.obs.columns


# ===================================================================
# Layer
# ===================================================================

class TestLayer:
    """layer parameter."""

    def test_uses_specified_layer(self):
        adata = _make_adata()
        adata.layers["norm"] = adata.X * 2
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, layer="norm", method="mean", min_genes=1)
        # Score from layer should be ~2x the score from X
        st.score_gene_sets(adata, {"S_x": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        # Not exact because zmean normalizes, use mean for this test
        ratio = adata.obs["S"].values / adata.obs["S_x"].values
        np.testing.assert_allclose(ratio, 2.0, atol=1e-10)

    def test_none_layer_uses_X(self):
        adata = _make_adata()
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, layer=None, method="mean", min_genes=1)
        expected = np.asarray(adata.X[:, :3]).mean(axis=1)
        np.testing.assert_allclose(adata.obs["S"].values, expected)


# ===================================================================
# Non-Finite Expression Values
# ===================================================================

class TestNonFinite:
    """P3 #4: Non-finite values in expression data are excluded (not just warned)."""

    def test_nan_excluded_from_zmean(self, caplog):
        """NaN values are excluded; remaining finite values produce finite scores."""
        X = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]})
        assert "non-finite" in caplog.text.lower()
        # Scores must be finite (NaN was excluded, not propagated)
        assert np.isfinite(adata.obs["S"]).all()

    def test_inf_excluded_from_zmean(self, caplog):
        """inf values are excluded; remaining finite values produce finite scores."""
        X = np.array([[1.0, 2.0, 3.0], [np.inf, 5.0, 6.0], [7.0, 8.0, 9.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]})
        assert "non-finite" in caplog.text.lower()
        assert np.isfinite(adata.obs["S"]).all()

    def test_nan_excluded_from_mean_dense(self, caplog):
        """NaN excluded from dense mean: mean of finite values only."""
        X = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        assert "non-finite" in caplog.text.lower()
        scores = adata.obs["S"].values
        # Cell 0: mean(1,2,3)=2; Cell 1: mean(5,6)=5.5 (NaN excluded); Cell 2: mean(7,8,9)=8
        np.testing.assert_allclose(scores, [2.0, 5.5, 8.0])

    def test_inf_excluded_from_mean_dense(self, caplog):
        """inf excluded from dense mean: mean of finite values only."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, np.inf], [7.0, 8.0, 9.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        assert "non-finite" in caplog.text.lower()
        scores = adata.obs["S"].values
        # Cell 0: mean(1,2,3)=2; Cell 1: mean(4,5)=4.5 (inf excluded); Cell 2: mean(7,8,9)=8
        np.testing.assert_allclose(scores, [2.0, 4.5, 8.0])

    def test_nan_excluded_from_sparse_mean(self, caplog):
        """Non-finite in sparse mean path: excluded, not propagated."""
        X = sp.csr_matrix(np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0]]))
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        with caplog.at_level(logging.WARNING, logger="sctrial.scoring"):
            st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        assert "non-finite" in caplog.text.lower()
        scores = adata.obs["S"].values
        # Cell 0: mean(1,2,3)=2; Cell 1: mean(5,6)=5.5 (NaN excluded)
        np.testing.assert_allclose(scores, [2.0, 5.5])

    def test_all_nonfinite_for_cell_gives_nan(self):
        """If ALL genes for a cell are non-finite, the score should be NaN."""
        X = np.array([[np.nan, np.inf, -np.inf], [1.0, 2.0, 3.0]])
        adata = AnnData(X=X)
        adata.var_names = ["G0", "G1", "G2"]
        st.score_gene_sets(adata, {"S": ["G0", "G1", "G2"]}, method="mean", min_genes=1)
        scores = adata.obs["S"].values
        assert np.isnan(scores[0]), "Cell with all non-finite should get NaN"
        np.testing.assert_allclose(scores[1], 2.0)


# ===================================================================
# Multiple Gene Sets
# ===================================================================

class TestMultipleSets:
    """Multiple gene sets in one call."""

    def test_multiple_sets(self):
        adata = _make_adata()
        gene_sets = {
            "A": ["G0", "G1", "G2"],
            "B": ["G2", "G3", "G4"],
        }
        st.score_gene_sets(adata, gene_sets)
        assert "A" in adata.obs.columns
        assert "B" in adata.obs.columns

    def test_mix_of_valid_and_nan(self):
        """One set passes min_genes, another doesn't."""
        adata = _make_adata()
        gene_sets = {
            "ok": ["G0", "G1", "G2"],
            "bad": ["MISSING1", "MISSING2", "MISSING3"],
        }
        st.score_gene_sets(adata, gene_sets)
        assert np.isfinite(adata.obs["ok"]).all()
        assert np.isnan(adata.obs["bad"]).all()


# ===================================================================
# AUCell (import guard)
# ===================================================================

class TestAucell:
    """score_gene_sets_aucell — test import guard only."""

    def test_import_error_when_pyscenic_missing(self):
        with patch("sctrial.scoring.create_rankings", None), \
             patch("sctrial.scoring.aucell", None):
            adata = _make_adata()
            with pytest.raises(ImportError, match="pyscenic"):
                st.score_gene_sets_aucell(adata, {"S": ["G0"]})
