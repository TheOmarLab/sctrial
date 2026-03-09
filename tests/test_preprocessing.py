"""Comprehensive tests for sctrial.preprocessing (add_log1p_cpm_layer)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

import sctrial as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(X, *, sparse: bool = False, layer: str = "counts"):
    """Create a minimal AnnData with counts in a layer."""
    if sparse:
        X_mat = sp.csr_matrix(X)
    else:
        X_mat = np.asarray(X, dtype=float)
    ad = AnnData(X=X_mat.copy())
    ad.layers[layer] = X_mat.copy()
    ad.var_names = [f"G{i}" for i in range(X_mat.shape[1])]
    return ad


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasicNormalization:
    """Core normalization correctness."""

    def test_creates_layer(self, sample_adata):
        st.add_log1p_cpm_layer(sample_adata, counts_layer="counts", out_layer="norm")
        assert "norm" in sample_adata.layers

    def test_output_non_negative(self, sample_adata):
        st.add_log1p_cpm_layer(sample_adata, counts_layer="counts", out_layer="norm")
        vals = sample_adata.layers["norm"]
        if sp.issparse(vals):
            vals = vals.toarray()
        assert np.all(vals >= 0)

    def test_matches_manual_cpm(self):
        """Verify output matches manual log1p(CPM) calculation."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        result = np.asarray(ad.layers["norm"])

        libsize = X.sum(axis=1, keepdims=True)
        expected = np.log1p(X / libsize * 1e6)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_custom_scale(self):
        """Custom scale factor is respected."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=1e4)
        result = np.asarray(ad.layers["norm"])

        libsize = X.sum(axis=1, keepdims=True)
        expected = np.log1p(X / libsize * 1e4)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dense_sparse_consistency(self):
        """Dense and sparse paths produce identical results."""
        X = np.array([[0, 3, 0, 5], [1, 0, 2, 0], [7, 8, 0, 1]], dtype=float)

        ad_dense = _make_adata(X, sparse=False)
        ad_sparse = _make_adata(X, sparse=True)

        st.add_log1p_cpm_layer(ad_dense, counts_layer="counts", out_layer="norm")
        st.add_log1p_cpm_layer(ad_sparse, counts_layer="counts", out_layer="norm")

        dense_result = np.asarray(ad_dense.layers["norm"])
        sparse_result = np.asarray(ad_sparse.layers["norm"].toarray())
        np.testing.assert_allclose(dense_result, sparse_result, atol=1e-12)


# ---------------------------------------------------------------------------
# Input validation (P1)
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Input validation: negative counts and invalid scale."""

    def test_negative_counts_raises(self):
        """Negative counts should raise ValueError."""
        X = np.array([[-1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="negative values"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")

    def test_negative_counts_sparse_raises(self):
        """Negative counts in sparse matrix should raise ValueError."""
        X = np.array([[-1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X, sparse=True)
        with pytest.raises(ValueError, match="negative values"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")

    def test_scale_nan_raises(self):
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="finite positive"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=np.nan)

    def test_scale_inf_raises(self):
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="finite positive"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=np.inf)

    def test_scale_negative_raises(self):
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="finite positive"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=-1)

    def test_scale_zero_raises(self):
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="finite positive"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=0)

    def test_nan_counts_raises(self):
        """NaN in counts should raise ValueError."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="non-finite"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")

    def test_inf_counts_raises(self):
        """Inf in counts should raise ValueError."""
        X = np.array([[1.0, np.inf], [3.0, 4.0]])
        ad = _make_adata(X)
        with pytest.raises(ValueError, match="non-finite"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")

    def test_nan_counts_sparse_raises(self):
        """NaN in sparse counts should raise ValueError."""
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        ad = _make_adata(X, sparse=True)
        with pytest.raises(ValueError, match="non-finite"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")

    def test_missing_counts_layer_raises(self):
        ad = AnnData(X=np.array([[1.0, 2.0]]))
        with pytest.raises(KeyError, match="not found"):
            st.add_log1p_cpm_layer(ad, counts_layer="missing", out_layer="norm")


# ---------------------------------------------------------------------------
# Stale-layer handling (P2)
# ---------------------------------------------------------------------------


class TestOverwriteBehavior:
    """overwrite flag and stale-layer detection."""

    def test_overwrite_false_skips(self):
        """Existing layer is returned unchanged when overwrite=False."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=1e6)
        first = np.asarray(ad.layers["norm"]).copy()

        # Call again with different scale but overwrite=False
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=1e4)
        second = np.asarray(ad.layers["norm"])
        np.testing.assert_array_equal(first, second)

    def test_overwrite_false_logs(self, caplog):
        """Skipping should emit a log message."""
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")

        with caplog.at_level(logging.INFO, logger="sctrial.preprocessing"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        assert "already exists" in caplog.text

    def test_overwrite_true_recomputes(self):
        """overwrite=True should recompute with new parameters."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=1e6)
        first = np.asarray(ad.layers["norm"]).copy()

        st.add_log1p_cpm_layer(
            ad, counts_layer="counts", out_layer="norm", scale=1e4, overwrite=True
        )
        second = np.asarray(ad.layers["norm"])
        assert not np.allclose(first, second), "overwrite=True should produce different values"


# ---------------------------------------------------------------------------
# Zero-library cells (P2)
# ---------------------------------------------------------------------------


class TestZeroLibraryCells:
    """Cells with zero total counts."""

    def test_zero_count_cell_produces_zeros_dense(self):
        """Zero-count cells should get all-zero CPM, not NaN/inf."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        result = np.asarray(ad.layers["norm"])
        assert np.all(result[0] == 0), "Zero-count cell should produce all-zero CPM"
        assert np.all(np.isfinite(result)), "No NaN/inf should be present"

    def test_zero_count_cell_produces_zeros_sparse(self):
        """Same for sparse input."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        ad = _make_adata(X, sparse=True)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        result = np.asarray(ad.layers["norm"].toarray())
        assert np.all(result[0] == 0), "Zero-count cell should produce all-zero CPM"
        assert np.all(np.isfinite(result)), "No NaN/inf should be present"

    def test_zero_count_cell_integer_dtype_dense(self):
        """Integer counts with zero-count cell should not crash (dense)."""
        X = np.array([[0, 0], [3, 4]], dtype=np.int32)
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        result = np.asarray(ad.layers["norm"])
        assert np.all(result[0] == 0)
        assert np.all(np.isfinite(result))

    def test_zero_count_cell_integer_dtype_sparse(self):
        """Integer counts with zero-count cell should not crash (sparse)."""
        X = np.array([[0, 0], [3, 4]], dtype=np.int32)
        ad = _make_adata(X, sparse=True)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        result = np.asarray(ad.layers["norm"].toarray())
        assert np.all(result[0] == 0)
        assert np.all(np.isfinite(result))

    def test_zero_count_cell_warns(self, caplog):
        """Zero-count cells should emit a warning."""
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        ad = _make_adata(X)
        with caplog.at_level(logging.WARNING, logger="sctrial.preprocessing"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        assert "zero total counts" in caplog.text

    def test_no_warning_when_all_cells_have_counts(self, caplog):
        """No warning when all cells have positive library sizes."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X)
        with caplog.at_level(logging.WARNING, logger="sctrial.preprocessing"):
            st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        assert "zero total counts" not in caplog.text


# ---------------------------------------------------------------------------
# counts_layer=None and layer_out alias
# ---------------------------------------------------------------------------


class TestLayerOptions:
    """counts_layer=None and layer_out backward-compat alias."""

    def test_counts_layer_none_uses_X(self):
        """counts_layer=None should normalize from adata.X."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ad = AnnData(X=X.copy())
        st.add_log1p_cpm_layer(ad, counts_layer=None, out_layer="norm")
        assert "norm" in ad.layers

        result = np.asarray(ad.layers["norm"])
        libsize = X.sum(axis=1, keepdims=True)
        expected = np.log1p(X / libsize * 1e6)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_layer_out_alias(self):
        """layer_out should work as alias for out_layer."""
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", layer_out="my_norm")
        assert "my_norm" in ad.layers

    def test_inplace_false_returns_copy(self):
        """inplace=False should not modify the original."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        ad = _make_adata(X)
        result = st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", inplace=False)
        assert "norm" in result.layers
        assert "norm" not in ad.layers, "Original should not be modified"


# ---------------------------------------------------------------------------
# Provenance tracking
# ---------------------------------------------------------------------------


class TestProvenance:
    """Provenance metadata in adata.uns."""

    def test_provenance_stored(self):
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm", scale=1e4)
        assert ad.uns.get("sctrial", {}).get("log1p_cpm_scale") == 1e4

    def test_provenance_default_scale(self):
        X = np.array([[1.0, 2.0]])
        ad = _make_adata(X)
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        assert ad.uns.get("sctrial", {}).get("log1p_cpm_scale") == 1e6


# ---------------------------------------------------------------------------
# Sparse format conversion
# ---------------------------------------------------------------------------


class TestSparseFormats:
    """Sparse matrix format conversion (CSC/COO → CSR)."""

    def test_csc_input(self):
        """CSC input should be handled correctly."""
        X = np.array([[0, 3, 0], [1, 0, 2]], dtype=float)
        ad = AnnData(X=sp.csc_matrix(X))
        ad.layers["counts"] = sp.csc_matrix(X)
        ad.var_names = ["G0", "G1", "G2"]
        st.add_log1p_cpm_layer(ad, counts_layer="counts", out_layer="norm")
        result = np.asarray(ad.layers["norm"].toarray())

        libsize = X.sum(axis=1, keepdims=True)
        expected = np.log1p(X / libsize * 1e6)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_counts_layer_none_sparse(self):
        """counts_layer=None with sparse X should work."""
        X = np.array([[0, 3, 0], [1, 0, 2]], dtype=float)
        ad = AnnData(X=sp.csr_matrix(X))
        ad.var_names = ["G0", "G1", "G2"]
        st.add_log1p_cpm_layer(ad, counts_layer=None, out_layer="norm")
        result = np.asarray(ad.layers["norm"].toarray())

        libsize = X.sum(axis=1, keepdims=True)
        expected = np.log1p(X / libsize * 1e6)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
