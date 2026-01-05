"""Extended tests for dataset loading and helper functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from anndata import AnnData

from sctrial.datasets import (
    _counts_like,
    _get_counts_matrix,
    _looks_log1p,
    _params_match,
    categorize_celltype,
    count_paired,
    ensure_fdr,
)


class TestCountsDetection:
    """Test automatic counts matrix detection."""

    def test_counts_like_with_integers(self):
        """Test that integer arrays are detected as counts."""
        data = np.array([0, 1, 2, 5, 10, 100])
        assert _counts_like(data)

    def test_counts_like_with_floats(self):
        """Test that float arrays are not detected as counts."""
        data = np.array([0.5, 1.2, 2.7, 5.1])
        assert not _counts_like(data)

    def test_counts_like_with_negative(self):
        """Test that arrays with negative values are not detected as counts."""
        data = np.array([0, 1, -2, 5])
        assert not _counts_like(data)

    def test_counts_like_sparse(self):
        """Test counts detection with sparse matrices."""
        data = sp.csr_matrix(np.array([[0, 1, 2], [3, 0, 5]]))
        assert _counts_like(data)

    def test_counts_like_empty(self):
        """Test counts detection with empty arrays."""
        data = np.array([])
        assert not _counts_like(data)


class TestLog1pDetection:
    """Test log1p-transformed data detection."""

    def test_looks_log1p_positive(self):
        """Test detection of log1p-transformed data."""
        # Simulated log1p CPM data (should be in range [0, ~15])
        data = np.array([0.0, 1.2, 2.3, 3.5, 5.7, 8.2])
        assert _looks_log1p(data)

    def test_looks_log1p_integers(self):
        """Test that integer counts are not detected as log1p."""
        data = np.array([0, 1, 2, 50, 100, 500])
        assert not _looks_log1p(data)

    def test_looks_log1p_large_values(self):
        """Test that arrays with very large values are not log1p."""
        data = np.array([0.5, 1.2, 50.0, 100.0])
        assert not _looks_log1p(data)

    def test_looks_log1p_sparse(self):
        """Test log1p detection with sparse matrices."""
        # Log1p CPM typically in range [0, 15]
        data = sp.csr_matrix(np.array([[0.0, 1.5, 2.3], [3.2, 0.0, 4.5]]))
        assert _looks_log1p(data)


class TestGetCountsMatrix:
    """Test automatic counts matrix extraction."""

    def test_get_counts_from_layer(self):
        """Test extraction from counts layer."""
        X = np.random.rand(100, 50) * 10  # Normalized data
        counts = np.random.poisson(5, (100, 50)).astype(float)
        adata = AnnData(X=X)
        adata.layers["counts"] = counts

        extracted, source = _get_counts_matrix(adata)
        assert extracted is not None
        assert source == "layers['counts']"
        assert np.array_equal(extracted, counts)

    def test_get_counts_from_X(self):
        """Test extraction when X contains counts."""
        counts = np.random.poisson(5, (100, 50)).astype(float)
        adata = AnnData(X=counts)

        extracted, source = _get_counts_matrix(adata)
        assert extracted is not None
        assert source == "adata.X"

    def test_get_counts_no_counts_available(self):
        """Test when no counts are available."""
        X = np.random.rand(100, 50) * 10  # Normalized, non-integer data
        adata = AnnData(X=X)

        extracted, source = _get_counts_matrix(adata)
        assert extracted is None
        assert source is None


class TestParamsMatch:
    """Test parameter matching for cached data."""

    def test_params_match_identical(self):
        """Test matching with identical parameters."""
        prev = {"version": "v1", "seed": 42, "max_cells": 100}
        curr = {"version": "v1", "seed": 42, "max_cells": 100}
        assert _params_match(prev, curr)

    def test_params_match_different(self):
        """Test non-matching with different parameters."""
        prev = {"version": "v1", "seed": 42}
        curr = {"version": "v2", "seed": 42}
        assert not _params_match(prev, curr)

    def test_params_match_with_lists(self):
        """Test matching with list parameters."""
        prev = {"days": [0, 7, 14]}
        curr = {"days": [0, 7, 14]}
        assert _params_match(prev, curr)

    def test_params_match_with_numpy_array(self):
        """Test matching with numpy array parameters."""
        prev = {"days": np.array([0, 7, 14])}
        curr = {"days": [0, 7, 14]}
        assert _params_match(prev, curr)

    def test_params_match_extra_in_prev(self):
        """Test when previous params have extra keys."""
        prev = {"version": "v1", "seed": 42, "extra": "value"}
        curr = {"version": "v1", "seed": 42}
        # Should match because all keys in curr match values in prev
        assert _params_match(prev, curr)


class TestCountPaired:
    """Test counting of paired participants."""

    def test_count_paired_basic(self):
        """Test basic paired participant counting."""
        obs = pd.DataFrame({
            "participant_id": ["P1", "P1", "P2", "P2", "P3"],
            "visit": ["V1", "V2", "V1", "V2", "V1"],
        })
        n_paired = count_paired(obs, "visit", ["V1", "V2"])
        assert n_paired == 2  # P1 and P2 have both visits

    def test_count_paired_no_pairs(self):
        """Test when no participants have both visits."""
        obs = pd.DataFrame({
            "participant_id": ["P1", "P2", "P3"],
            "visit": ["V1", "V1", "V2"],
        })
        n_paired = count_paired(obs, "visit", ["V1", "V2"])
        assert n_paired == 0

    def test_count_paired_all_paired(self):
        """Test when all participants have both visits."""
        obs = pd.DataFrame({
            "participant_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "visit": ["V1", "V2", "V1", "V2", "V1", "V2"],
        })
        n_paired = count_paired(obs, "visit", ["V1", "V2"])
        assert n_paired == 3


class TestCategorizeCelltype:
    """Test cell type categorization helper."""

    def test_categorize_cd4_t(self):
        """Test CD4 T cell categorization."""
        assert categorize_celltype("CD4 T cell") == "CD4_T"
        assert categorize_celltype("CD4_memory") == "CD4_T"
        assert categorize_celltype("Treg") == "CD4_T"
        assert categorize_celltype("Th1") == "CD4_T"

    def test_categorize_cd8_t(self):
        """Test CD8 T cell categorization."""
        assert categorize_celltype("CD8 T cell") == "CD8_T"
        assert categorize_celltype("CD8_cytotoxic") == "CD8_T"

    def test_categorize_nk(self):
        """Test NK cell categorization."""
        assert categorize_celltype("NK cell") == "NK"
        assert categorize_celltype("Natural Killer") == "NK"

    def test_categorize_b_cells(self):
        """Test B cell categorization."""
        assert categorize_celltype("B cell") == "B_cells"
        assert categorize_celltype("Plasma cell") == "B_cells"

    def test_categorize_monocytes(self):
        """Test monocyte categorization."""
        assert categorize_celltype("Monocyte") == "Monocytes"
        assert categorize_celltype("CD14+ Mono") == "Monocytes"
        assert categorize_celltype("CD16+ Mono") == "Monocytes"

    def test_categorize_dcs(self):
        """Test dendritic cell categorization."""
        assert categorize_celltype("DC") == "DCs"
        assert categorize_celltype("Dendritic cell") == "DCs"

    def test_categorize_truly_unknown(self):
        """Test unknown cell type categorization."""
        # Only test truly unknown types that don't match any pattern
        assert categorize_celltype("Fibroblast") == "Other"
        assert categorize_celltype("Epithelial") == "Other"


class TestEnsureFDR:
    """Test FDR calculation helper."""

    def test_ensure_fdr_adds_column(self):
        """Test that FDR column is added."""
        df = pd.DataFrame({
            "feature": ["G1", "G2", "G3"],
            "p_time": [0.001, 0.05, 0.5],
        })
        result = ensure_fdr(df, p_col="p_time", fdr_col="FDR_time")
        assert "FDR_time" in result.columns
        assert result["FDR_time"].notna().all()

    def test_ensure_fdr_preserves_existing(self):
        """Test that existing FDR column is preserved."""
        df = pd.DataFrame({
            "feature": ["G1", "G2"],
            "p_time": [0.001, 0.05],
            "FDR_time": [0.01, 0.1],
        })
        result = ensure_fdr(df, p_col="p_time", fdr_col="FDR_time")
        # Should not recalculate
        assert result["FDR_time"].tolist() == [0.01, 0.1]

    def test_ensure_fdr_with_na(self):
        """Test FDR calculation with NA p-values."""
        df = pd.DataFrame({
            "feature": ["G1", "G2", "G3"],
            "p_time": [0.001, np.nan, 0.5],
        })
        result = ensure_fdr(df, p_col="p_time", fdr_col="FDR_time")
        assert result["FDR_time"].notna().sum() == 2  # Only 2 valid p-values

    def test_ensure_fdr_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["feature", "p_time"])
        result = ensure_fdr(df, p_col="p_time", fdr_col="FDR_time")
        assert result.empty

    def test_ensure_fdr_ordering(self):
        """Test that FDR is correctly ordered."""
        df = pd.DataFrame({
            "feature": ["G1", "G2", "G3", "G4"],
            "p_time": [0.001, 0.01, 0.05, 0.1],
        })
        result = ensure_fdr(df, p_col="p_time", fdr_col="FDR_time")
        # FDR should be monotonically increasing (or equal) for increasing p-values
        fdr_values = result["FDR_time"].values
        assert all(fdr_values[i] <= fdr_values[i + 1] for i in range(len(fdr_values) - 1))
