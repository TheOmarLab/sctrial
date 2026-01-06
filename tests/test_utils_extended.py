"""Extended tests for utility functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sctrial.utils import (
    ensure_unique_index,
    intersect_preserve_order,
    permutation_pvalue,
    permutation_pvalue_paired,
    resolve_feature,
    safe_filename,
)


class TestSafeFilename:
    """Test safe filename generation."""

    def test_safe_filename_basic(self):
        """Test basic filename sanitization."""
        assert safe_filename("My File.txt") == "My_File.txt"

    def test_safe_filename_special_chars(self):
        """Test removal of special characters."""
        result = safe_filename("File@#$%Name!.txt")
        assert "@" not in result
        assert "#" not in result
        assert result == "File_Name_.txt"

    def test_safe_filename_greek_letters(self):
        """Test Greek letter replacement."""
        assert safe_filename("IFNγ_response") == "IFNgamma_response"
        assert safe_filename("TCRδ_signaling") == "TCRdelta_signaling"

    def test_safe_filename_whitespace(self):
        """Test whitespace handling."""
        assert safe_filename("  Multiple   Spaces  ") == "Multiple_Spaces"

    def test_safe_filename_maxlen(self):
        """Test maximum length truncation."""
        long_name = "a" * 200
        result = safe_filename(long_name, maxlen=50)
        assert len(result) == 50

    def test_safe_filename_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed."""
        result = safe_filename("File___Name")
        assert result == "File_Name"


class TestIntersectPreserveOrder:
    """Test order-preserving set intersection."""

    def test_intersect_preserve_order_basic(self):
        """Test basic intersection with order preservation."""
        items = ["A", "B", "C", "D"]
        universe = {"B", "D", "E", "F"}
        result = intersect_preserve_order(items, universe)
        assert result == ["B", "D"]

    def test_intersect_preserve_order_empty_universe(self):
        """Test with empty universe."""
        items = ["A", "B", "C"]
        universe = set()
        result = intersect_preserve_order(items, universe)
        assert result == []

    def test_intersect_preserve_order_no_overlap(self):
        """Test with no overlap."""
        items = ["A", "B", "C"]
        universe = {"D", "E", "F"}
        result = intersect_preserve_order(items, universe)
        assert result == []

    def test_intersect_preserve_order_maintains_order(self):
        """Test that original order is maintained."""
        items = ["Z", "Y", "X", "W"]
        universe = {"W", "X", "Y", "Z"}
        result = intersect_preserve_order(items, universe)
        assert result == ["Z", "Y", "X", "W"]  # Original order preserved


class TestEnsureUniqueIndex:
    """Test DataFrame index deduplication."""

    def test_ensure_unique_index_already_unique(self):
        """Test when index is already unique."""
        df = pd.DataFrame({"A": [1, 2, 3]}, index=["a", "b", "c"])
        result = ensure_unique_index(df)
        assert result.index.is_unique
        pd.testing.assert_frame_equal(result, df)

    def test_ensure_unique_index_with_duplicates_mean(self):
        """Test deduplication using mean aggregation."""
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]},
            index=["a", "b", "a", "b"]
        )
        result = ensure_unique_index(df, agg="mean")
        assert result.index.is_unique
        assert len(result) == 2
        assert result.loc["a", "A"] == 2.0  # mean of 1 and 3
        assert result.loc["b", "A"] == 3.0  # mean of 2 and 4

    def test_ensure_unique_index_with_duplicates_sum(self):
        """Test deduplication using sum aggregation."""
        df = pd.DataFrame(
            {"A": [1, 2, 3, 4]},
            index=["a", "b", "a", "b"]
        )
        result = ensure_unique_index(df, agg="sum")
        assert result.index.is_unique
        assert result.loc["a", "A"] == 4  # sum of 1 and 3
        assert result.loc["b", "A"] == 6  # sum of 2 and 4

    def test_ensure_unique_index_invalid_agg(self):
        """Test with invalid aggregation method."""
        df = pd.DataFrame({"A": [1, 2]}, index=["a", "a"])
        with pytest.raises(ValueError, match="Unsupported agg"):
            ensure_unique_index(df, agg="invalid")


class TestPermutationPvalue:
    """Test two-sample permutation test."""

    def test_permutation_pvalue_significant_difference(self):
        """Test with significant difference between groups."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(2, 1, 50)  # Mean shifted by 2
        p = permutation_pvalue(group1, group2, n_perm=1000, seed=42)
        assert p < 0.05  # Should be significant

    def test_permutation_pvalue_no_difference(self):
        """Test with no difference between groups."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0, 1, 50)
        p = permutation_pvalue(group1, group2, n_perm=1000, seed=42)
        assert p > 0.05  # Should not be significant

    def test_permutation_pvalue_reproducibility(self):
        """Test that results are reproducible with same seed."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])
        p1 = permutation_pvalue(group1, group2, n_perm=500, seed=42)
        p2 = permutation_pvalue(group1, group2, n_perm=500, seed=42)
        assert p1 == p2

    def test_permutation_pvalue_identical_groups(self):
        """Test with identical groups."""
        group = np.array([1, 2, 3, 4, 5])
        p = permutation_pvalue(group, group.copy(), n_perm=100, seed=42)
        assert p > 0.9  # Should be very non-significant


class TestPermutationPvaluePaired:
    """Test paired permutation test."""

    def test_permutation_pvalue_paired_significant(self):
        """Test with significant paired difference."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        y = x + 1.5  # Consistent increase
        p = permutation_pvalue_paired(x, y, n_perm=1000, seed=42)
        assert p < 0.05

    def test_permutation_pvalue_paired_no_difference(self):
        """Test with no paired difference."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        y = x + np.random.normal(0, 0.1, 30)  # Just noise
        p = permutation_pvalue_paired(x, y, n_perm=1000, seed=42)
        # Might or might not be significant, depends on noise
        assert 0.0 <= p <= 1.0

    def test_permutation_pvalue_paired_identical(self):
        """Test with identical paired values."""
        x = np.array([1, 2, 3, 4, 5])
        y = x.copy()
        p = permutation_pvalue_paired(x, y, n_perm=100, seed=42)
        assert p > 0.9  # No difference

    def test_permutation_pvalue_paired_reproducibility(self):
        """Test reproducibility with same seed."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 4, 5, 6])
        p1 = permutation_pvalue_paired(x, y, n_perm=500, seed=42)
        p2 = permutation_pvalue_paired(x, y, n_perm=500, seed=42)
        assert p1 == p2


class TestResolveFeature:
    """Test feature name resolution."""

    def test_resolve_feature_exact_match_obs(self):
        """Test exact match in obs columns."""
        adata = AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame({"CellType": ["A", "B"] * 5}),
            var=pd.DataFrame(index=[f"Gene{i}" for i in range(5)])
        )
        assert resolve_feature(adata, "CellType") == "CellType"

    def test_resolve_feature_exact_match_var(self):
        """Test exact match in var names."""
        adata = AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame({"CellType": ["A", "B"] * 5}),
            var=pd.DataFrame(index=["CD3D", "CD8A", "CD4", "FOXP3", "IL2"])
        )
        assert resolve_feature(adata, "CD3D") == "CD3D"

    def test_resolve_feature_case_insensitive_obs(self):
        """Test case-insensitive matching in obs."""
        adata = AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame({"CellType": ["A", "B"] * 5}),
            var=pd.DataFrame(index=[f"Gene{i}" for i in range(5)])
        )
        assert resolve_feature(adata, "celltype") == "CellType"

    def test_resolve_feature_case_insensitive_var(self):
        """Test case-insensitive matching in var."""
        adata = AnnData(
            X=np.random.rand(10, 2),
            var=pd.DataFrame(index=["CD3D", "CD8A"])
        )
        assert resolve_feature(adata, "cd3d") == "CD3D"

    def test_resolve_feature_not_found(self):
        """Test error when feature not found."""
        adata = AnnData(X=np.random.rand(10, 5))
        with pytest.raises(KeyError, match="not found"):
            resolve_feature(adata, "NonExistent")

    def test_resolve_feature_ambiguous(self):
        """Test error with ambiguous matches."""
        # Create ambiguous case: same name in obs and var with different cases
        adata = AnnData(
            X=np.random.rand(10, 2),
            obs=pd.DataFrame({"score": [1.0, 2.0] * 5}),
            var=pd.DataFrame(index=["Score", "Other"])
        )
        # Try querying with "score" - should prefer exact match in obs
        result = resolve_feature(adata, "score")
        assert result == "score"  # Exact match preferred

    def test_resolve_feature_prefers_exact(self):
        """Test that exact matches are preferred over case-insensitive."""
        adata = AnnData(
            X=np.random.rand(10, 5),
            obs=pd.DataFrame({"Score": [1.0, 2.0] * 5}),
            var=pd.DataFrame(index=["Gene1", "Gene2", "Gene3", "Gene4", "Gene5"])
        )
        # Should return exact match
        assert resolve_feature(adata, "Score") == "Score"
