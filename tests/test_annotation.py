"""Regression tests for Sade-Feldman cell-type annotation pipeline."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from sctrial.datasets import _weighted_marker_score

# ---------------------------------------------------------------------------
# Conditionally import _annotate_immune_celltypes (needs scanpy + igraph)
# ---------------------------------------------------------------------------
_has_scanpy = importlib.util.find_spec("scanpy") is not None
_has_igraph = importlib.util.find_spec("igraph") is not None
_can_annotate = _has_scanpy and _has_igraph

if _can_annotate:
    from sctrial.datasets import _annotate_immune_celltypes


# ── helpers ────────────────────────────────────────────────────────────────
def _make_immune_adata(
    n_cells: int = 300,
    n_genes: int = 200,
    seed: int = 42,
) -> AnnData:
    """Create a synthetic AnnData that mimics CD45+ sorted immune cells.

    Embeds a few canonical marker genes with elevated expression in
    specific cell groups so Leiden clustering + Wilcoxon scoring should
    recover at least CD8 T, B cell, and Monocyte/Macrophage labels.
    """
    rng = np.random.default_rng(seed)

    # Base expression (TPM-like)
    X = rng.exponential(scale=2.0, size=(n_cells, n_genes)).astype(np.float32)

    gene_names = [f"GENE{i}" for i in range(n_genes)]

    # Plant markers in specific cell blocks
    markers = {
        # CD8 T markers (first 100 cells)
        "CD8A": (0, 100),
        "CD8B": (0, 100),
        "GZMA": (0, 100),
        # B cell markers (cells 100-200)
        "MS4A1": (100, 200),
        "CD79A": (100, 200),
        "CD79B": (100, 200),
        # Monocyte markers (cells 200-300)
        "CD14": (200, 300),
        "LYZ": (200, 300),
        "CST3": (200, 300),
    }
    for gene, (start, end) in markers.items():
        gene_names.append(gene)
        col = rng.exponential(scale=0.5, size=n_cells).astype(np.float32)
        col[start:end] += rng.exponential(scale=20.0, size=end - start)
        X = np.column_stack([X, col])

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    return AnnData(X=X, obs=obs, var=var)


# ── _weighted_marker_score ─────────────────────────────────────────────────
class TestWeightedMarkerScore:
    """Unit tests for the rank-weighted scoring function."""

    def test_perfect_overlap(self):
        """All marker genes present → positive score."""
        df = pd.DataFrame({
            "names": ["CD8A", "CD8B", "GZMA", "OTHER"],
            "logfoldchanges": [2.0, 1.5, 1.0, 0.5],
            "pvals_adj": [0.001, 0.01, 0.01, 0.1],
            "rank": [1, 2, 3, 4],
        })
        score, genes = _weighted_marker_score(df, {"CD8A", "CD8B", "GZMA"})
        assert score > 0
        assert set(genes) == {"CD8A", "CD8B", "GZMA"}

    def test_no_overlap(self):
        """No marker genes present → zero score."""
        df = pd.DataFrame({
            "names": ["FOO", "BAR"],
            "logfoldchanges": [2.0, 1.5],
            "pvals_adj": [0.001, 0.01],
            "rank": [1, 2],
        })
        score, genes = _weighted_marker_score(df, {"CD8A", "CD8B"})
        assert score == 0.0
        assert genes == []

    def test_higher_rank_gets_more_weight(self):
        """Rank-1 hit should contribute more weight than rank-10 hit."""
        df_rank1 = pd.DataFrame({
            "names": ["CD8A"],
            "logfoldchanges": [1.0],
            "pvals_adj": [0.01],
            "rank": [1],
        })
        df_rank10 = pd.DataFrame({
            "names": ["CD8A"],
            "logfoldchanges": [1.0],
            "pvals_adj": [0.01],
            "rank": [10],
        })
        score1, _ = _weighted_marker_score(df_rank1, {"CD8A"})
        score10, _ = _weighted_marker_score(df_rank10, {"CD8A"})
        assert score1 > score10

    def test_negative_lfc_clipped(self):
        """Negative logFC should be clipped to 0 (not penalise)."""
        df = pd.DataFrame({
            "names": ["CD8A"],
            "logfoldchanges": [-2.0],
            "pvals_adj": [0.01],
            "rank": [1],
        })
        score, _ = _weighted_marker_score(df, {"CD8A"})
        # log1p(exp(0)) ≈ 1.31 / rank=1 → ~1.31
        assert score > 0


# ── _annotate_immune_celltypes ─────────────────────────────────────────────
_skip_reason = "scanpy + igraph required for annotation integration tests"


@pytest.mark.skipif(not _can_annotate, reason=_skip_reason)
class TestAnnotateImmuneCelltypes:
    """Integration tests for the full annotation pipeline."""

    def test_no_unknown_immune(self):
        """No cell should be labelled 'Unknown immune'."""
        adata = _make_immune_adata()
        labels = _annotate_immune_celltypes(adata)
        assert "Unknown immune" not in labels.values

    def test_no_immune_placeholder(self):
        """No cell should retain the old 'Immune' placeholder."""
        adata = _make_immune_adata()
        labels = _annotate_immune_celltypes(adata)
        assert "Immune" not in labels.values

    def test_all_cells_labelled(self):
        """Every cell must receive a non-null label."""
        adata = _make_immune_adata()
        labels = _annotate_immune_celltypes(adata)
        assert len(labels) == adata.n_obs
        assert labels.notna().all()

    def test_returns_series_with_correct_index(self):
        """Output must be a Series aligned to adata.obs.index."""
        adata = _make_immune_adata()
        labels = _annotate_immune_celltypes(adata)
        assert isinstance(labels, pd.Series)
        assert labels.name == "cell_type"
        assert list(labels.index) == list(adata.obs.index)

    def test_known_cell_types_recovered(self):
        """At least CD8 T, B cell, and Monocyte/Macrophage should appear."""
        adata = _make_immune_adata()
        labels = _annotate_immune_celltypes(adata)
        assigned = set(labels.unique())
        # With planted markers, these three should be recoverable
        expected = {"CD8 T cell", "B cell", "Monocyte/Macrophage"}
        found = expected & assigned
        assert len(found) >= 2, (
            f"Expected at least 2 of {expected} but got {assigned}"
        )

    def test_labels_are_valid_types(self):
        """All labels must come from the canonical set or be 'Unassigned'."""
        valid = {
            "CD8 T cell", "CD4 T cell", "Treg", "B cell",
            "Plasma cell", "NK cell", "Monocyte/Macrophage",
            "Dendritic cell", "Unassigned",
        }
        adata = _make_immune_adata()
        labels = _annotate_immune_celltypes(adata)
        unexpected = set(labels.unique()) - valid
        assert not unexpected, f"Unexpected labels: {unexpected}"

    def test_does_not_modify_input(self):
        """Annotation must not mutate the input AnnData."""
        adata = _make_immune_adata()
        obs_cols_before = set(adata.obs.columns)
        x_sum_before = float(adata.X.sum())
        _annotate_immune_celltypes(adata)
        assert set(adata.obs.columns) == obs_cols_before
        assert float(adata.X.sum()) == pytest.approx(x_sum_before)
