"""sctrial: trial-aware inference utilities for single-cell AnnData."""

from __future__ import annotations
from importlib.metadata import version as _pkg_version
from .design import TrialDesign
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
from .adata_tools import subset_primary, subset_cells, profile_features
from .utils import resolve_feature
from .stats.did import did_table, did_fit, did_table_by_celltype
from .stats.abundance import abundance_did
from .stats.gsea import run_gsea_did
from .stats.comparisons import within_arm_comparison, between_arm_comparison
from .stats.summary import summarize_did_results
from .plotting import (
    plot_trial_interaction, 
    plot_abundance_interaction,
    plot_did_forest,
    plot_within_arm_comparison,
    plot_trial_umap,
    plot_trial_umap_panel,
    plot_gsea_radar,
    plot_gsea_heatmap,
    plot_trial_dotplot
)


__all__ = [
    "TrialDesign",
    "add_log1p_cpm_layer",
    "score_gene_sets",
    "subset_primary",
    "subset_cells",
    "profile_features",
    "resolve_feature",
    "did_table",
    "did_fit",
    "did_table_by_celltype",
    "abundance_did",
    "run_gsea_did",
    "within_arm_comparison",
    "between_arm_comparison",
    "summarize_did_results",
    "plot_trial_interaction",
    "plot_abundance_interaction",
    "plot_did_forest",
    "plot_within_arm_comparison",
    "plot_trial_umap",
    "plot_trial_umap_panel",
    "plot_gsea_radar",
    "plot_gsea_heatmap",
    "plot_trial_dotplot",
]

try:
    __version__ = _pkg_version("sctrial")
except Exception:  # pragma: no cover
    __version__ = "0.2.1"