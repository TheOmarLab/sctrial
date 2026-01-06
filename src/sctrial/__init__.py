"""sctrial: trial-aware inference utilities for single-cell AnnData."""

from __future__ import annotations
from importlib.metadata import version as _pkg_version
from .design import TrialDesign
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
from .adata_tools import subset_primary, subset_cells, profile_features
from .utils import resolve_feature
from .datasets import (
    load_sade_feldman,
    load_stephenson_data,
    load_vaccine_gse171964,
    count_paired,
    verify_paired_participants,
    categorize_celltype,
    ensure_fdr,
)
from .stats._extract import extract_gene_vector
from .stats.did import did_table, did_fit, did_table_by_celltype
from .stats.abundance import abundance_did
from .stats.gsea import run_gsea_did
from .stats.comparisons import within_arm_comparison, between_arm_comparison
from .stats.summary import summarize_did_results
from .stats.pseudobulk import pseudobulk_expression, pseudobulk_within_arm
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
from .validation import (
    validate_adata,
    validate_features,
    diagnose_trial_data,
    TrialDataValidator,
)
from .convenience import quick_did, auto_detect_design


__all__ = [
    "TrialDesign",
    "add_log1p_cpm_layer",
    "score_gene_sets",
    "subset_primary",
    "subset_cells",
    "profile_features",
    "resolve_feature",
    "load_sade_feldman",
    "load_stephenson_data",
    "load_vaccine_gse171964",
    "count_paired",
    "verify_paired_participants",
    "categorize_celltype",
    "ensure_fdr",
    "extract_gene_vector",
    "did_table",
    "did_fit",
    "did_table_by_celltype",
    "abundance_did",
    "run_gsea_did",
    "within_arm_comparison",
    "between_arm_comparison",
    "summarize_did_results",
    "pseudobulk_expression",
    "pseudobulk_within_arm",
    "plot_trial_interaction",
    "plot_abundance_interaction",
    "plot_did_forest",
    "plot_within_arm_comparison",
    "plot_trial_umap",
    "plot_trial_umap_panel",
    "plot_gsea_radar",
    "plot_gsea_heatmap",
    "plot_trial_dotplot",
    "validate_adata",
    "validate_features",
    "diagnose_trial_data",
    "TrialDataValidator",
    "quick_did",
    "auto_detect_design",
]

try:
    __version__ = _pkg_version("sctrial")
except Exception:  # pragma: no cover
    __version__ = "0.2.1.dev1"
