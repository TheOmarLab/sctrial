"""sctrial: trial-aware inference utilities for single-cell AnnData."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .adata_tools import profile_features, subset_cells, subset_primary
from .convenience import auto_detect_design, quick_did
from .datasets import (
    categorize_celltype,
    count_paired,
    ensure_fdr,
    load_sade_feldman,
    load_stephenson_data,
    load_vaccine_gse171964,
    verify_paired_participants,
)
from .design import TrialDesign
from .plotting import (
    plot_abundance_interaction,
    plot_did_forest,
    plot_gsea_heatmap,
    plot_gsea_radar,
    plot_trial_dotplot,
    plot_trial_interaction,
    plot_trial_umap,
    plot_trial_umap_panel,
    plot_module_umap_panel,
    plot_within_arm_comparison,
)
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets, score_gene_sets_aucell
from .stats._extract import extract_gene_vector
from .stats.abundance import abundance_did
from .stats.comparisons import (
    between_arm_comparison,
    compare_gene_in_celltype,
    within_arm_comparison,
)
from .stats.cv import cv_summary, influence_diagnostics, kfold_cv_did, loo_cv_did
from .stats.did import did_fit, did_table, did_table_by_celltype
from .stats.effect_size import (
    add_effect_sizes_to_did,
    bootstrap_effect_size_ci,
    cohens_d,
    hedges_g,
)
from .stats.gsea import (
    run_gsea_did,
    run_gsea_did_by_celltype,
    run_gsea_did_multi,
    run_gsea_pseudobulk,
)
from .stats.mixed_effects import compare_fixed_vs_mixed, did_table_mixed
from .stats.power import (
    design_effect,
    effective_sample_size,
    power_curve,
    power_did,
    sample_size_did,
)
from .stats.pseudobulk import (
    pseudobulk_did,
    pseudobulk_expression,
    pseudobulk_within_arm,
)
from .stats.module_scores import (
    module_score_did_by_pool,
    module_score_pseudobulk,
    module_score_within_arm_by_pool,
)
from .stats.summary import summarize_did_results
from .stats.timeseries import event_study_did, test_parallel_trends, trend_interaction
from .utils import resolve_feature
from .validation import (
    TrialDataValidator,
    diagnose_trial_data,
    validate_adata,
    validate_features,
)

__all__ = [
    # Core design
    "TrialDesign",
    # Preprocessing
    "add_log1p_cpm_layer",
    "score_gene_sets",
    "score_gene_sets_aucell",
    # Data tools
    "subset_primary",
    "subset_cells",
    "profile_features",
    "resolve_feature",
    # Datasets
    "load_sade_feldman",
    "load_stephenson_data",
    "load_vaccine_gse171964",
    "count_paired",
    "verify_paired_participants",
    "categorize_celltype",
    "ensure_fdr",
    "extract_gene_vector",
    # Core statistics
    "did_table",
    "did_fit",
    "did_table_by_celltype",
    "abundance_did",
    "run_gsea_did",
    "run_gsea_did_multi",
    "run_gsea_did_by_celltype",
    "run_gsea_pseudobulk",
    "within_arm_comparison",
    "between_arm_comparison",
    "summarize_did_results",
    "pseudobulk_expression",
    "pseudobulk_within_arm",
    "pseudobulk_did",
    "module_score_pseudobulk",
    "module_score_did_by_pool",
    "module_score_within_arm_by_pool",
    "compare_gene_in_celltype",
    # Effect sizes
    "cohens_d",
    "hedges_g",
    "add_effect_sizes_to_did",
    "bootstrap_effect_size_ci",
    # Power analysis
    "power_did",
    "sample_size_did",
    "power_curve",
    "design_effect",
    "effective_sample_size",
    # Mixed effects
    "did_table_mixed",
    "compare_fixed_vs_mixed",
    # Time series
    "trend_interaction",
    "event_study_did",
    "test_parallel_trends",
    # Cross-validation
    "loo_cv_did",
    "kfold_cv_did",
    "influence_diagnostics",
    "cv_summary",
    # Plotting
    "plot_trial_interaction",
    "plot_abundance_interaction",
    "plot_did_forest",
    "plot_within_arm_comparison",
    "plot_trial_umap",
    "plot_trial_umap_panel",
    "plot_module_umap_panel",
    "plot_gsea_radar",
    "plot_gsea_heatmap",
    "plot_trial_dotplot",
    # Validation
    "validate_adata",
    "validate_features",
    "diagnose_trial_data",
    "TrialDataValidator",
    # Convenience
    "quick_did",
    "auto_detect_design",
]

try:
    __version__ = _pkg_version("sctrial")
except (PackageNotFoundError, ValueError):  # pragma: no cover
    __version__ = "0.2.1.dev1"
