"""sctrial: trial-aware inference utilities for single-cell AnnData."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .adata_tools import profile_features, subset_cells, subset_primary
from .analysis import DiDAnalyzer
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
    did_volcano_frame,
    plot_abundance_interaction,
    plot_did_forest,
    plot_did_forest_interactive,
    plot_did_volcano_interactive,
    plot_gsea_heatmap,
    plot_gsea_radar,
    plot_module_umap_panel,
    plot_parallel_trends,
    plot_trial_dotplot,
    plot_trial_interaction,
    plot_trial_umap,
    plot_trial_umap_panel,
    plot_within_arm_comparison,
    signed_logp,
)
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets, score_gene_sets_aucell
from .stats._extract import extract_gene_vector
from .stats.abundance import abundance_did
from .stats.bayes import did_table_bayes, prior_predictive_check
from .stats.comparisons import (
    between_arm_comparison,
    compare_gene_in_celltype,
    within_arm_comparison,
)
from .stats.cv import cv_summary, influence_diagnostics, kfold_cv_did, loo_cv_did
from .stats.diagnostics import check_did_assumptions
from .stats.did import DiDConfig, did_fit, did_table, did_table_by_celltype, did_table_parallel
from .stats.effect_size import (
    add_effect_sizes_to_did,
    bootstrap_effect_size_ci,
    cohens_d,
    cohens_d_from_did,
    effect_size_ci,
    hedges_g,
)
from .stats.gsea import (
    run_gsea_did,
    run_gsea_did_by_celltype,
    run_gsea_did_multi,
    run_gsea_pseudobulk,
)
from .stats.heterogeneity import test_treatment_heterogeneity
from .stats.mixed_effects import compare_fixed_vs_mixed, did_mixed, did_table_mixed
from .stats.module_scores import (
    module_score_did_by_pool,
    module_score_pseudobulk,
    module_score_within_arm_by_pool,
)
from .stats.power import (
    design_effect,
    effective_sample_size,
    power_curve,
    power_did,
    sample_size_did,
    sensitivity_analysis,
)
from .stats.pseudobulk import (
    pseudobulk_did,
    pseudobulk_export,
    pseudobulk_expression,
    pseudobulk_within_arm,
)
from .stats.sensitivity import e_value_rr
from .stats.summary import summarize_did_results
from .stats.survival import hazard_regression_with_features
from .stats.timeseries import (
    event_study_did,
    polynomial_trend,
    test_parallel_trends,
    trend_interaction,
)
from .utils import resolve_feature
from .validation import (
    TrialDataValidator,
    check_covariate_balance,
    diagnose_trial_data,
    validate_adata,
    validate_features,
)
from .workflow import TrialWorkflow, workflow

__all__ = [
    # Core design
    "TrialDesign",
    "DiDAnalyzer",
    "DiDConfig",
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
    "did_table_parallel",
    "did_table_bayes",
    "prior_predictive_check",
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
    "pseudobulk_export",
    "module_score_pseudobulk",
    "module_score_did_by_pool",
    "module_score_within_arm_by_pool",
    "compare_gene_in_celltype",
    # Effect sizes
    "cohens_d",
    "hedges_g",
    "cohens_d_from_did",
    "effect_size_ci",
    "add_effect_sizes_to_did",
    "bootstrap_effect_size_ci",
    # Power analysis
    "power_did",
    "sample_size_did",
    "power_curve",
    "design_effect",
    "effective_sample_size",
    # Mixed effects
    "did_mixed",
    "did_table_mixed",
    "compare_fixed_vs_mixed",
    # Workflow
    "TrialWorkflow",
    "workflow",
    # Time series
    "trend_interaction",
    "event_study_did",
    "polynomial_trend",
    "test_parallel_trends",
    # Cross-validation
    "loo_cv_did",
    "kfold_cv_did",
    "influence_diagnostics",
    "cv_summary",
    # Diagnostics
    "check_did_assumptions",
    # Plotting helpers
    "did_volcano_frame",
    "signed_logp",
    # Plotting
    "plot_trial_interaction",
    "plot_parallel_trends",
    "plot_abundance_interaction",
    "plot_did_forest",
    "plot_did_forest_interactive",
    "plot_did_volcano_interactive",
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
    "check_covariate_balance",
    # Heterogeneity
    "test_treatment_heterogeneity",
    # Sensitivity
    "e_value_rr",
    "sensitivity_analysis",
    # Survival
    "hazard_regression_with_features",
    "TrialDataValidator",
    # Convenience
    "quick_did",
    "auto_detect_design",
]

try:
    __version__ = _pkg_version("sctrial")
except (PackageNotFoundError, ValueError):  # pragma: no cover
    __version__ = "0.2.1.dev1"
