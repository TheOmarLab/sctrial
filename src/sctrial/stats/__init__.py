"""Statistical modules for trial-aware single-cell inference.

This subpackage provides the core statistical methods for analyzing
clinical trial data with single-cell resolution.

Core Methods
------------
- **did_table**: Difference-in-Differences with fixed effects
- **abundance_did**: Cell-type proportion changes
- **within_arm_comparison**: Paired longitudinal contrasts
- **between_arm_comparison**: Cross-sectional arm comparisons

Advanced Methods
----------------
- **did_table_mixed**: Mixed effects DiD models
- **trend_interaction**: Multi-timepoint trend analysis
- **event_study_did**: Generalized event study design
- **loo_cv_did**: Leave-one-out cross-validation
- **kfold_cv_did**: K-fold cross-validation

Effect Sizes & Power
--------------------
- **add_effect_sizes_to_did**: Cohen's d / Hedge's g for DiD
- **power_did**: Power calculations
- **sample_size_did**: Sample size determination
"""
from .abundance import abundance_did
from .comparisons import between_arm_comparison, compare_gene_in_celltype, within_arm_comparison
from .cv import cv_summary, influence_diagnostics, kfold_cv_did, loo_cv_did
from .diagnostics import check_did_assumptions
from .did import DiDConfig, did_fit, did_table, did_table_by_celltype
from .effect_size import (
    add_effect_sizes_to_did,
    bootstrap_effect_size_ci,
    cohens_d,
    cohens_d_from_did,
    effect_size_ci,
    hedges_g,
)
from .gsea import run_gsea_did, run_gsea_did_by_celltype, run_gsea_did_multi, run_gsea_pseudobulk
from .heterogeneity import test_treatment_heterogeneity
from .mixed_effects import compare_fixed_vs_mixed, did_mixed, did_table_mixed
from .module_scores import (
    module_score_did_by_pool,
    module_score_pseudobulk,
    module_score_within_arm_by_pool,
)
from .power import design_effect, effective_sample_size, power_curve, power_did, sample_size_did
from .pseudobulk import (
    pseudobulk_did,
    pseudobulk_export,
    pseudobulk_expression,
    pseudobulk_within_arm,
)
from .summary import summarize_did_results
from .timeseries import event_study_did, polynomial_trend, test_parallel_trends, trend_interaction

__all__ = [
    # Core DiD
    "did_table",
    "did_fit",
    "did_table_by_celltype",
    "DiDConfig",
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
    "test_treatment_heterogeneity",
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
]
