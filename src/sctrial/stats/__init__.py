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
from .comparisons import between_arm_comparison, within_arm_comparison
from .did import did_fit, did_table, did_table_by_celltype
from .gsea import run_gsea_did
from .pseudobulk import pseudobulk_expression, pseudobulk_within_arm
from .summary import summarize_did_results

# Effect size calculations
from .effect_size import (
    cohens_d,
    hedges_g,
    cohens_d_from_did,
    effect_size_ci,
    add_effect_sizes_to_did,
    bootstrap_effect_size_ci,
)

# Power analysis
from .power import (
    power_did,
    sample_size_did,
    power_curve,
    design_effect,
    effective_sample_size,
)

# Mixed effects models
from .mixed_effects import (
    did_mixed,
    did_table_mixed,
    compare_fixed_vs_mixed,
)

# Time series analysis
from .timeseries import (
    trend_interaction,
    event_study_did,
    polynomial_trend,
    test_parallel_trends,
)

# Cross-validation
from .cv import (
    loo_cv_did,
    kfold_cv_did,
    influence_diagnostics,
    cv_summary,
)

__all__ = [
    # Core DiD
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
]
