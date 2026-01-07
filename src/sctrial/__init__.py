"""sctrial: trial-aware inference utilities for single-cell AnnData."""

from __future__ import annotations

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
    plot_within_arm_comparison,
)
from .preprocessing import add_log1p_cpm_layer
from .scoring import score_gene_sets
from .stats._extract import extract_gene_vector
from .stats.abundance import abundance_did
from .stats.comparisons import between_arm_comparison, within_arm_comparison
from .stats.did import did_fit, did_table, did_table_by_celltype
from .stats.gsea import run_gsea_did
from .stats.pseudobulk import pseudobulk_expression, pseudobulk_within_arm
from .stats.summary import summarize_did_results
from .utils import resolve_feature
from .validation import (
    TrialDataValidator,
    diagnose_trial_data,
    validate_adata,
    validate_features,
)

# New: Effect sizes
from .stats.effect_size import (
    cohens_d,
    hedges_g,
    add_effect_sizes_to_did,
    bootstrap_effect_size_ci,
)

# New: Power analysis
from .stats.power import (
    power_did,
    sample_size_did,
    power_curve,
    design_effect,
    effective_sample_size,
)

# New: Mixed effects
from .stats.mixed_effects import (
    did_table_mixed,
    compare_fixed_vs_mixed,
)

# New: Time series
from .stats.timeseries import (
    trend_interaction,
    event_study_did,
    test_parallel_trends,
)

# New: Cross-validation
from .stats.cv import (
    loo_cv_did,
    kfold_cv_did,
    influence_diagnostics,
    cv_summary,
)

__all__ = [
    # Core design
    "TrialDesign",
    # Preprocessing
    "add_log1p_cpm_layer",
    "score_gene_sets",
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
    "within_arm_comparison",
    "between_arm_comparison",
    "summarize_did_results",
    "pseudobulk_expression",
    "pseudobulk_within_arm",
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
except Exception:  # pragma: no cover
    __version__ = "0.2.1.dev1"
