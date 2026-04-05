# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-04-05

### Added
- `sctrial.benchmark` subpackage: hierarchical gamma-Poisson simulator (`SimulationConfig`, `simulate_trial`, `calibrate_from_real_data`) and benchmark orchestrator (`run_benchmark`) for controlled method comparisons
- `run_gsea_cross_sectional()` and `run_gsea_within_arm()` now exported from `sctrial.stats`
- `sensitivity_analysis()` now exported from `sctrial.stats`
- Concepts and Methods documentation page with statistical background and references
- Changelog page in documentation
- API reference split into navigable sections (Design, Statistics, Scoring, Plotting, Datasets, Workflow, Utilities, Benchmark)

### Fixed
- `stats/__init__.py`: missing exports for `run_gsea_cross_sectional`, `run_gsea_within_arm`, and `sensitivity_analysis`
- Documentation URLs standardized to Read the Docs

### Improved
- CITATION.cff updated to v0.3.0 with full author list matching the preprint
- README updated with preprint citation, PyPI badge, and corrected API links
- Module-level docstrings added to 8 core modules
- Tutorial notebooks cleaned up: removed AI-generated language, use sctrial API consistently

## [0.3.0] - 2026-03-18

### Added

#### Single-Arm Power Analysis (`sctrial.stats.power`)
- `power_paired()`: Power for paired pre/post designs (vaccine, CAR-T, etc.)
- `sample_size_paired()`: Required sample size for paired designs
- `sensitivity_paired()`: Minimum detectable effect size for paired designs

#### Single-Arm Design Support (`sctrial.design`)
- `TrialDesign.arm_col` now accepts `None` for single-arm studies
- `required_cols()`, `validate()`, `subset_cells()` handle `arm_col=None`
- `arm_bin()` raises clear `ValueError` for single-arm designs

#### Monte Carlo Simulation (`sctrial.stats.simulation`)
- `simulate_did_data()`: Generate synthetic DiD datasets with known ground truth
- `run_method_comparison()`: Compare sctrial DiD vs pseudobulk OLS vs Wilcoxon

### Fixed
- `n_boot` default: 1000 → 999 in `bootstrap_effect_size_ci()`
- Zero-SE paired delta returns `NaN` instead of `t=0, p=1`
- Non-finite SE coerced to `NaN` instead of `0.0`
- `validate()` no longer crashes when `arm_col=None`
- `check_covariate_balance()` raises clear error for single-arm designs
- mypy: fixed `catch_warnings` type in `comparisons.py`

#### Bayesian DiD (`sctrial.stats.bayes`)
- `prior_predictive_check()`: Run prior predictive checks to calibrate Bayesian priors before fitting.
- `did_table_bayes()` now accepts `prior_scale` and `sigma_scale` parameters for prior sensitivity analysis.

#### Effect Size Module (`sctrial.stats.effect_size`)
- `cohens_d()`: Calculate Cohen's d effect size between two groups
- `hedges_g()`: Bias-corrected effect size (recommended for n < 20)
- `add_effect_sizes_to_did()`: Add standardized effect sizes to DiD results
- `bootstrap_effect_size_ci()`: Bootstrap confidence intervals for effect sizes
- `effect_size_ci()`: Analytical confidence intervals using noncentral t

#### Power Analysis Module (`sctrial.stats.power`)
- `power_did()`: Calculate power for DiD analysis given sample size
- `sample_size_did()`: Determine required sample size for desired power
- `power_curve()`: Generate power curves across sample sizes
- `design_effect()`: Calculate design effect for clustered data
- `effective_sample_size()`: Compute effective n accounting for ICC

#### Mixed Effects Models (`sctrial.stats.mixed_effects`)
- `did_table_mixed()`: DiD with random participant effects
- `compare_fixed_vs_mixed()`: Compare both approaches for sensitivity analysis
- Returns ICC (intraclass correlation) and variance components

#### Time Series Analysis (`sctrial.stats.timeseries`)
- `trend_interaction()`: Treatment × time trend for 3+ timepoints
- `event_study_did()`: Generalized DiD comparing each visit to baseline
- `polynomial_trend()`: Fit polynomial trajectories by arm
- `test_parallel_trends()`: Validate parallel trends assumption

#### Cross-Validation (`sctrial.stats.cv`)
- `loo_cv_did()`: Leave-one-out CV for influence diagnostics
- `kfold_cv_did()`: K-fold CV for effect stability assessment
- `influence_diagnostics()`: Identify influential participants
- `cv_summary()`: Summary statistics from CV results

### Improved
- `add_effect_sizes_to_did()` now uses residual SD directly from the OLS/WLS fit when available, instead of back-calculating from standard errors with a balanced-design assumption.
- `module_score_did_by_pool()` now accepts `fdr_global` parameter; emits a global FDR column (`FDR_DiD_global`) and a warning when per-group FDR is used.
- `kfold_cv_did()` now stratifies fold assignment by treatment arm for balanced folds.
- `did_table_mixed()` convergence handling improved: trusts `fit.converged` as primary indicator with `lbfgs` fallback optimizer.
- `design_effect()` and `effective_sample_size()` now validate inputs (ICC, cluster_size).
- Missing feature error messages in `did_fit()` now show total count when more than 5 are missing.
- `add_log1p_cpm_layer()` now supports `overwrite=True` to replace an existing layer and `inplace=True` to modify the AnnData in place without copying.
- `_to_bool_series()` now emits a `WARNING`-level log when non-finite values are encountered in a boolean column, aiding data quality diagnostics.
- `profile_features()` docstring now documents that aggregation is cell-weighted; for balanced participant-level comparisons, pre-aggregate to pseudobulk first.
- Replaced `print()` with `logging` throughout `convenience`, `datasets`, and `validation` modules.
- Enhanced `did_fit()` docstring with full mathematical model specification
- Added explicit null hypothesis statements to all statistical functions
- Expanded FAQ with bootstrap vs standard errors guidance
- Expanded FAQ with minimum sample size recommendations
- Expanded FAQ with missing data handling guidance
- Reorganized API documentation by statistical method category

### Fixed
- **Statistical**: Fixed WLS weighting in `did_fit()` — now uses `n_cells` (correct inverse-variance weights) instead of `sqrt(n_cells)` for pre-aggregated participant-level means.
- **Statistical**: Fixed `pseudobulk_expression()` to drop groups with zero total counts before CPM normalization instead of adding a `1e-12` epsilon.
- **Preprocessing**: `add_log1p_cpm_layer()` now validates input: raises `ValueError` on negative counts, warns on NaN/inf values, and warns when all cells in the counts layer are zero.
- **Preprocessing**: `add_log1p_cpm_layer()` now records provenance metadata in `adata.uns["log1p_cpm_info"]` (source layer, target layer, timestamp).
- **adata_tools**: Fixed `_to_bool_series()` truncating fractional crossover values (e.g. `0.5` was cast to `int` → `0` → `False`). Now any non-zero finite value is truthy.
- **adata_tools**: Fixed `_to_bool_series()` crashing with `IntCastingNaNError` on `np.inf` crossover values. Non-finite values (NaN, inf) are now treated as `False` with a logged warning.
- **analysis**: Fixed `DiDAnalyzer.fit()` returning a mutable alias to internal `results_` DataFrame. Now returns a copy so external mutations cannot corrupt internal state.
- Fixed duplicate `_params_match()` definition in `datasets.py` that silently overwrote the robust version; also fixed numpy array comparison bug.
- Fixed README.md example with incorrect `arm_col` parameter value.
- Fixed `auto_detect_design` docstring example showing incorrect mutation of frozen dataclass.
- Fixed `test_plot_gsea_heatmap` test to skip when matplotlib is not installed.

### Internal
- Added `__all__` exports to all modules for better API clarity: `design`, `preprocessing`, `scoring`, `adata_tools`, `utils`, `plotting`.
- Added `validation` and `convenience` modules to API documentation.

## [0.2.0] - 2026-01-01

### Added
- Core trial-aware inference engine: Difference-in-Differences (DiD) with participant fixed effects and covariate support.
- Robust statistical methods: Wild Cluster Bootstrap (Rademacher) and Permutation tests.
- Cell-type abundance DiD for compositional analysis.
- GSEA integration with trial-aware rankings (signed confidence, beta, or t-statistic).
- Stratified DiD analysis across cell-type hierarchies (`did_table_by_celltype`).
- Advanced visualizations: Trial interaction plots, forest plots, spaghetti plots, radar plots, and trial-stratified UMAP panels.
- Automated summary reporting for DiD results.
- Comprehensive Sphinx documentation with API reference and detailed tutorials.
- Professional infrastructure: MIT License, pre-commit hooks, and GitHub Actions CI.
- Full support for sparse `AnnData` objects.
