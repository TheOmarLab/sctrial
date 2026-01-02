# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
