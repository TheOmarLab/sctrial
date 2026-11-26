# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-11-25

This release revises the default estimator of the participant-level difference-in-differences (DiD) and the
small-sample inference, and restricts two-arm analyses to the two arms named by the design. Estimates,
p-values and confidence intervals obtained with version 1.0.0 can change. The first section lists every
change that can alter a result, by topic; the sections after it list additions and changes that leave
results as they were.

### Changes that affect results

#### Estimation
- `did_fit()` now fits ordinary least squares (OLS) by default. It previously switched to weighted least squares whenever the input carried an `n_cells` column, so every participant-level DiD estimate from `did_table(aggregate="participant_visit")` and `did_table_parallel()` was weighted by cell count. Cell-count weights assume that all variance in a participant-visit mean is cell-sampling noise; in clinical cohorts between-participant variation dominates and cell counts span two orders of magnitude, so a few participants would determine each estimate. Under equal weighting the two-visit estimate equals the difference between arms in the mean within-participant change, which is the estimand the simulation benchmark evaluates. Weighted fitting remains available through the new `weight_col` argument of `did_fit()`, `did_table()`, `did_table_parallel()` and `DiDConfig`. Participant-level DiD estimates change; cell-level estimates do not.
- Two-arm analyses now use only the two arms named by the design. The treated indicator was defined as equality to `arm_treated` and `arm_control` was never read, so a participant in a third arm was analyzed as a control, and a design whose control label was absent from the data ran with every other participant as control. This affected `did_table()`, `did_table_parallel()`, `did_table_by_celltype()`, `get_did_aggregated_df()`, `did_table_bayes()`, `between_arm_comparison()`, `get_between_arm_aggregated_df()`, `abundance_did()`, `pseudobulk_did()`, `test_treatment_heterogeneity()`, `did_table_mixed()`, `compare_fixed_vs_mixed()`, `trend_interaction()`, `test_parallel_trends()`, `polynomial_trend()`, `event_study_did()`, `loo_cv_did()`, and `kfold_cv_did()`. These functions now build their model frame through `subset_arms()`: rows outside the two named arms are excluded with one warning per call. Data that contain such rows but no row of one named arm raise `ValueError`, as does a design with `arm_treated == arm_control` (for a single arm, use `within_arm_comparison()` or a design with `arm_col=None`). Results change only for data that contain rows outside the two named arms.
- `loo_cv_did()`, `kfold_cv_did()`, `did_table_by_celltype()`, and `event_study_did()` now restrict to the two named arms before their loops, rather than catching the resulting `ValueError` inside them, so a design that does not match the data's arm labels raises immediately instead of being recorded as one failed fit per iteration. `kfold_cv_did()` now assigns folds within the two named arms only; a third arm had previously formed a stratum of its own, which changed the cross-validated estimates.
- `did_fit()`, `between_arm_comparison()`, and `did_table_bayes()` now return a missing estimate when only one arm has participants in the model, for example a cell type present in a single arm. Previously, with only the treated arm the interaction coincided with the time term, so `did_fit()` reported half the time effect as the DiD coefficient with a finite p-value and `did_table_bayes()` split the time effect across the two coefficients; with only the control arm `did_fit()` returned 0, `did_table_bayes()` returned the prior mean, and `between_arm_comparison()` returned a coefficient of 0. These estimates are now reported as missing (NaN).
- `did_fit()` now fits only the participants observed at both visits for the outcome and returns a missing estimate when the interaction is not identified. A participant with a value at only one visit previously remained in the model and was counted in `n_units` and among the clusters of the robust variance. When a whole arm lacked one visit, the visit and interaction terms were the same regressor, so the fit split the other arm's change between them and reported it with a standard error near zero; the same held for a covariate that duplicated the interaction.
- Each feature is now analyzed using only the participants with a value at every visit of the design, matching `did_fit()`. In `did_table_mixed()`, `within_arm_comparison()`, `trend_interaction()`, `did_table_bayes()` and the covariate path of `abundance_did()`, a participant missing a value for one feature previously kept its remaining visits in that feature's model, where it was counted in `n_units` and as a cluster and, in the functions that standardize the outcome, entered that standardization. In `did_table_mixed()` and `trend_interaction()`, `n_units` was additionally reported once for the whole table rather than per feature. Point estimates, standard errors and `n_units` for these analyses change accordingly, and `within_arm_fit_beta()` and `between_arm_comparison()` now standardize the outcome over the rows that actually enter the fit.
- The `agg="pct_pos"` aggregation, the percentage of cells with a positive value, now works in `did_table()`, `did_table_bayes()`, `within_arm_comparison()`, `between_arm_comparison()` and their frame builders, where it previously raised `AttributeError`. `test_treatment_heterogeneity()` now applies the requested `agg` to the features and to numeric covariates instead of always taking the mean. A numeric covariate keeps its mean under `pct_pos`, and an unrecognized `agg` raises `ValueError`.

#### Inference
- The wild cluster bootstrap now imposes the null hypothesis by refitting the model without the tested term. `wild_cluster_bootstrap_t()` had imposed the null by subtracting the estimated coefficient from the unrestricted fit, which leaves the other coefficients at their unrestricted values. In a two-arm DiD this centers each participant's change on the control-arm mean, so the bootstrap p-value depended on which arm was coded as treated and the test was conservative. The function now drops the tested term, refits under the null, and resamples the restricted residuals. P-values of two-arm DiD tests change; in single-arm (within-participant) models the two constructions coincide, so those p-values change only through the resampling fixes below.
- When the participants permit no more sign assignments than the requested number of bootstrap replicates (nine or fewer participants at the default of 999), every assignment is now enumerated rather than sampled. The test is then exact and no longer depends on the random seed.
- Bootstrap confidence intervals are now obtained by inverting the bootstrap test, rather than by applying quantiles of the null bootstrap distribution to the point estimate. The interval contains every value the test does not reject, so it contains zero whenever the bootstrap p-value is at least `1 - ci_level`. In small or unbalanced designs the non-rejected values can form more than one interval; the reported interval then spans them and can contain zero even when the p-value is below `1 - ci_level`. Under enumeration the attainable p-value has a lower bound that depends on the number of participants, so with five or fewer participants the test cannot reject at the 5% level and the interval is unbounded. `did_table()`, `did_table_parallel()`, `pseudobulk_did()`, `within_arm_comparison()`, and `abundance_did()` issue one warning per table when this occurs.
- The bootstrap standard error (`se_DiD_boot`, `se_time_boot`) is now the standard deviation of the unrestricted wild cluster bootstrap, which equals the cluster-robust standard error without the small-sample correction and does not depend on the hypothesized value. It had been the standard deviation of the null-imposed draws, which grows with the size of the estimated effect. `BootstrapResult.boot_distribution` holds the unrestricted estimates.
- The observed test statistic is now studentized with the same variance estimator (selected by `cov_type`) as the bootstrap statistics it is compared against. `abundance_did()` on paired differences had compared a homoskedastic observed statistic against cluster-robust bootstrap statistics, which mismatched the two and distorted the p-value.
- Resampling tests now count a replicate whose statistic equals the observed one as a tie, regardless of floating-point rounding. The wild cluster bootstrap computes the observed and the replicate statistics by different arithmetic paths, so a sign assignment that reproduces the observed statistic in exact arithmetic (for example flipping a participant whose outcome did not change) could be omitted from the count, which understated the exact p-value. A replicate statistic within a small relative tolerance of the observed one now counts as a tie. `permutation_pvalue()`, `permutation_pvalue_paired()`, and the permutation test in `module_score_did_by_pool()` likewise count a mean difference within rounding error of the observed one; the two-sample tests center the pooled values first and the paired test works on the differences, so the result does not depend on the origin or the units of the measurements.
- A bootstrap replicate that the model fits exactly now counts as extreme. Such a replicate has no residual variance and an unbounded test statistic; rounding could leave its variance just above or just below zero, and the sign determined whether the replicate entered the count, so the p-value of the same data could change with the coding of the arms or the units of the outcome.
- The wild cluster bootstrap now applies observation weights correctly, scaling the outcome and the regressors by the square root of the weights so that an unweighted ordinary least squares fit is left unchanged and a weighted least squares fit is bootstrapped on its weighted design. It had inferred weighting from the presence of a `weights` attribute, which statsmodels sets on every model. Only weighted fits are affected.
- Participant fixed effects are now built only from the participants present in the model. AnnData retains every category of a categorical `obs` column, so a participant excluded from a paired analysis (observed at only one visit, or in another arm) still contributed an empty indicator column to the participant factor. These empty columns were counted as parameters in the small-sample correction of the cluster-robust covariance, which inflated the standard error when a few participants were excluded and made it undefined once the parameter count reached the number of observations; `did_fit()` and `within_arm_comparison()` then fell back to the homoskedastic standard error and studentized the wild cluster bootstrap with it. The model frames of `did_fit()`, `did_table()`, `within_arm_comparison()`, `within_arm_fit_beta()`, `between_arm_comparison()`, `abundance_did()`, and the time-series, heterogeneity, and mixed-effects models now drop unused participant categories. Results change only for inputs that carry participants outside the fitted model. For any such input the analytical standard error and p-value change; the bootstrap p-value and interval also change in two-arm designs with unequal arms, but not in single-arm designs or in two-arm designs with equal arms, where the correction rescales the observed and the resampled statistics by the same factor.
- Cluster-robust standard errors no longer grow with redundant regressors. statsmodels bases its small-sample correction on the number of design columns rather than on the rank of the design, so a covariate already determined by the participant fixed effects inflated the standard error without changing the point estimate. `did_fit()`, `within_arm_comparison()`, and `abundance_did()` now fit a covariate that is constant within participant on a full-rank design, and `within_arm_comparison()` returns a missing estimate when a covariate duplicates the visit indicator. In `test_treatment_heterogeneity()`, the arm and biomarker main effects are absorbed by the participant fixed effects instead of entered as separate terms, and the `covariates` argument now reaches the model, where it previously raised an error.
- `test_treatment_heterogeneity()` now treats the biomarker as a participant-level attribute and uses only participants observed at both visits. A participant with no biomarker value was previously assigned to the low subgroup, because a comparison with a missing value is false, and is now excluded and reported; the median split is now taken over participants rather than over cells, so a participant contributing more cells no longer carries more weight; and participants seen at only one visit are no longer kept in the model or counted in `n_units`. When an arm-by-subgroup cell holds fewer than two participants, the coefficient and p-value are returned as missing rather than estimated on a rank-deficient design.
- `bootstrap_effect_size_ci()` now warns when bootstrap replicates with no within-group variation are left out of the interval. `module_score_within_arm_by_pool()`, `module_score_did_by_pool()`, and `pseudobulk_within_arm()` now treat a participant's paired change as absent only when it is exactly zero, rather than when its absolute value falls below 1e-8.
- `between_arm_comparison(method="wilcoxon")` now excludes a participant that has no value for a feature, instead of returning a missing statistic for the entire feature while still counting that participant in `n_units`. `compare_gene_in_celltype()` now drops cells with no value before testing, which previously left a missing p-value beside finite group means, and `loo_cv_did()` no longer returns a row for a missing participant label. These affect between-arm Wilcoxon rank-sum comparisons and per-cell-type single-gene comparisons.
- Collinear (redundant) design columns are now removed before the cluster-robust covariance is computed. Its small-sample correction divides by the number of rows minus the number of design columns, so a design carrying a redundant column (for example four participants with a site covariate) previously raised `ZeroDivisionError` even though it had residual degrees of freedom once that column was dropped. A model with genuinely no residual degrees of freedom now returns the estimate with a non-finite standard error and a missing p-value, as `cov_type="nonrobust"` does, rather than raising; `abundance_did()` with two participants no longer errors. This affects cluster-robust inference in small cohorts.
- The wild cluster bootstrap no longer treats a resample with a small but positive residual variance as having none. A resample variance below `1e-12` of the largest resample variance was counted as zero, so a resample the model fit almost exactly was recorded as extreme regardless of its test statistic, biasing the bootstrap p-value. Each resample's variance is now taken from its own residuals and is treated as zero only within the rounding error of the outcome; when the model fits the observed outcome exactly, the bootstrap returns a missing p-value. This affects bootstrap p-values from `did_fit()` and `within_arm_comparison()`.
- The wild cluster bootstrap p-value no longer depends on the location of the outcome. For outcomes of large magnitude, resamples whose test statistic equals the observed one in exact arithmetic could fall outside the numerical tie tolerance and go uncounted, perturbing the p-value. The outcome is now centered on its mean when the remaining regressors hold a constant, which removes the dependence; this affects bootstrap p-values for high-magnitude outcomes.
- A least-squares model that fits its outcome exactly now reports the point estimate but no test. When every residual is zero in exact arithmetic, the standard error and p-value are computed entirely from rounding error and can be arbitrarily small, falsely flagging as significant a within-arm change that no participant shows or an arm difference with no within-arm spread. `did_fit()`, `within_arm_comparison()`, `between_arm_comparison()`, `test_treatment_heterogeneity()`, `trend_interaction()`, `abundance_did()` and the unpaired path of `module_score_did_by_pool()` now return a missing standard error and p-value for such a feature, removing this source of false significance.
- A result that could not be computed is no longer reported as a passing test. `test_parallel_trends()` now labels a feature with no test `"Not estimable"` rather than `"OK"`; `influence_diagnostics()` leaves `is_influential` missing when the influence is undefined, and `compare_fixed_vs_mixed()` leaves `agreement` missing when either model produced no estimate or test, both as nullable boolean columns instead of the result of comparing a missing value with a threshold, which was always false. `summarize_did_results()` now counts only features that have a p-value as tested and reports the untested ones separately, and `module_score_did_by_pool(allow_unpaired=True)` returns no within-arm p-value, rather than 1.0, for an arm with as many rows as model parameters.

#### Enrichment analysis
- Gene-set enrichment analysis (GSEA) now ranks only genes that have an estimate. `run_gsea_did()`, `run_gsea_within_arm()`, `run_gsea_cross_sectional()`, and the wrappers built on them previously gave a gene whose model could not be fitted a rank of zero; such genes sat in the middle of the ranked list and shifted the enrichment scores of every gene set. A gene now enters the ranking only with the quantities its rank requires: the estimate, together with the standard error or p-value when the chosen metric uses one.
- A GSEA ranking is now the same in every run. Genes with identical data have identical statistics, but statistics that are equal in exact arithmetic can differ in their last bits between runs, and the enrichment score of a gene set depends on where its genes fall among such ties. Ranks are now rounded to ten significant digits and equal ranks are ordered by gene name, so the same data give the same ranking each time.
- `run_gsea_did_by_celltype()` now omits, with a single warning, any cell type in which the DiD cannot be estimated: one present in a single arm, at a single visit, or with fewer than `min_units` participants. Previously such a cell type stopped the call for all cell types, as did a cell type belonging to an arm outside the design, which the default cell-type list could include. Eligibility is assessed after crossover participants are excluded.

#### Multi-visit and mixed models
- `did_mixed()` now accepts `arm_control`. When supplied, the analysis is restricted to the two named arms as in `subset_arms()`: rows in any other arm are excluded with a warning, and a `ValueError` is raised if such rows exist while one named arm has no rows. Without `arm_control`, every row that is not `arm_treated` forms the reference group, as before, and a warning is raised when that group holds more than one label or missing labels.
- `did_table_mixed()` now reads `covariates` from `adata.obs`, consistent with the other table builders. It previously searched the feature columns, so a covariate stored in `obs` was dropped from the model without warning and the intended adjustment was not applied. A numeric covariate is aggregated with the features, any other covariate must be constant within a participant-visit, and a covariate absent from `obs`, or one that is also a feature, now raises; `did_mixed()` likewise now raises on a covariate missing from its table rather than silently skipping it. Mixed-effects DiD tables fit with covariates are affected.
- A coefficient the data do not identify is now reported as missing rather than as a finite estimate with a standard error and p-value. In `trend_interaction()`, when every participant in one arm lacked a visit for a feature, the feature was fit on the other arm alone, where the arm-by-time regressor collapses onto the time regressor and a fraction of the time trend was returned as the interaction; the same arose with more polynomial terms than the visits support, and in `between_arm_comparison()` and `abundance_did()` with a covariate collinear with arm, such as a site that enrolled one arm or a dose that is zero in the control arm. `polynomial_trend()` now raises when a time or arm-by-time term is not identified (for example its default quadratic on two visits), `trend_interaction()` now returns the estimate without a test on a saturated design instead of raising `ZeroDivisionError`, `did_fit()` and `abundance_did()` leave `beta_time` and `p_time` missing when a covariate moves by the same amount for every participant between visits, and the pseudobulk ordinary least squares comparator in `run_method_comparison()` applies the same identification check.

#### Power and sample size
- `sample_size_paired()`, `sample_size_did()`, and `sample_size_two_group()` now return the smallest sample size at which the matching power function reaches the target. For the paired and two-group functions, rounding the closed-form solution up could return a size one larger than needed when the real-valued solution was already an integer up to rounding, so the size returned for a detectable effect could exceed the size that effect was computed from. For two groups, the smaller group is its allocation share of the larger group, rounded up to a whole participant.
- The two-sided power of the t-based functions is now finite for every design. In recent SciPy releases the lower tail of the noncentral t distribution returns NaN at some combinations of degrees of freedom and noncentrality, which stopped `sensitivity_paired()` and `sensitivity_two_group()` at two participants and produced a NaN power for some well-powered designs. The far tail is now computed from the upper tail of the mirrored distribution.

#### Data handling and validation
- `did_fit()` now raises an error on a missing weight, as its documentation states. Previously a row with a missing value in `weight_col` was dropped together with its participant, silently altering the weighted analysis.
- Defined-but-absent levels of categorical columns no longer leak into results. `module_score_did_by_pool(allow_unpaired=True)` emitted a missing-value row for an arm that lacked a visit instead of skipping it; `diagnose_trial_data()` counted cell types that were absent from the data; and the benchmark's real-data arm permutation could replace real arm labels with missing values.
- `did_mixed()`, `test_treatment_heterogeneity()`, `trend_interaction()` and `abundance_did()` now build each model and its participant labels on the same rows, namely the rows with a non-missing value for every variable in the model. A missing outcome, covariate, visit or participant label was previously dropped from the design matrix but kept in the participant labels passed beside it. This raised an error in `did_mixed()` (`IndexError` for a missing outcome, covariate or visit label, `TypeError` for a missing participant label) and in `test_treatment_heterogeneity()` (a missing covariate), and was recorded in the `error` field in place of the estimate by `trend_interaction()`; `abundance_did()` instead fell back silently to a homoskedastic standard error with no bootstrap. This affects cell-type abundance difference-in-differences (DiD) estimates fitted with a covariate that contains missing values.

#### Compatibility
- `pseudobulk_within_arm()` and `module_score_within_arm_by_pool()` now return a missing p-value for a feature whose paired differences are all zero. The Wilcoxon signed-rank test raised an error on such input in older SciPy releases (giving NaN) and returns a p-value of 1 with a warning in recent ones, so both the feature's value and the false discovery rate (FDR) of the other features had depended on the SciPy version.

### Added

#### Estimation
- `within_arm_comparison()` now returns `resid_sd`, the residual standard deviation of the fitted model, as `did_table()` does. With two visits per participant, `sqrt(2) * resid_sd` estimates the standard deviation of the participant changes.

#### Inference
- `get_between_arm_aggregated_df()` returns the participant-level model frame fitted by `between_arm_comparison()`, and `between_arm_fit_beta()` returns the between-group coefficient from that frame. Exposing the two steps separately supports permutation tests, in which the group labels are permuted across participants and the coefficient refit to build a null distribution. They mirror the existing `get_within_arm_aggregated_df()` and `within_arm_fit_beta()`.
- A `cov_type` argument on `did_table()`, `did_table_parallel()`, `DiDConfig` and `run_gsea_did()` selects the covariance estimator for the analytical test: `"cluster"`, cluster-robust by participant (the default, unchanged), or `"nonrobust"`, the homoskedastic estimator. With two visits and participant fixed effects the homoskedastic estimator is the pooled two-sample t-test on the within-participant changes. The cluster-robust variance of the group-by-time coefficient is estimated separately within each arm and is therefore unstable when an arm holds very few participants; the homoskedastic estimator uses the participants of both arms and assumes equal variance of the changes across arms. The existing `cov_type_used` column records the estimator applied to each row.

#### Enrichment analysis
- `rank_genes_did()`, `rank_genes_within_arm()`, `rank_genes_cross_sectional()` and `run_gsea_prerank()` expose the two steps of the gene-set enrichment functions separately. Ranking fits one model per gene and is the costly step; a ranking is computed once, and `run_gsea_prerank()` then tests it against one gene-set collection per call, each with its own permutations and its own false discovery rate (FDR) control. `run_gsea_did()`, `run_gsea_within_arm()` and `run_gsea_cross_sectional()` run the two steps in sequence and return the same output as before.

#### Power and sample size
- `power_two_group()`, `sample_size_two_group()` and `sensitivity_two_group()` give the power, the required group sizes, and the minimum detectable difference for a comparison of two groups of unequal size, such as a cross-sectional comparison between groups or a difference-in-differences (DiD) analysis whose arms differ in size. `sample_size_two_group()` takes the planned allocation ratio and returns both group sizes. With equal groups the three functions reduce to `power_did()`, `sample_size_did()` and `sensitivity_analysis()`.
- A `reference` argument on `power_two_group()`, `sample_size_two_group()`, `sensitivity_two_group()`, `power_paired()`, `sample_size_paired()` and `sensitivity_paired()` selects the reference distribution: `"normal"` (the default, unchanged) or `"t"`, which uses the exact power of the two-sample or paired t-test from the noncentral t distribution. The normal approximation overstates power in small samples. For six paired participants and an effect of 1.2 standard deviations of the change, it gives a power of 0.836, against 0.655 from the t distribution.

#### Data handling and validation
- `is_ribosomal_gene()` identifies cytosolic ribosomal protein genes: the `RPS`, `RPL` and `RPLP` families together with `RPSA`, `FAU` and `UBA52`. The test matches the full gene symbol, so genes that only share a prefix are not matched, including the ribosomal protein S6 kinases (`RPS6KA1` and relatives), read-through transcripts, antisense RNAs and processed pseudogenes. This is the same test already applied by `flag_artifact_genes()`, now available to callers that need to identify these genes directly rather than matching the prefix themselves.
- `subset_arms()` restricts an AnnData object, or a table that holds the arm column, to the two arms named by a `TrialDesign`. Rows belonging to any other arm, and rows with a missing arm label, are removed, with a single warning that names the removed labels and the number of participants carrying them. Data that hold such rows but contain no row of one of the two named arms do not match the design and raise `ValueError`.

#### Plotting
- A `seed` argument on `plot_trial_interaction()`, `plot_parallel_trends()` and `plot_abundance_interaction()` (default 0) makes their error bars reproducible. These error bars are bootstrap 95% confidence intervals of the mean over participants, drawn by seaborn, and previously varied from one call to the next. They are now identical on every call with the same seed; `seed=None` resamples them on each call.

#### Documentation
- A [Study Designs](https://www.omar-lab.com/sctrial/study_designs.html) page in the documentation shows ten schematic animations of how the package handles each study design: the design definition, participant-level inference, two-arm and single-arm longitudinal designs, cross-sectional comparison, repeated visits, small cohorts, incomplete and unbalanced data, cell-type-specific analysis, and power. The README shows the same clips as one animation.

### Changed

#### Estimation
- `did_table()`, `did_table_parallel()` and `get_did_aggregated_df()` now carry a weight column of `adata.obs`, named by `weight_col`, through to the fit. Under aggregation the column must be constant within each aggregated unit; `"n_cells"` remains the cell count of the aggregate. Before this change a weight column other than `n_cells` raised `KeyError` in these functions.

#### Enrichment analysis
- The gene-set enrichment functions now return gene sets with an equal normalized enrichment score (NES) in a fixed order. Gene sets that contain the same ranked genes receive the same NES, and the underlying gseapy routine returned them in an order that varied between runs, so two runs on the same data could list such sets differently. Every enrichment function, `run_gsea_pseudobulk()` (enrichment on pseudobulk profiles, formed by summing counts across the cells of each participant at each visit) included, now passes through `run_gsea_prerank()`, which orders results by decreasing absolute NES and breaks ties by gene-set name. No enrichment values change.
- `run_gsea_did_multi()` now ranks the genes once and tests the same ranking against every gene-set collection; it previously recomputed the gene-level table for each collection. The ranking arguments (`layer`, `rank_by`, `cov_type` and the others of `run_gsea_did()`) are now named explicitly in its signature. Results are unchanged.

#### Multi-visit and mixed models
- `within_arm_comparison()`, `between_arm_comparison()`, `test_treatment_heterogeneity()`, `did_table_mixed()` and `trend_interaction()` now fit each feature on a frame holding only the columns of that feature's own model. Each feature previously received a frame holding the columns of all features, copied once per feature, so the time per feature grew with the number of features requested in a single call. Results are unchanged.

#### Data handling and validation
- `drop_artifact_genes()` now records, before removing the genes, each dropped class's share of every cell's total counts in `counts_layer` (`obs['pct_counts_ribo']`, `obs['pct_counts_hb']`, `obs['pct_counts_histone']`; new `counts_layer` argument, default `"counts"`). Once the genes are removed these fractions can no longer be computed, so quality-control plots of ribosomal or hemoglobin content had no valid input. Existing processed objects are unchanged; the columns are added the next time a dataset is processed.

### Fixed

#### Inference
- The wild cluster bootstrap is computed in closed form and no longer refits a regression for each replicate, which reduces its runtime. `cov_type` accepts `"cluster"` and `"nonrobust"`.

#### Multi-visit and mixed models
- `did_mixed()` now returns a missing estimate, with the reason recorded in the `error` field, when the data contain visit labels other than the two named in `visits`, instead of raising `IndexError`.
- `trend_interaction()` now returns its estimate columns even when no feature yields an estimate, so `test_parallel_trends()`, which reads those columns, no longer raises `KeyError` on such input. The `test_parallel_trends()` docstring was corrected to name the label the function actually returns, `"Potential violation"`.

#### Data handling and validation
- Gene filtering now recognizes replication-dependent histone genes under their current HGNC symbols as well as the pre-2020 `HIST*` symbols. `drop_artifact_genes()` previously matched only the `HIST*` symbols, so data aligned to GENCODE v35 or later references (for example the Cell Ranger 2024-A reference, which uses symbols such as H4C3, H1-2, and H2AW) retained these highly expressed housekeeping genes, which can dominate variable-gene selection and enrichment rankings. Both conventions are now recognized. Replication-independent variant histones (H2AZ1, H2AX, H3-3A, H1-0) are kept. Bundled datasets use the older symbols and are unchanged.
- `TrialDesign.arm_bin()` now raises on rows whose arm is neither `arm_treated` nor `arm_control`, instead of coding them as control (0). `TrialDesign.validate()` now reports arms other than the two named ones, and missing arm labels, as excluded from two-arm analyses.
- `check_covariate_balance()` and `module_score_did_by_pool()` already compared only the two named arms. They now report excluded rows and raise when the design does not match the arm labels in the data, consistent with the other two-arm functions. Their reported values are unchanged.
- `pseudobulk_export()` now gives the exported pseudobulk table a string index, so anndata no longer emits a warning that it is converting the index.

#### Compatibility
- `subset_primary()` works under pandas 3, where an `obs` column's backing array can be a read-only view. The function updated its mask in place, so any analysis of a design with a crossover column raised `ValueError: output array is read-only`; it now operates on a copy. Separately, the error message for a missing counts layer in benchmark input preparation now lists the available layer names correctly under anndata 0.13, which adds a `None` key for `X`.

#### Documentation
- The worked example in the `power_did()` docstring now shows the value the function returns (35.3%).
- `bootstrap_effect_size_ci()` is now listed in `sctrial.stats.effect_size.__all__` and appears in the API reference; it was already importable from `sctrial` and `sctrial.stats` but was missing from the documentation. `extract_gene_vector()` and `sctrial.benchmark.runners.finalize_results()` were likewise added to the API reference.
- Docstrings and documentation pages were corrected to match the code. The complete set of returned columns is now documented for `trend_interaction()`, `event_study_did()`, `polynomial_trend()`, `test_parallel_trends()`, `loo_cv_did()`, `kfold_cv_did()`, `cv_summary()`, `influence_diagnostics()`, `abundance_did()`, `pseudobulk_did()`, `test_treatment_heterogeneity()`, `did_fit()` and `within_arm_comparison()`, and the `within_arm_comparison()` interval is documented as using the normal quantile with the cluster-robust covariance rather than a t quantile. The documented behavior of `ensure_fdr()`, `add_log1p_cpm_layer()`, `load_sade_feldman()` (which downloads only with `allow_download=True`) and the `min_cells_per_participant_visit` argument of `load_vaccine_gse171964()` was corrected, and docstring lists now render as lists in the API reference.

### Simulation benchmark

These entries concern `sctrial.benchmark` and the benchmark scripts of the repository.

- The real-data benchmark path now passes a cohort to the method runners under the labels those runners expect. The runners read the arm labels `Treated` and `Control` and the visit labels `Pre` and `Post`, but `prepare_inputs_from_adata()` kept the cohort's own labels; on a cohort with other labels, such as the melanoma cohort's `Responder` and `Non-responder`, no row matched either arm and `run_permutation_test()` and `run_subsampling()` returned a missing result for every gene and method. `canonical_cohort()` now maps the columns, the two visits, and the two arms from labels the caller supplies; `run_permutation_test()` and `run_subsampling()` take the new arguments `visits`, `arm_treated`, and `arm_control` (and `run_subsampling()` also takes `design_type`), and `prepare_inputs_from_adata()` raises on any other label.
- The counts layer of a real cohort must now be named explicitly. `prepare_inputs_from_adata()` previously read `adata.X` when the requested layer was absent, which could hand a normalized matrix to the count models of a cohort that holds no raw counts. A missing layer now raises an error (with `counts_layer=None` denoting `adata.X`), and `require_raw_counts()` stops `dreamlet`, `limma_voom`, `edger_qlf`, and `nebula` on a matrix with non-integer values. The real-data benchmark phase therefore runs only the participant-level log(1 + CPM) methods on the melanoma cohort, which has no raw counts.
- The single-arm real-data permutation of the benchmark now exchanges the two visit labels for whole participants. It previously shuffled the labels cell by cell within a participant, mixing the cells of the two samples. Separately, the fixed-effects rung of the benchmark ablation no longer carries the arm as a column beside the participant indicators.
- In the benchmark, a gene returned without a p-value is now recorded as not tested. Method runners previously marked every returned gene as converged with no failure even when the p-value was missing, as in a t-test on changes with no variance or a missing value in an R method's result table, so `compute_failure_rates()` and the convergence rate counted such genes as successes. `finalize_results()` now sets `converged` to false, assigns a failure mode and leaves the interval missing, and every method row and `run_ablation()` row passes through it; this affects the benchmark's reported convergence and failure rates.
- `scripts/run_benchmark.py --phase all` now runs the four active benchmark phases; it previously halted at the retired `validate` phase and ran none of them.

### Internal

- Resolved two static type-checking errors, in the benchmark gate verdict and in the benchmark orchestrator command-line entry point. Behavior is unchanged.
- The project description on PyPI is built from the README with its relative image paths and file links rewritten to the files of the release tag, so the logo, the overview figure and the study-design animation are displayed there. The build requires `hatch-fancy-pypi-readme`, which is installed automatically as a build dependency.
- The source distribution holds the package, the tests, the README, the license and `pyproject.toml`; a file named `README.md` in another folder is no longer included.

## [1.0.0] - 2026-10-05

Version bump to stable release. No functional changes from 0.3.4.

### Added
- `simulate_trial_v2()` and `TranscriptomeSimConfig`: transcriptome-scale hierarchical gamma-Poisson simulator calibrated from real single-cell data, superseding the original `SimulationConfig` API
- TNBC (triple-negative breast cancer) dataset support and accompanying tutorial
- Second-order DiD analysis demonstrated in tutorials
- Signal-fraction sensitivity benchmark (gene panel × signal percentage grid)

### Fixed
- Double log bug in benchmark orchestrator and permutation modules (`orchestrator.py`, `permutation.py`)
- GSEA ranking coerced to finite, unique values before `gseapy.prerank`: infinite betas from degenerate OLS and duplicate gene symbols caused "float division by zero" across all five AML gene-set libraries
- Cluster-robust fit falls back to non-robust fit when within-arm degrees of freedom are exhausted, fixing silent failure of AML GSEA
- NEBULA runner: filter zero-count cells before computing log library-size offset; set `ncore=1` to prevent PSOCK port conflicts; robust R error capture and crash detection
- COVID loader: CITE-seq antibody-capture features (~49% of aggregate counts) were sitting in the gene matrix, confounding library-size normalisation; split to `obsm["protein"]`
- Melanoma: sample filter silently dropped 19% of QC-passing cells (3,108 of 16,291) and 7 of 32 patients; per-cell annotations now sourced from the published labels; RECIST arm normalised to leading code
- Vaccine loader: 45 CITE-seq antibody-capture features moved from the gene matrix to `obsm["protein"]`, so they no longer enter the library-size denominator
- AML loader: hemoglobin genes kept in bone marrow, where they define the erythroid populations; the deposited MUTZ-3 and OCI-AML3 cell lines excluded from the healthy-donor reference
- Benchmark: signal-count truncation, frozen design leaking into scenarios, split jobs overwriting combined results

### Improved
- QC: ambient and housekeeping gene classes excluded upstream in all loaders, so GSEA rankings inherit a clean gene set
- Vaccine loader (GSE171964): raw UMI counts now normalized with 10x QC, matching other loaders
- Benchmark simulator calibrated from real TNBC data: within-cell-type coherent, mean-dependent dispersion, arm-stratified bootstrap gate
- 27 mypy type errors fixed across benchmark and dataset modules
- Tutorials re-executed against `v1.0.0`

## [0.3.4] - 2026-09-22

### Added
- `simulate_trial_v2()` and `TranscriptomeSimConfig`: transcriptome-scale hierarchical gamma-Poisson simulator calibrated from real single-cell data, superseding the original `SimulationConfig` API
- TNBC (triple-negative breast cancer) dataset support and accompanying tutorial
- Second-order DiD analysis demonstrated in tutorials
- Signal-fraction sensitivity benchmark (gene panel × signal percentage grid)


### Fixed
- Double log bug in benchmark orchestrator and permutation modules (`orchestrator.py`, `permutation.py`)
- GSEA ranking coerced to finite, unique values before `gseapy.prerank`: infinite betas from degenerate OLS and duplicate gene symbols caused "float division by zero" across all five AML gene-set libraries
- Cluster-robust fit falls back to non-robust fit when within-arm degrees of freedom are exhausted, fixing silent failure of AML GSEA
- NEBULA runner: filter zero-count cells before computing log library-size offset; set `ncore=1` to prevent PSOCK port conflicts; robust R error capture and crash detection
- COVID loader: CITE-seq antibody-capture features (~49% of aggregate counts) were sitting in the gene matrix, confounding library-size normalisation; split to `obsm["protein"]`
- Melanoma: sample filter silently dropped 19% of QC-passing cells (3,108 of 16,291) and 7 of 32 patients; per-cell annotations now sourced from the published labels; RECIST arm normalised to leading code
- Vaccine loader: 45 CITE-seq antibody-capture features moved from the gene matrix to `obsm["protein"]`, so they no longer enter the library-size denominator
- AML loader: hemoglobin genes kept in bone marrow, where they define the erythroid populations; the deposited MUTZ-3 and OCI-AML3 cell lines excluded from the healthy-donor reference
- Benchmark: signal-count truncation, frozen design leaking into scenarios, split jobs overwriting combined results

### Improved
- QC: ambient and housekeeping gene classes excluded upstream in all loaders, so GSEA rankings inherit a clean gene set
- Vaccine loader (GSE171964): raw UMI counts now normalized with 10x QC, matching other loaders
- Benchmark simulator calibrated from real TNBC data: within-cell-type coherent, mean-dependent dispersion, arm-stratified bootstrap gate
- 27 mypy type errors fixed across benchmark and dataset modules
- Tutorials re-executed against `v0.3.4`

## [0.3.3] - 2026-04-05

### Added
- `sctrial.benchmark` subpackage: hierarchical gamma-Poisson simulator and benchmark orchestrator for controlled method comparisons. (Superseded in a later release by the transcriptome-scale `TranscriptomeSimConfig` / `simulate_trial_v2`; the original `SimulationConfig` API described here no longer exists.)
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
- Tutorial notebooks revised for clarity and consistent use of sctrial API

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
- **Statistical**: Fixed WLS weighting in `did_fit()`: it now uses `n_cells` (correct inverse-variance weights) instead of `sqrt(n_cells)` for pre-aggregated participant-level means.
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
