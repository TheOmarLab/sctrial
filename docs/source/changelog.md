# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-11-24

### Added
- `is_ribosomal_gene()`: the anchored ribosomal-protein test already used by `flag_artifact_genes()`, exposed so callers stop matching the `RPS`/`RPL` prefix, which also catches the RPS6K kinases, read-through transcripts and ribosomal pseudogenes.
- `get_between_arm_aggregated_df()` and `between_arm_fit_beta()`: the frame and estimator of `between_arm_comparison()`, for permutation tests, mirroring `get_within_arm_aggregated_df()` and `within_arm_fit_beta()`.
- `power_two_group()`, `sample_size_two_group()` and `sensitivity_two_group()`: power, required group sizes and minimum detectable difference for a comparison of two groups of unequal size, such as a cross-sectional comparison or a DiD whose arms differ in size. `sample_size_two_group()` takes the allocation ratio of the planned study and returns both group sizes. With equal groups the three functions reproduce `power_did()`, `sample_size_did()` and `sensitivity_analysis()`.
- `subset_arms()`: restricts an AnnData, or a table that holds the arm column, to the two arms named by a `TrialDesign`. Rows of any other arm and rows with a missing arm label are removed with one warning that names the labels and counts their participants. Data that hold such rows and no row of one named arm raise `ValueError`.
- `cov_type` argument of `did_table()`, `did_table_parallel()`, `DiDConfig` and `run_gsea_did()`: the covariance of the analytical test, `"cluster"` (the default, as before) or `"nonrobust"`. `did_fit()` already took it; the tables and the GSEA ranking could not pass it on. With two visits and participant fixed effects the homoskedastic estimate is the pooled two-sample t-test on the within-participant changes. The cluster-robust variance of the group-by-time coefficient is estimated within each arm, so it is unstable when an arm holds very few participants; the pooled estimate uses the participants of both arms and assumes that the changes vary equally in the two arms. The existing `cov_type_used` column records the estimate each row reports.
- `reference` argument of `power_two_group()`, `sample_size_two_group()`, `sensitivity_two_group()`, `power_paired()`, `sample_size_paired()` and `sensitivity_paired()`: `"normal"` (the default, unchanged) or `"t"`, the exact power of the two-sample or paired t-test from the noncentral t distribution. The normal approximation overstates power in small samples (six paired participants and an effect of 1.2 SD of the change: 0.836 with the normal formula, 0.655 with the t distribution).
- `within_arm_comparison()` returns `resid_sd`, the residual standard deviation of the fitted model, as `did_table()` does. With two visits per participant, `sqrt(2) * resid_sd` is the standard deviation of the participant changes.
- `rank_genes_did()`, `rank_genes_within_arm()`, `rank_genes_cross_sectional()` and `run_gsea_prerank()`: the two steps of the GSEA functions as separate calls. Ranking fits one model per gene and is the expensive step; a ranking is computed once and `run_gsea_prerank()` tests it against one gene-set collection per call, each with its own permutations and FDR. `run_gsea_did()`, `run_gsea_within_arm()` and `run_gsea_cross_sectional()` are these two calls in sequence and return what they returned before.
- `seed` argument of `plot_trial_interaction()`, `plot_parallel_trends()` and `plot_abundance_interaction()` (default 0). The error bars of these plots are bootstrap confidence intervals of the mean over participants, drawn by seaborn, and they changed from call to call. They are now the same on every call; `seed=None` draws new ones each time.

### Changed
- `drop_artifact_genes()` records, before removing the genes, each dropped class's share of every cell's total in `counts_layer` (`obs['pct_counts_ribo']`, `obs['pct_counts_hb']`, `obs['pct_counts_histone']`; new `counts_layer` argument, default `"counts"`). Once the genes are gone these fractions cannot be recovered, so quality-control plots of ribosomal content previously had nothing valid to show. Existing processed objects are unchanged; the columns appear when a dataset is next processed.
- `did_fit()` now fits ordinary least squares by default. It previously switched silently to weighted least squares whenever the input carried an `n_cells` column, so every participant-level DiD from `did_table(aggregate="participant_visit")` and `did_table_parallel()` was weighted by cell count. Cell-count weights assume that all variance in a participant-visit mean is cell-sampling noise; in clinical cohorts between-participant variation dominates and cell counts span two orders of magnitude, so a few participants determined each estimate. With two visits the default estimate equals the difference in mean within-participant change between arms, which is the estimator the simulation benchmark validates. Weighted fitting is still available through the new `weight_col` argument of `did_fit()`, `did_table()`, `did_table_parallel()` and `DiDConfig`. Participant-level DiD estimates change; cell-level estimates do not.
- `did_table()`, `did_table_parallel()` and `get_did_aggregated_df()` carry a weight column of `adata.obs` named by `weight_col` into the fit. Under aggregation the column must be constant within each aggregated unit; `"n_cells"` remains the cell count of the aggregate. A weight column other than `n_cells` raised `KeyError` in the table functions.
- The GSEA functions return gene sets with the same NES in name order. Gene sets that hold the same ranked genes have the same NES, and gseapy returns them in an order that changes from run to run, so two runs on the same data could list such sets in a different order. Every GSEA function, `run_gsea_pseudobulk()` included, now goes through `run_gsea_prerank()`, which keeps the order of gseapy (decreasing absolute NES) and orders equal NES by name. No value changes.
- `run_gsea_did_multi()` ranks the genes once and tests the same ranking against every collection. It computed the gene-level table again for each collection. The arguments of the ranking (`layer`, `rank_by`, `cov_type` and the others of `run_gsea_did()`) are now named in its signature. Results are unchanged.
- `within_arm_comparison()`, `between_arm_comparison()`, `test_treatment_heterogeneity()`, `did_table_mixed()` and `trend_interaction()` fit each feature on a frame that holds the columns of its own model. The frame of every feature held the columns of all features and was copied once per feature, so the time per feature grew with the number of features in the call. Results are unchanged.

### Fixed
- Replication-dependent histone genes are recognised under their current HGNC symbols (H4C3, H1-2, H2AW, ...) as well as the pre-2020 HIST* symbols. Data aligned to references from GENCODE v35 on (for example Cell Ranger 2024-A) kept their histone genes through `drop_artifact_genes()`. Variant histones (H2AZ1, H2AX, H3-3A, H1-0) are still kept. No bundled dataset changes: all use the older symbols.
- The wild cluster bootstrap now refits the model under the null. `wild_cluster_bootstrap_t()` imposed the null by subtracting the estimated coefficient from the unrestricted fit, which keeps the other coefficients at their unrestricted values. In a two-arm difference-in-differences that centres every participant's change on the control-arm mean, so the p-value depended on which arm was coded as treated and the test was conservative (3.5% rejections at a nominal 5% over all 924 arm relabelings of a 12-participant cohort, 4.9% after the fix). The function now drops the tested term, refits, and resamples the restricted residuals. In single-arm (within-participant) models the two constructions coincide, so those p-values change only through the resampling changes below.
- When the clusters admit no more than `n_boot` sign assignments (nine or fewer clusters at the default 999), all assignments are enumerated instead of sampled. The test is then exact and no longer depends on `seed`; its p-value is a multiple of `1 / 2**G`.
- Bootstrap confidence intervals are obtained by inverting the bootstrap test and no longer by applying quantiles of the null bootstrap distribution to the estimate. The interval contains every value the test does not reject, so it contains zero whenever the bootstrap p-value is at least `1 - ci_level`. Its bounds are found from the values at which a draw's statistic meets the observed one. In small or unbalanced designs the values that are not rejected can form more than one interval; the reported interval then spans them and can contain zero although the p-value is below `1 - ci_level`. Under enumeration the p-value cannot fall below `2**(1 - G)`, so with five or fewer clusters (enumerated whenever `n_boot` is at least 32) the test cannot reject at the 5% level and the interval is unbounded (`-inf`, `inf`); `did_table()`, `did_table_parallel()`, `pseudobulk_did()`, `within_arm_comparison()` and `abundance_did()` issue one warning per table when that happens.
- The bootstrap standard error (`se_DiD_boot`, `se_time_boot`) is the exact standard deviation of the unrestricted wild cluster bootstrap, which equals the cluster-robust standard error without small-sample correction. It was the standard deviation of null-imposed draws, which grows with the estimated effect. `BootstrapResult.boot_distribution` holds the unrestricted estimates.
- The observed statistic is studentized with the variance estimator requested through `cov_type`, as the bootstrap statistics are. `abundance_did()` on paired differences compared a homoskedastic observed statistic with cluster-robust bootstrap statistics.
- The bootstrap is computed in closed form and no longer refits a regression per draw. `cov_type` accepts `"cluster"` and `"nonrobust"`.
- Resampling tests count a statistic that equals the observed one as a tie whatever the rounding. The wild cluster bootstrap computes the observed and the bootstrap statistics along different arithmetic paths, so an assignment that reproduces the observed statistic, such as flipping a participant whose outcome did not change, could be left out of the count, which lowered the exact p-value. A bootstrap statistic within `1e-9` of the observed one is now a tie. `permutation_pvalue()`, `permutation_pvalue_paired()` and the permutation test of `module_score_did_by_pool()` count a difference of means within rounding error of the observed one. The two-sample tests centre the pooled values first and the paired test works on the differences, so the result does not depend on the origin or the units of the measurements; a permutation that reproduced the observed split in another order was not always counted.
- A bootstrap draw that the model fits exactly counts as extreme. Such a draw has no residual variance and an unbounded statistic; rounding left its variance just above or just below zero, and the sign decided whether the draw entered the count, so the p-value of the same data could change with the coding of the arms or the units of the outcome.
- The wild cluster bootstrap refit OLS models with WLS. It detected weighting from the presence of `model.weights`, which statsmodels also sets, to ones, on OLS models; it now scales the outcome and the regressors by the square root of the weights, which leaves an OLS model unchanged.
- Two type errors reported by mypy in the benchmark gate verdict and the orchestrator command-line entry point.
- Participant fixed effects are built from the participants in the model. anndata keeps every category of an `obs` column, so a participant left out of a paired analysis (one visit only, or another arm) still contributed an empty indicator column to `C(participant)`. The empty columns counted as parameters in the small-sample correction of the cluster-robust covariance, which inflated the standard error when a few participants were left out and made it undefined once the parameters reached the number of observations; `did_fit()` and `within_arm_comparison()` then fell back to the homoskedastic standard error and studentized the wild cluster bootstrap with it. The model frames of `did_fit()`, `did_table()`, `within_arm_comparison()`, `within_arm_fit_beta()`, `between_arm_comparison()`, `abundance_did()` and the time-series, heterogeneity and mixed-effects models now drop unused categories. Results change only for inputs that carry participants outside the model: the analytical standard error and p-value, and, in two-arm designs with unequal arms, the bootstrap p-value and interval. Bootstrap results of single-arm designs and of two-arm designs with equal arms do not change, because there the two studentizations differ by a constant factor.
- Two-arm analyses use the two arms named by the design. The treatment indicator was `arm == arm_treated` and `arm_control` was never read. A participant of a third arm was therefore analyzed as control by `did_table()`, `did_table_parallel()`, `did_table_by_celltype()`, `get_did_aggregated_df()`, `did_table_bayes()`, `between_arm_comparison()`, `get_between_arm_aggregated_df()`, `abundance_did()`, `pseudobulk_did()`, `test_treatment_heterogeneity()`, `did_table_mixed()`, `compare_fixed_vs_mixed()`, `trend_interaction()`, `test_parallel_trends()`, `polynomial_trend()`, `event_study_did()`, `loo_cv_did()` and `kfold_cv_did()`. A participant with a missing arm label was analyzed as control by `did_table(aggregate="cell")` and `pseudobulk_did()` and dropped without notice by the others. A design whose control label did not occur in the data ran with every other participant as control. These functions now build their model frame through `subset_arms()`: rows outside the two named arms are excluded with one warning per call. Data that hold such rows and no row of one named arm raise `ValueError`, and so does a design with `arm_treated == arm_control` (for one arm use `within_arm_comparison()` or a design with `arm_col=None`). Results change only for data that hold rows outside the two named arms.
- `TrialDesign.arm_bin()` raises on rows whose arm is neither `arm_treated` nor `arm_control` instead of coding them 0. `TrialDesign.validate()` reports other arms, and now also missing arm labels, as excluded from two-arm analyses.
- `did_mixed()` takes `arm_control`. When it is given, the table is restricted to the two arms. Without it every row that is not `arm_treated` forms the reference group, as before, and a warning is raised when that group holds more than one label or missing labels.
- `check_covariate_balance()` and `module_score_did_by_pool()` already compared the two named arms. They now report excluded rows, and raise when the design does not match the arm labels of the data, in the same way as the other two-arm functions.
- `loo_cv_did()`, `kfold_cv_did()`, `did_table_by_celltype()` and `event_study_did()` catch `ValueError` inside their loops. They restrict to the two named arms before the loop, so a design that does not match the labels of the data raises and is not recorded as one failed fit per iteration. `kfold_cv_did()` assigns folds within the two named arms only; a third arm was a stratum of its own.
- `did_fit()`, `between_arm_comparison()` and `did_table_bayes()` return a missing estimate when one arm has no participant in the model, for example for a cell type present in one arm only. With only the treated arm in the model the interaction equals the time term: `did_fit()` returned half of the time effect as the DiD coefficient, with a finite p-value, and the posterior of `did_table_bayes()` split the time effect between the two coefficients. With only the control arm `did_fit()` returned 0 and `did_table_bayes()` the mean of the prior. `between_arm_comparison()` returned a coefficient of 0.
- Cluster-robust standard errors no longer grow with redundant regressors. statsmodels counts the columns of the design, not its rank, in the small-sample factor, so a regressor that the participant effects already determine enlarged the standard error without changing the estimate. `did_fit()`, `within_arm_comparison()` and `abundance_did()` now fit a covariate that is constant within participant on a full-rank design, and `within_arm_comparison()` returns a missing estimate when a covariate duplicates the visit. `test_treatment_heterogeneity()` no longer enters the arm, the biomarker group and their product as separate terms beside the participant effects; its `covariates` argument raised an error, because the covariates never reached the model frame, and now enters the fit.
- `did_fit()` raises on a missing weight, as documented. A row with a missing value in `weight_col` was dropped, and its participant with it.
- `did_fit()` fits the participants observed at both visits for the outcome and returns a missing estimate when the interaction is not identified. A participant with a missing value at one visit stayed in the model and was counted in `n_units` and among the clusters of the robust variance. When a whole arm lacked one visit, the visit and the interaction were the same regressor: the fit split the other arm's change between them and reported it with a standard error near zero. The same holds for a covariate that duplicates the interaction.
- The real-data benchmark path hands a cohort to the method runners in the vocabulary they read. `prepare_inputs_from_adata()` renamed the columns of a cohort and carried its label values, while the runners read `Treated`/`Control` and `Pre`/`Post`: on a cohort with other labels (the melanoma cohort has `Responder`/`Non-responder`) no row matched either arm and `run_permutation_test()` and `run_subsampling()` returned a missing result for every gene and method. `canonical_cohort()` now maps the columns, the two visits and the two arms from labels the caller states, through `subset_arms()`; the two functions take them as the new keyword arguments `visits`, `arm_treated` and `arm_control`, map the cohort once before their loops, and `prepare_inputs_from_adata()` raises on any other label. `run_subsampling()` takes `design_type`.
- The counts of a real cohort are named explicitly. `prepare_inputs_from_adata()` read `adata.X` when the layer it was given was absent, which hands a normalised matrix to the count models of a cohort that has no raw counts. A missing layer now raises (`counts_layer=None` names `adata.X`), and `require_raw_counts()` stops `dreamlet`, `limma_voom`, `edger_qlf` and `nebula` on a matrix with non-integer values. `scripts/run_benchmark.py --phase realdata` names the labels and the TPM layer of the melanoma cohort and runs the two methods that take participant-level log(1 + CPM), because that cohort holds no raw counts.
- `test_treatment_heterogeneity()` uses the participants observed at both visits and treats the biomarker as a participant attribute. A participant without a biomarker value was assigned to the low group, because a comparison with a missing value is false; such participants are now excluded and reported. The median split was taken over cells, so participants with more cells weighed more; it is now taken over the participants in the model. Participants with one visit were kept in the model and counted in `n_units`. When an arm by biomarker group has fewer than two participants, the function warns and returns NaN for the coefficient and the p-value. It previously fitted the model in that case, on a rank-deficient design when a group was empty. The group sizes are counted for each feature, among the participants with a value of that feature at both visits.
- GSEA ranks only genes that have an estimate. `run_gsea_did()`, `run_gsea_within_arm()`, `run_gsea_cross_sectional()` and the wrappers built on them gave a gene whose model could not be fitted a rank of zero. Such genes sat in the middle of the ranked list, counted as genes outside every set and as tied ranks, and changed the enrichment scores of every gene set. A gene now needs its estimate, and its standard error or p-value where the ranking uses one.
- A GSEA ranking is the same in every run. Genes with the same data across participants have the same statistic, and statistics that are equal in exact arithmetic can differ in the last bits between runs. The enrichment score of a gene set depends on where its genes fall among such ties, which were left in an arbitrary order. Ranks are rounded to ten significant digits and equal ranks are ordered by gene name.
- `run_gsea_did_by_celltype()` leaves out, with one warning, a cell type in which the DiD cannot be estimated: one found in a single arm, at a single visit, or with fewer than `min_units` participants. Such a cell type stopped the call for every cell type, and so did one found only in an arm outside the design, which the default list of cell types included. Eligibility is counted after crossover participants are excluded.
- `sample_size_paired()`, `sample_size_did()` and `sample_size_two_group()` return the smallest size at which the matching power function reaches the target. For the paired and the two-group function, a formula result rounded up was one too large when the real-valued solution was an integer up to rounding, so the size returned for a detectable effect could exceed the size that effect was computed from. For two groups the smaller group is its share of the larger one rounded up to a whole participant.
- The two-sided power of the t-based functions is finite for every design. With recent SciPy releases the lower tail of the noncentral t returns NaN at some degrees of freedom and noncentralities, which stopped `sensitivity_paired()` and `sensitivity_two_group()` for two participants and gave a NaN power for some strongly powered designs.
- `bootstrap_effect_size_ci()` warns when bootstrap samples have no within-group variation and are left out of the interval. `module_score_within_arm_by_pool()`, `module_score_did_by_pool()` and `pseudobulk_within_arm()` treat paired changes as absent only when they are exactly zero, not when they are below `1e-8`.
- The single-arm real-data permutation of the benchmark exchanges the two visit labels of whole participants. It shuffled the labels cell by cell within a participant, which mixed the cells of the two samples. The fixed-effect rung of the benchmark ablation no longer carries the arm as a column beside the participant indicators.
- `pseudobulk_within_arm()` and `module_score_within_arm_by_pool()` return NaN for a feature whose paired differences are all zero. The Wilcoxon test raised an error for such input in older SciPy releases, which gave NaN, and returns a p-value of 1 with a `RuntimeWarning` in recent ones, so the value, and the FDR of the other features, depended on the SciPy version.
- `pseudobulk_export()` gives the pseudobulk table a string index, so anndata no longer warns that it converts the index.
- The example output in the `power_did()` docstring is the value the function returns (35.3%).
- Unused levels of Categorical columns no longer leak into results. `module_score_did_by_pool(allow_unpaired=True)` emitted a NaN row for an arm missing a visit instead of skipping it; `diagnose_trial_data()` counted cell types that were absent; the benchmark's real-data arm permutation could replace real arm labels with missing values.
- A model is fitted on the rows that hold every variable of it. The formula interface drops a row with a missing value from the model matrices, while the participant labels passed beside the formula kept it. `did_mixed()` raised `IndexError` on a missing outcome, covariate or visit label and `TypeError` on a missing participant label; `test_treatment_heterogeneity()` raised on a missing covariate; `trend_interaction()` returned an error in place of the estimate. `abundance_did()` caught the error for a missing covariate and fitted change scores with a homoskedastic standard error and no bootstrap, without saying so.
- Each feature is fitted on the participants that have a value for it at every visit of the design, as `did_fit()` does. In `did_table_mixed()`, `within_arm_comparison()`, `trend_interaction()`, `did_table_bayes()` and, for covariates, `abundance_did()`, a participant with a missing value kept its other visits in the model: it was counted in `n_units` and as a cluster, and its values entered the scaling of the outcome. `n_units` of `did_table_mixed()` and `trend_interaction()` is the count for the feature; it was one count for the table. `within_arm_fit_beta()` uses the same participants as the table, and `between_arm_comparison()` scales the outcome on the rows of its model, not on rows that a missing covariate keeps out of the fit.
- `between_arm_comparison(method="wilcoxon")` leaves out a participant without the feature. One missing value made every statistic of the feature missing while `n_units` counted the participant. `compare_gene_in_celltype()` drops the cells without a value, which gave a missing p-value beside finite means. `loo_cv_did()` no longer returns a row for a missing participant label.
- `did_mixed()` returns a missing estimate, with the reason under `error`, when the table holds visit labels other than `visits`. It raised `IndexError`.
- Redundant design columns are removed before the requested covariance is computed. The small-sample factor of the cluster-robust covariance divides by the number of rows minus the number of design columns, so a design with as many columns as rows raised `ZeroDivisionError` although it had residual degrees of freedom once the redundant columns were removed (four participants and a site covariate). A model without residual degrees of freedom raised the same error; it now returns the estimate with a standard error that is not finite and a missing p-value, as `cov_type="nonrobust"` does. `abundance_did()` with two participants no longer raises.
- A bootstrap draw with a small positive variance is no longer taken for a draw without variance. A variance below `1e-12` of the largest among the draws counted as zero, so a draw that the model fits almost exactly was counted as extreme whatever its statistic. Such a variance is now computed from the residuals of the draw and is zero only within the rounding error of the outcome. The same criterion applies to the observed statistic: when the model fits the observed outcome exactly the bootstrap returns NaN, where rounding left a positive variance and a p-value.
- The wild cluster bootstrap does not depend on the origin of the outcome. Far from zero (from about `1e7` times the spread of the outcome), draws whose statistic equals the observed one in exact arithmetic differed from it by more than the tie margin and were not counted. The outcome is measured from its mean when the other regressors hold a constant.
- `trend_interaction()` returns its estimate columns when no feature has an estimate, so `test_parallel_trends()` no longer raises `KeyError`. The docstring of `test_parallel_trends()` names the label the function returns, `"Potential violation"`.
- A least-squares model that fits its outcome exactly reports the estimate and no test. Every residual is then zero in exact arithmetic, and the standard error and the p-value were computed from rounding error: a within-arm change that no participant shows could return `p = 0.008`, and an arm difference without spread within the arms `p = 4e-216`. `did_fit()`, `within_arm_comparison()`, `between_arm_comparison()`, `test_treatment_heterogeneity()`, `trend_interaction()`, `abundance_did()` and the unpaired path of `module_score_did_by_pool()` return a missing standard error and p-value for such a feature. A fit is exact when its residuals, computed from the outcome measured from its mean, are within `8 * n * eps` times the largest absolute outcome.
- `did_table_mixed()` reads `covariates` from `adata.obs`, as the other tables do. It looked for them among the feature columns, so a covariate of `obs` was left out of the model without notice. A numeric covariate is aggregated with the features, and any other must be the same for every cell of a participant-visit. A covariate that is absent from `obs`, or that is also a feature, raises. `did_mixed()` raises on a covariate that is not a column of its table; it skipped it.
- `agg="pct_pos"` works in `did_table()`, `did_table_bayes()`, `within_arm_comparison()`, `between_arm_comparison()` and their frame builders, where it raised `AttributeError`. `test_treatment_heterogeneity()` applies `agg` to the features and to numeric covariates; it took the mean whatever the value. The percentage is taken over the cells that have a value, a numeric covariate keeps its mean under `pct_pos`, and an unknown `agg` raises `ValueError`.
- A coefficient that the data do not identify has no estimate. When every participant of one arm lacked a visit for a feature, `trend_interaction()` fitted that feature on the other arm alone: the arm-by-time regressor then equals the time regressor or is zero, and the function returned a share of the time trend as the interaction, with a standard error and a p-value. The same happened for more polynomial terms than the visits allow (a cubic model on three visits), and in `between_arm_comparison()` and `abundance_did()` for a covariate that holds the arm, such as a site that enrolled one arm or a dose that is zero in the control arm. These rows now hold missing values, and a higher-order term that the other regressors determine is left out of the `trend_interaction()` row. `did_fit()` and `abundance_did()` leave `beta_time` and `p_time` missing when a covariate changes by the same amount for every participant between the visits, which makes it the visit regressor; the interaction and its test are unchanged. `polynomial_trend()` raises when a time term or an arm-by-time term is not identified, for example with its default quadratic on two visits; it returned coefficients, p-values and a curve between the visits. `trend_interaction()` raised `ZeroDivisionError` on a design with as many parameters as rows and now returns the estimate without a test. The pseudobulk OLS comparator of `run_method_comparison()` goes through the same check.
- A result that could not be computed is not reported as a pass. `test_parallel_trends()` labels a feature without a test `"Not estimable"`; it was labelled `"OK"`. `influence_diagnostics()` leaves `is_influential` missing when the influence is missing (a fit without an estimate, or a fit on everyone without a standard error), and `compare_fixed_vs_mixed()` leaves `agreement` missing when either model gave no estimate or no test. Both columns were filled by comparing a missing value with a threshold, which is false, and are now nullable booleans: `.astype(bool)` and `np.where()` raise on a missing value, while `.sum()`, `.mean()` and boolean indexing skip it. `summarize_did_results()` counts the features that have a p-value as tested, reports the others on a line of their own, and lists only tested features. `module_score_did_by_pool(allow_unpaired=True)` returns no within-arm p-value for an arm with as many rows as parameters; it returned 1.0.
- A benchmark gene without a p-value is recorded as not tested. A method runner marked every gene it returned as converged and without failure, also when the p-value was missing (a t-test on changes without variance, a missing value in the result table of an R method), and the model rungs of the ablation returned no flags at all, so `compute_failure_rates()` counted such a gene as a success and the convergence rate counted it as converged. `finalize_results()` in `sctrial.benchmark.runners` sets `converged` to false, names a failure mode and leaves the interval missing; the method dispatcher and `run_ablation()` pass every row through it. The model rungs of the ablation return no test for a gene without variation, where the p-value was computed from rounding error.
- `bootstrap_effect_size_ci()` is listed in `sctrial.stats.effect_size.__all__`. It was exported from `sctrial` and `sctrial.stats` and missing from the API reference. `extract_gene_vector()` and `sctrial.benchmark.runners.finalize_results()` were missing from the API reference too.
- `scripts/run_benchmark.py --phase all` runs the four benchmark phases. It stopped at the retired `validate` phase and ran nothing.
- Docstrings and documentation pages corrected where they did not match the code. Returned columns are listed in full for `trend_interaction()`, `event_study_did()`, `polynomial_trend()`, `test_parallel_trends()`, `loo_cv_did()`, `kfold_cv_did()`, `cv_summary()`, `influence_diagnostics()`, `abundance_did()`, `pseudobulk_did()`, `test_treatment_heterogeneity()`, `did_fit()` and `within_arm_comparison()`. The interval of `within_arm_comparison()` uses the normal quantile with the cluster-robust covariance, not a t quantile. `ensure_fdr()` adds its column in place and returns the same table. `add_log1p_cpm_layer()` raises on non-finite counts and records its scale in `adata.uns["sctrial"]["log1p_cpm_scale"]`. `load_sade_feldman()` downloads only with `allow_download=True`, which the README example now passes. `min_cells_per_participant_visit` of `load_vaccine_gse171964()` reports thin groups and removes no cell. Lists in docstrings render as lists in the API reference.
- `subset_primary()` works with pandas 3. The array of an `obs` column can be a read-only view there, and the function updated its mask in place, so every analysis of a design with a crossover column raised `ValueError: output array is read-only`. The message for a missing counts layer in the benchmark input preparation lists the layer names under anndata 0.13, which adds a `None` key for `X`.

## [1.0.0] - 2026-10-05

Version bump to stable release. No functional changes from 0.3.4.

### Added
- `simulate_trial_v2()` and `TranscriptomeSimConfig`: transcriptome-scale hierarchical gamma-Poisson simulator calibrated from real single-cell data, superseding the original `SimulationConfig` API
- TNBC (triple-negative breast cancer) dataset support and accompanying tutorial
- Second-order DiD analysis demonstrated in tutorials
- Signal-fraction sensitivity benchmark (gene panel × signal percentage grid)

### Fixed
- Double log bug in benchmark orchestrator and permutation modules (`orchestrator.py`, `permutation.py`)
- GSEA ranking coerced to finite, unique values before `gseapy.prerank` — infinite betas from degenerate OLS and duplicate gene symbols caused "float division by zero" across all five AML gene-set libraries
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
- GSEA ranking coerced to finite, unique values before `gseapy.prerank` — infinite betas from degenerate OLS and duplicate gene symbols caused "float division by zero" across all five AML gene-set libraries
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
