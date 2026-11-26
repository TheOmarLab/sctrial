# sctrial Interactive

The workspace connects a single-cell study’s participant and visit structure to its statistical results. An investigator can select a comparison, inspect the observations behind an estimate, and export the result with its study design.

[Open the interactive preview](sctrial-interactive-preview.html). The offline preview includes a synthetic longitudinal study with precomputed model results. Select a feature in the forest plot to inspect its trajectories. Researchers can also import a local CSV with one row per participant, arm, visit, and cell population, including cell counts and numeric scores, to explore and export descriptive summaries. H5AD input and model fitting on imported data are planned for the local Python interface.

## Workspace

| View | What the investigator can inspect |
|---|---|
| Overview | Cohort size, visit coverage, group trajectories, and feature estimates. The forest plot and trajectory view share the selected feature. |
| Participants | Specimen availability by visit, coverage in the selected cell population, and reasons for exclusion. |
| Results | Effect estimates, interval bounds, fitted participant counts, raw and FDR-adjusted p-values, and the selected contrast. |
| Analysis settings | Study design, group order, visits, and cell population. These selections apply across the workspace. |

Keep group colors consistent across views. Label trajectories with their measurement units and the number of contributing participants at each visit. Show the fitted contrast on its own axis. Each plot needs a numerical equivalent, keyboard access, and a usable narrow-screen layout.

## Comparison definitions

| Design | Quantity estimated | Package function |
|---|---|---|
| Two-group longitudinal | Change from baseline to follow-up in the first group minus change in the reference group | `did_table()` |
| Single-arm paired | Mean within-participant change from baseline to follow-up in the selected arm | `within_arm_comparison()` |
| Cross-sectional | Difference between the first and reference groups at one visit | `between_arm_comparison()` |

Use participant-level aggregation and the study’s own group labels. Describe contrasts between observed response groups as associations. Display the full cohort size, the number with the required specimens and cell population, and the number included in each feature-specific fit. An absent cell type contributes no expression measurement.

Intervals, tests, and fitted counts must agree across plot, table, and export. For a restricted wild cluster bootstrap, use the returned test-inversion bounds, including asymmetric or unbounded intervals. Preserve unavailable results with the reason. Record the tested feature family and retain its FDR adjustment when the investigator filters or sorts the table.

## Running a local study

1. **Open and map.** Offer an optional `sctrial interactive` launcher that opens a local browser session. Read H5AD metadata first. Confirm participant ID, ordered visits, arm or group, cell type, and the score or expression layer. Use `TrialDesign` and `diagnose_trial_data()` after confirmation. Support a single-arm study without an arm column and a declared single collection occasion without a visit column.
2. **Fit.** Send the confirmed settings to the existing Python function. Read only the needed expression features where the file encoding permits. Record the aggregation, outcome scale, test, covariance method, seed, and exclusions. Run the fit outside the interface thread. Publish a completed result only after the worker finishes successfully.
3. **Export.** Write the result table, figure, settings, and a Python script that accepts the original H5AD path. Include input fingerprint, package versions, mappings, contrast direction, and uncertainty method. Keep participant-level exports separate from aggregate results.

The example study and its fitted results can be regenerated with [the reference script](https://github.com/TheOmarLab/sctrial/blob/feature/interactive-mode/design/interactive-mode/build_reference_study.py).

## Release checks

- Compare all three designs against direct Python calls and exported reruns on a fixed synthetic study and public tutorial data. Check feature-specific counts, effect direction, intervals, tests, and FDR-adjusted p-values.
- Exercise missing visits and covariates, empty populations, unidentified contrasts, and estimates with unavailable uncertainty. Verify that reversing group order reverses the sign of an estimable two-group contrast.
- Test a fresh install offline, dense and sparse H5AD inputs, cancellation, and reopening a completed run. Record load time, peak memory, and fit time on specified hardware. Changing the selected result should reuse the completed fit.

[Statistical methods and study tutorials](https://www.omar-lab.com/sctrial/)
