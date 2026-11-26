# sctrial Interactive

We want a researcher to open their own single-cell study, see exactly which participants support a comparison, and reproduce the result in Python. The first release should do those three things exceptionally well. The primary design decision is to make the participant and visit structure visible before anyone fits a model.

[Open the comparison preview](sctrial-interactive-preview.html). Download the HTML file and open it in a browser. It works offline. The example contains 12 synthetic participants. Try switching from week 4 to week 12, restricting to CD8 T cells, and reversing the group order. “Inspect contrast” shows what would be fitted. The preview reads only the built-in example. Local file import, model fitting, and reproducible exports are engineering milestones below.

## The first five minutes

| Step | What a researcher does | What the interface gives back |
|---|---|---|
| Open a study | Choose an annotated local H5AD file or open the bundled example. | A short inventory of participants if identifiable, cells, expression layers, and available annotations. Opening a file never modifies it. |
| Define the study | Map participant, visit, arm or group, optional cell type, and the source of the outcome. Confirm whether groups represent assigned treatment or an observed phenotype. | A compact map from study roles to actual columns, with ambiguous fields left open. A sample ID requires confirmation that it tracks the same person across visits. |
| Review the comparison | Choose the study design, visit or ordered visit pair, groups, features, and population. | The comparison in words, participants per group, complete pairs when relevant, and exclusions with reasons. Changing a selection updates all four in place. |
| Run the analysis | Confirm the contrast direction, outcome units, estimation method, and the family of features to test. | A cancellable local run. Editing the design after submission creates a new draft. The completed run keeps its original settings. |
| Interpret and export | Select a feature, inspect the participant observations and estimate, then export. | A figure, complete results table, structured settings, and runnable Python. The result records who contributed, what was compared, and how uncertainty was calculated. |

The first-run example should start at the comparison screen with a link back to its mappings. A researcher bringing their own data enters through the file and mapping steps. Returning to a saved run should open its results directly.

## The comparison screen

“Review the comparison” is the central screen. A researcher can answer three questions without opening the settings panel. Which groups and visits are being compared? How many participants contribute? What quantity will the model estimate?

On a wide screen, give roughly three quarters of the width to the evidence and the remainder to comparison settings. Put visit coverage above the analysis question. For small studies, each column represents one participant. A filled circle denotes an observed participant-visit in the selected population, an open circle denotes a missing observation, and a line joins the two observed visits from the same person. Keep group labels, ordered visit labels and counts visible. List exclusions underneath. Use charcoal type, a warm neutral surface, thin rules, and a single amber accent for the selected group. Keep the sign of a biological effect independent of this color.

On a narrow screen, show the evidence first, with direct links to settings and back to the updated evidence. Support native keyboard selection, visible focus, a text equivalent for every mark, and a readable contrast at 200% zoom. For larger cohorts, replace the per-person columns with an arm-by-visit count table and a separate complete-pair count per arm. A local drill-down gives the same exclusions without rendering thousands of markers. Start with a visible-space cutoff and test it against actual viewport sizes.

### Three study designs

| Study design | Comparison presented to the researcher | Package entry point |
|---|---|---|
| Two-arm or two-group longitudinal | Change from baseline to follow-up in A minus the corresponding change in B. Show complete pairs within each group. Use treatment language only when group definitions and study design support it. | `did_table()` |
| Single-arm paired | Within-participant change from baseline to follow-up, in the selected arm or the whole single-arm study. | `within_arm_comparison()` |
| Cross-sectional | A minus B at the specified visit. Count independent participants in each group. A participant needs only the selected visit. | `between_arm_comparison()` |

Use the study’s actual arm and visit names in the interface. “Responder” and “Non-responder” are observed groups. A contrast between their trajectories describes differential change by response status. With more than two arms or more than two visits, ask for the specific pair and show the omitted levels. For a cross-sectional study without a visit field, let the researcher declare a single collection occasion and record any constant visit field needed by the package in the exported script. For a single-arm study, support an absent arm column directly through the package’s single-arm convention.

## Statistical behavior

- **Participant eligibility.** Distinguish the full study, participants with coverage after selecting visits and cell type, and the feature-specific number actually fitted. Obtain fitted denominators from the model output and the same model-frame preparation used by the package. A missing covariate or feature can change that count. A cell type absent from a specimen provides no expression measurement for that cell type. Show the reason for each exclusion, including any active crossover rule.
- **Mapping and preprocessing.** Use `auto_detect_design()` to suggest mappings when it succeeds. A failed suggestion opens manual mapping. Construct the confirmed `TrialDesign` and call `diagnose_trial_data()`. Whole-study diagnostics and comparison-specific eligibility have different denominators and need distinct labels. The user selects an existing numeric observation score or a gene and expression layer for the first release. Record whether values are standardized. Keep raw counts, normalized expression, and existing scores explicit.
- **Estimation.** Default to participant-level aggregation for all three routes. Keep the package’s weighting and covariance defaults. Show any actual covariance fallback from the fit. Offer restricted wild cluster bootstrap where the selected package function supports it. Record settings, seed, and the returned method. Direct unadjusted group means in a plot are descriptive. An adjusted or standardized coefficient retains its own label and units.
- **Uncertainty.** The plot, table, and export must use the same interval and test. The current interactive DiD forest helper constructs symmetric intervals from the SE. A bootstrap run must plot the returned test-inversion endpoints, including asymmetric or unbounded results. If a supported path lacks compatible endpoints, report the estimate with the available uncertainty fields and label the interval unavailable until a common package helper supplies it.
- **Multiplicity.** Show the tested feature panel, cell population, FDR adjustment scope, raw p-value, and FDR-adjusted p-value. Sorting or filtering a completed result changes the view only. Selecting a different feature panel for testing starts a new run and records a new family.
- **Unavailable results.** Distinguish an inestimable coefficient, an estimable effect without a valid test, and a fitted effect whose interval includes zero. Retain the row and its explanation in all three cases. Warnings belong beside the affected result. Reversing group order must reverse the sign of an estimable two-group contrast and preserve its two-sided p-value for deterministic analytical fits.

## Results and reproducibility

The first results view should show the question, a compact forest plot, and the fitted participant count for each feature. Reserve color for selection. When a feature is selected, a detail view shows the group-specific changes or cross-sectional distributions on a labeled scale. Participants with repeated measurements can be connected locally. Give the user an accessible numerical table for each view.

“Export analysis” writes a folder containing the full results CSV, figure, versioned settings JSON, and a plain Python script that takes the original data path as an argument. Include the package and dependency versions, input fingerprint, mapping and ordering, layer, selected features, grouping, aggregation, inference options, seed, warnings, and adjustment family. Persist the code needed to reconstruct any derived field. The analyst should be able to run the script in a fresh environment and recover the numerical table. Keep participant-level observations and local paths out of the standard figure and summary bundle. Supply individual-level exports only through a separate local selection.

## Local implementation

**Entry point.** Add an optional `interactive` installation extra and a proposed `sctrial interactive` launcher. Support an optional positional H5AD path for users who already work in a terminal. Start a loopback-only service and open the browser to its session. Ship a small offline synthetic example and frontend assets with the installed extra. A core-only installation should retain its current dependency footprint. Test Python compatibility on the project’s supported versions and declare any narrower requirement for the extra explicitly.

**Interface and execution.** Use a small TypeScript/React frontend served as static files by a local Python service. A contributor needs the frontend build tools; an end user installs Python packages only. The service owns the file handle, immutable run settings, an isolated analysis worker, and the result tables. Pass only the fields needed for the current view to the browser. Use an explicit local file selection or command-line path and avoid copying a large file through the browser when a native path is available. Limit file access to the selected input and output destinations. Bind to loopback, validate the browser origin and session token, and bundle fonts and visual assets locally. The study can be analyzed with outbound network access disabled.

**Read once, compute on request.** Read metadata and layer names first. Defer expression access until the requested features are known. Backed H5AD access needs separate tests for dense, sparse, and layered inputs because individual package calls can materialize slices. Cache eligibility by input identity, mapping, visits, grouping, and cell population. Cache fits by the full analysis specification and package version. Changing a display sort or selecting a result should never start a fit. Keep a bounded worker count, report the current operation, make cancellation release open resources, and commit results atomically after completion.

**One source for the calculation.** Use a versioned run specification to drive validation, the approved package call, and the Python export. Keep analysis code in Python. Route all three study designs through their existing public functions. Restrict by the selected population before calling a function that lacks a cell-type argument. If a canonical denominator or interval is unavailable, extend and test the relevant shared package routine so the notebook and interface can report the same quantity.

**Integration with other research software.** Once the local workflow is stable, expose the same versioned specification and a read-only aggregate summary to other clients. A collaborator can prepare a proposed comparison or ask which estimates have usable uncertainty. The researcher sees the exact fields selected for sharing and confirms any new run in the local interface. For external use, omit participant rows, raw values, paths, free-text metadata, and small or identifiable strata. Apply complementary suppression where totals would reveal a hidden count. The standard workflow requires only the local interface and Python.

## Build order and acceptance

| Milestone | Deliverable | Acceptance criteria |
|---|---|---|
| 1. Open and map | Installable local launcher, offline example, H5AD inspection, manual and suggested mapping. | A fresh install opens the example offline. Unfamiliar column names, an ambiguous sample ID, a missing arm in a single-arm study, and a missing visit in a declared single-occasion study all have an explicit resolution. The original input checksum is unchanged. |
| 2. Run one comparison end to end | Compare two groups over two visits, first with an existing score and then a selected gene. Export the full run. | On a frozen fixture and a documented public tutorial cohort, the direct Python call, interface result, and exported rerun agree on feature identities, counts, estimates, tests, and available intervals within a declared tolerance. Reverse the groups and introduce a missing participant-visit to verify the sign and denominator. |
| 3. Cover all three designs | Add the paired and cross-sectional routes, local participant detail, documented uncertainty, and sparse-data states. | A single-arm paired fixture estimates change without a control. A one-visit fixture fits without requiring pairs. A missing feature, empty cell population, unbounded interval fixture, exact fit, and unidentified contrast each produce their intended display and export. |
| 4. Validate usability and performance | Package the frontend and test actual installs, keyboard access, restart, cancellation, and larger inputs. | At least four of five first-time users should reach and explain the correct contrast unaided within five minutes on the example. Target a warm visit or population change under 250 ms at p95 on the reference fixture. The benchmark report must identify file shape, encoding, hardware, cold and warm times, peak memory, and direct-Python fit time. Treat these numbers as targets until measured. |

For the first milestone, benchmark a small tutorial H5AD and a sparse stress fixture of roughly one million cells on a named 16 GB machine. The opening interaction should not materialize the whole expression matrix. Measure responsiveness independently of fit duration. For the production review, test a dense file and a multilevel categorical input as well. A cancelled or interrupted run must never appear complete.

The first release should support one specified comparison, a user-selected panel of genes or existing scores, and full export across the three designs. Add custom gene-set scoring, abundance analysis, GSEA, power, and multi-contrast workflows after the initial routes have numerical parity and have been tested by researchers outside the development team. Each extension should reuse the established study mapping and result provenance.

## References for implementation

- [Trial design and validation](https://github.com/TheOmarLab/sctrial/blob/0315adfb91023d774b595cf5185d72874cd3b5ba/src/sctrial/validation.py)
- [Difference-in-differences](https://github.com/TheOmarLab/sctrial/blob/0315adfb91023d774b595cf5185d72874cd3b5ba/src/sctrial/stats/did.py)
- [Paired and cross-sectional comparisons](https://github.com/TheOmarLab/sctrial/blob/0315adfb91023d774b595cf5185d72874cd3b5ba/src/sctrial/stats/comparisons.py)
- [Existing plots and interval construction](https://github.com/TheOmarLab/sctrial/blob/0315adfb91023d774b595cf5185d72874cd3b5ba/src/sctrial/plotting.py)
- [Public study tutorials](https://www.omar-lab.com/sctrial/)
