# sctrial Contrast Studio

Explore how individual participants contribute to a longitudinal comparison. Choose the study design, group order, cell population, and visits. The trajectories, participant inspector, and outcome estimates follow the same selection.

## Open a study

From a checkout of this branch, install the package and start the local workspace.

```bash
python -m pip install -e .
python -m sctrial.interactive study.h5ad
```

The browser opens on the local machine. Map the participant ID and one to twelve numeric score columns from `adata.obs`. Map the group, visit, and cell population when available. The workspace supports one or two fixed groups. It reads the selected observation columns and leaves the source file unchanged. Close the local server with Ctrl+C.

To try the interface without installing Python, download the [self-contained workspace](sctrial-interactive-preview.html) and open it in a browser. It includes a synthetic study with precomputed sctrial results. **Open study** also accepts a participant-summary CSV and provides a downloadable template.

## Work through a comparison

1. **Define the contrast.** Choose a paired change, difference-in-differences, or group difference at one visit. For longitudinal comparisons, the change is follow-up minus baseline. For two groups, the effect is the first group minus the second.
2. **Inspect the participants.** Select a trace or a roster ID to see its visit measurements and cell counts. Move the visit handle or use its arrow keys to choose a follow-up. Switch to **Changes** to see one point per participant with a complete pair. Selecting a participant highlights their record and preserves the cohort fit.
3. **Read the estimate.** The inspector reports the selected score, its interval when available, and its contributing participant count. Switch scores in the outcome panel. Open **Results** for the full table, including model p-values and adjusted values for fitted comparisons.
4. **Compare decisions.** Pin a contrast, change the follow-up or population, and pin again. In **Pins**, compare both sets of estimates and reopen either configuration. Pins from a local study last for the current study session.
5. **Export.** Save results as CSV and comparison settings as JSON. For a local H5AD fit, the settings include the mapped columns and package version.

## What each input supports

| Input | Observations | Results |
| --- | --- | --- |
| Local H5AD | Cell-level scores in mapped observation columns, averaged within participant and visit for the selected population | sctrial model estimates, intervals, p-values, and FDR when estimable |
| Participant-summary CSV | One row per participant, group, visit, and cell population, with a cell count and numeric scores | Descriptive changes and group differences |
| Bundled synthetic study | Twelve participants in two groups measured across three visits | Precomputed reference fits for the available comparisons |

The local fit uses participant-visit means with equal participant weighting. A longitudinal outcome requires a finite score at both selected visits. Coverage in the roster counts visits with cells, so an outcome-specific model can include fewer participants. A single group measured at one visit yields a descriptive mean.

The browser draws a bounded set of individual traces for larger studies. Group summaries and fits use the full eligible cohort. The interactive mapping supports up to 50,000 participant-visit-population summaries. Large fixed-effects designs can be run through the package's Python workflow.
