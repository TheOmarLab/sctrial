# sctrial interactive workspace

The canvas opens with a synthetic study of 12 participants in two arms across three visits. Each cluster represents an annotated cell population. Select a cluster, choose a measurement, and plot a participant-level comparison while keeping the clusters in view.

## Explore the clusters

1. Select a cell population on the canvas. The illustrated cells identify that population. Use the reported counts for abundance. Cluster positions and distances are schematic.
2. Choose **Expression** and select a gene, or choose **Pathways** or **Scores** and select a recorded outcome. The same clusters stay in place. The ring around each cluster shows the mean across participants with a measurement for that population at the selected visit. Participants contribute equally, even when they contributed different numbers of cells. A measured zero contributes to the mean. A missing measurement is excluded. A dashed gray ring marks a population without a measured value.
3. Change **Visit**, or play the visits in order, to see how the measurements change. The color scale for a given measurement stays fixed as you move between visits. Participant counts can change over follow-up.
4. Choose **Plot results** for the selected population and outcome. Check the group order and visits in the plot header. For a paired change, the estimate uses participants measured at both visits. A two-arm difference-in-differences subtracts the second arm’s mean paired change from the first arm’s. The difference between two visit means on the canvas can differ from the paired estimate when participants miss a visit.

For a pathway with a known gene-set definition, **Gene set** lists its members. Use **Link gene set** to associate a local GMT definition with an imported pathway. Linking describes membership and leaves the measured pathway score intact.

## Open an H5AD study

From this branch’s checkout, install sctrial and run the local launcher.

```bash
python -m pip install -e .
python -m sctrial.interactive study.h5ad
```

The launcher opens a browser and reads the H5AD on your machine. Map the participant ID and, where available, the arm, visit, and cell population. The workspace supports one or two fixed arms. Confirm the visit order for longitudinal data.

Choose up to 12 measurements in total. Under **Genes from X**, type the first two characters of a variable name and select a match from `adata.var_names`. For precomputed pathway or module scores, select numeric columns from `adata.obs` and assign their measurement types. Each participant’s value is the mean across their measured cells for that population and visit. Expression retains the scale stored in `adata.X`. The launcher supports dense, CSR, and CSC matrices and up to 12 million selected cell–gene values.

For estimable H5AD contrasts, the result includes the participant-level estimate and confidence interval. A single arm at a single visit yields a descriptive mean. Stop the local server with Ctrl+C.

## Open a participant-summary CSV

Download the [standalone HTML](sctrial-interactive-preview.html?raw=true) and open it in a browser. Choose **Open study** to load a CSV or download the example template.

Supply one row per participant, visit, and population. Required columns are `participant`, `arm`, `visit`, `population`, and `cells`. Use a positive observed cell count. Supply at least one numeric outcome, leaving missing measurements blank. Prefix outcome names to choose their display layer, for example `gene:ISG15`, `pathway:Interferon response`, or `module:Cytotoxicity`. Unprefixed columns remain ordinary recorded scores. CSV plots report descriptive changes or group differences and the number of contributing participants.

## Save an analysis

**File → Save workbook** downloads the analysis settings. Open the same study and use **File → Open study → Open analysis workbook** to restore them. **File → Export results** offers separate exports for estimates, comparison settings, and participant summaries. The participant-summary export includes participant IDs.
