# sctrial interactive workspace

The Peripheral immune atlas opens with 18,840 simulated cells from 12 participants in two arms across three visits. At baseline the atlas shows 7,429 cells across five populations, from 45.4% CD4 T cells to 5.4% NK cells. Select a population, switch molecular layers, and plot a participant-level comparison on the same page. The earlier Longitudinal immune profiling example is available under **Open study**.

## Explore the clusters

1. Select a cluster to see its fraction and observed count. Each cluster’s area follows its share of captured cells at the selected visit and arm. A population representing 30% of the cells occupies three times the area of one representing 10%. Pooled cells are counted once. Cluster positions are schematic.
2. Choose **Gene expression**, **Pathways** or **Module scores**, then select a measurement. In an H5AD study, color within a cluster represents the distribution of its recorded cell values. The bundled study uses its original simulated cell values. Colors are sampled from the distribution for that population. The positions of individual marks carry no spatial or embedding information.
3. Select a cluster to inspect its measured-cell distribution. Hover a bin or move through the bins with the arrow keys to read its count. Switch to **Participants** to see one mean per participant for that population and visit, together with the median and interquartile range. Participant-summary CSVs open directly in this view. Blank measurements are excluded and measured zeros are retained.
4. Change **Atlas arm** or **Visit**, or play the visits. The sizes, counts, and measurements update together. Each feature keeps a common cell-color scale and a common participant axis across arms and visits. The two scales are labeled separately.
5. Choose **Plot results** to inspect the participant-level comparison alongside the atlas. The plot identifies its molecular outcome, arms, visits and population. Where available, the estimate and 95% confidence interval appear above the individual participant observations. A paired change uses participants measured at both visits. A two-arm difference-in-differences subtracts the second arm’s mean paired change from the first arm’s. The atlas arm filter controls exploration. The plot follows the configured contrast. Selecting a follow-up updates that contrast. Playing the timeline leaves the analytical follow-up in place.

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
