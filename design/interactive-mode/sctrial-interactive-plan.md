# sctrial cell atlas

## Open your H5AD

Choose **Open H5AD** in the [downloadable atlas](sctrial-interactive-preview.html?raw=true), or launch the installed branch with a filename.

```bash
python -m pip install -e .
python -m sctrial.interactive study.h5ad
```

The file picker reads the selected H5AD in a browser worker. The Python launcher binds to loopback and serves requested byte ranges from the file you named. Both paths use the same reader and all captured cells at the two selected visits. Running `python -m sctrial.interactive` without a filename opens the public TNBC example and the file picker is available in the header.

Map the population and visit columns from `obs`, then select the two recorded visits you want to compare. The participant and arm columns are optional. Participant IDs must identify the same person across the file. When a participant has multiple specimens at one visit, their displayed fraction pools those specimens’ cells within the selected arm. The pooled fraction uses all captured cells in the selected arm and visit. Cells with missing population annotations remain in the denominator under a separate label.

Choose a stored embedding from `obsm`, or choose two measured genes as the horizontal and vertical coordinates. For expression, select `X`, a named layer, or `raw/X` and the matching variable identifiers. Select the scale of that matrix. For raw counts, the reader divides by each cell’s library size across all features of that matrix, scales to 10,000, and computes `log1p`. The other settings preserve the selected nonnegative expression values.

Add genes by their exact identifiers. The three bundled human Reactome sets use the 2022 collection. A local GMT can supply additional sets, including sets for other organisms. Each cell’s pathway score is the unweighted mean expression of its uniquely matched member genes. The interface reports coverage. Use **Study settings** to select another visit pair, expression matrix or panel from the same file.

## Explore visits

Select a population on the map or in the inspector. **Visit lens** reveals cells from the other visit at the same coordinates inside a movable circle. Drag the lens or focus the map and use the arrow keys. The timeline changes the opacity of each visit’s observed cells. The fraction plot connects paired measurements for participants represented at both visits in the selected group.

**Gene expression** colors cells of the selected population on a common feature scale and displays detection counts and positive-expression histograms. **Pathways** shows each measured member gene at both visits. Select **Cell map** in that view to inspect per-cell pathway scores with the lens. Per-cell map colors use 256 display levels. The numerical summaries use the expression matrix at its original precision.

The bundled example contains 141,553 CD45+ cells from paired tumor biopsies in 12 participants. [Zhang et al., Cancer Cell, 2021](https://doi.org/10.1016/j.ccell.2021.09.010) · [GSE169246](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169246)

## Participant-level analyses

Open **Workspace** to configure participant-level comparisons and fitted contrasts for the H5AD named at launcher startup, including precomputed numeric pathway or module scores in `obs`. The worksheet also supports participant-summary CSVs and saved analysis settings.

## Open a participant-summary CSV

Download the [standalone worksheet](sctrial-analysis-worksheet.html?raw=true) and open it in a browser. Choose **Open study** to load a CSV or download the example template.

Supply one row per participant, visit, and population. Required columns are `participant`, `arm`, `visit`, `population`, and `cells`. Use a positive observed cell count. Supply at least one numeric outcome, leaving missing measurements blank. Prefix outcome names to choose their display layer, for example `gene:ISG15`, `pathway:Interferon response`, or `module:Cytotoxicity`. Unprefixed columns remain ordinary recorded scores. CSV plots report descriptive changes or group differences and the number of contributing participants.

## Save an analysis

**File → Save workbook** downloads the analysis settings. Open the same study and use **File → Open study → Open analysis workbook** to restore them. **File → Export results** offers separate exports for estimates, comparison settings, and participant summaries. The participant-summary export includes participant IDs.
