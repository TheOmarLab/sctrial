# sctrial analysis workbook

Open a study, select a cell population, and inspect the participants behind a comparison. Each analysis sheet records the outcome, population, groups, visits, and design.

## Open a study

The workspace opens with a synthetic study. To work offline, download the [standalone HTML](sctrial-interactive-preview.html?raw=true) and open it in a browser. Use **File → Open study** to load a participant-summary CSV or download the CSV template.

To analyze a local H5AD file, install the package from this branch and start the local server.

```bash
python -m pip install -e .
python -m sctrial.interactive study.h5ad
```

Map the participant ID and, where available, group, visit, and cell population. Choose up to twelve measurements across numeric `adata.obs` columns and genes in `adata.X`. In **Genes from X**, type the first two characters of a variable name and select a match from `adata.var_names`. A gene’s participant-level value is the arithmetic mean of its stored expression across observed cells for that visit and population. Counts, normalized values, and log-transformed values retain their input scale. The expression reader supports dense, CSR, and CSC matrices and up to 12 million selected cell–gene values. Leave this field empty to work from observation columns alone.

For an observation column containing a precomputed pathway or module score, choose its **Measurement type**. The workbook supports one or two fixed groups and reads the selected measurements locally. The H5AD file remains intact. Stop the server with Ctrl+C.

| Study | Measurements | Output |
| --- | --- | --- |
| Local H5AD | Gene expression and observation scores averaged within each participant, visit, and population | Participant-level model estimates, intervals, p-values, and FDR when estimable |
| Local CSV | One row per participant, group, visit, and population, with a cell count and score columns | Descriptive changes, group differences, and participant counts |
| Synthetic example | Twelve participants in two groups across three visits | Precomputed reference fits |

For a participant-summary CSV, use headers such as `gene:ISG15`, `pathway:Interferon response`, or `module:Cytotoxicity` to classify measured columns. An unprefixed column remains an ordinary numeric outcome.

## Explore a study

Choose a population in **Cells**. Switching views keeps the study and population selected.

| View | What the figure shows |
| --- | --- |
| **Cells** | Rotate or zoom an illustrative cell and select its population to see the observed cell and participant counts. |
| **Expression** | Inspect the participant-by-gene heatmap at a chosen visit. Each gene has its own color scale. Select a measurement to see its value and units. |
| **Pathways** | Inspect a measured pathway score. When a gene-set definition is available, select a member to inspect its expression where measured. For a private study, **Link gene set** lets you choose and confirm a definition from a local GMT file. Linking a definition preserves the supplied score. |
| **Scores** | Compare distributions within each arm at a chosen visit. Points represent participants. The white mark shows the arm median. Each score has its own axis. |
| **Time** | Follow participants across observed visits. Playback highlights each visit and preserves the fitted comparison. Visit means include everyone measured at that visit. |

In a molecular view, **Analyze gene**, **Analyze pathway**, or **Analyze score** opens the corresponding population and outcome in the analysis plot. The synthetic example uses simulated expression and illustrative gene sets.

## Analyze a population

1. In **Cell populations**, select an illustrated cell to choose one of the populations available in the study. The adjacent counts report the observed abundance. The illustration is schematic.
2. Select **Analyze population** to open its analysis plot. Choose the outcome and visits in the plot header. **Analysis → Configure comparison** sets the design and group order. Paired change is the comparison visit minus the reference visit. Between-group contrasts subtract the second group from the first.
3. Select a measurement to inspect the participant record. Paired lines join the same participant across visits. For larger groups the plot labels how many traces are drawn. The estimate uses all eligible participants. A measured zero contributes. A blank score is missing.
4. Use **View** beside the analysis tabs to switch to **Participant measurements**, **Time course**, or the analysis plot. In the measurement table, **Excluded** identifies missing scores at the required visits. Table filters change which rows are visible. The complete-pair estimate includes participants measured at both selected visits.
5. Use **Analysis → New analysis** to work on another comparison. **Pin contrast** keeps a snapshot. **View → Results table** shows the estimates and participant counts for the full outcome panel.

## Save and reopen

**File → Save workbook** downloads the named sheets and their settings. Reopen the same study, then choose **File → Open study → Open analysis workbook** to add those sheets. The workbook contains study and column labels plus analysis definitions. It omits participant records and fitted results.

**File → Export results** also offers separate exports for results, comparison settings, and participant summaries. The participant-summary CSV includes participant IDs. Local H5AD fits weight participants equally and require finite scores at the selected visits. A single group at a single visit yields a descriptive mean.
