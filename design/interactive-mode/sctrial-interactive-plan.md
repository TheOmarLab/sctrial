# sctrial analysis workbook

Open a study, define a comparison, and inspect the participants behind its estimate. Each analysis sheet records the outcome, cell population, groups, visits, and design.

## Open a study

Download the [offline workbook](sctrial-interactive-preview.html?raw=true) and open the HTML file in a browser. The bundled study is synthetic. **Open study** accepts a participant-summary CSV and includes a CSV template.

For a local H5AD file, install the package from this branch and launch the workbook.

```bash
python -m pip install -e .
python -m sctrial.interactive study.h5ad
```

Map the participant ID and one to twelve numeric score columns from `adata.obs`. Map the group, visit, and cell population where available. The workbook supports one or two fixed groups. The local server reads the mapped observation columns and leaves the source file intact. Close it with Ctrl+C.

| Input | Participant measurements | Available output |
| --- | --- | --- |
| Local H5AD | Cell-level scores averaged by participant, visit, and selected population | Participant-level model estimates, intervals, p-values, and FDR when estimable |
| Participant-summary CSV | One row per participant, group, visit, and population, including cell count and score columns | Descriptive changes, group differences, and counts |
| Synthetic study | Twelve participants in two groups across three visits | Precomputed reference fits |

## Work through a comparison

1. Choose an **Outcome**. In **Configure**, select the design, group order, population, and visits. Longitudinal change is the comparison visit minus the reference visit. A two-group effect takes the first group minus the second.
2. Read the effect and its interval. The points below show individual observed changes or scores. Select a point or a column in **Sample pairing** to inspect the participant's measurements. Dense plots show counts per bin and link to the searchable measurement table.
3. In **Participant measurements**, choose **Excluded** to see which measurement is missing for this outcome. Filtering and sorting the table leave the fitted cohort intact. A measured zero contributes; a blank score is missing.
4. Use **Trajectories** to inspect visits in study order. A group mean at each visit uses everyone measured at that visit. **Changes** uses participants with measurements at both selected visits. The numbers can differ when participants miss a visit.
5. Create or duplicate a sheet under **Analyses** to compare another design. **Pin contrast** keeps a snapshot. **Results table** shows the full outcome panel, including uncertainty for fitted comparisons.

## Save and reopen

Choose **Export → Save analysis workbook** to download named sheets and their settings. Reopen the data, then use **Open study → Open analysis workbook** to add the saved sheets. The workbook file contains study and column labels plus analysis definitions. Participant records and fitted results stay out of that file.

Results, comparison settings, and participant summaries each have their own export. The participant-summary export includes participant IDs. Local H5AD fits weight participants equally and require finite scores at the selected visits. A single group observed at a single visit yields a descriptive mean.
