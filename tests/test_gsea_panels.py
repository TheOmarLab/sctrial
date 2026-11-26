"""GSEA panels of Figure 4 and Supp Fig 7: pathway names and significance come from the table."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def test_numbered_label_makes_repeats_unique(load_figure_module):
    f4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    seen: dict = {}
    assert [f4._numbered_label(x, seen) for x in ["A", "B", "A", "A"]] == ["A", "B", "A (2)", "A (3)"]


def test_leading_edge_keeps_every_pathway_whose_display_names_collide(load_figure_module):
    """Three terms that truncate to the same 32-character name must stay three columns."""
    f4 = load_figure_module("main/figure4_biological_discovery_multi_dataset")
    stem = "Antigen Processing And Presentation Of "
    rows = [
        {"Term": stem + "Exogenous Peptide Antigen", "NES": 2.5, "FDR q-val": 0.01,
         "Lead_genes": "CTSD;CTSL;LGMN"},
        {"Term": stem + "Peptide Antigen Via MHC Class II", "NES": 2.4, "FDR q-val": 0.01,
         "Lead_genes": "CTSS;CD74"},
        {"Term": stem + "Endogenous Peptide Antigen", "NES": 2.3, "FDR q-val": 0.01,
         "Lead_genes": "TAP1;TAP2"},
        {"Term": "Oxidative Phosphorylation", "NES": -2.2, "FDR q-val": 0.01,
         "Lead_genes": "NDUFA9;NDUFB7"},
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    f4._tnbc_leading_edge(ax, {"gsea_results": pd.DataFrame(rows)}, composite=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    collided = [lab for lab in labels if lab.startswith("Antigen Processing")]
    assert len(collided) == 3 and len(set(collided)) == 3
    genes = {t.get_text() for t in ax.get_yticklabels()}
    assert {"CTSD", "CD74", "TAP1"} <= genes
    plt.close(fig)


def test_supp_fig7_pathway_bars_use_the_table_fdr(load_figure_module):
    """A pathway with GSEApy FDR 0.8 is drawn as not significant; no FDR is recomputed."""
    sf7 = load_figure_module("supp/supp_fig7_cross_dataset_biology")
    rows = [{"Term": f"Up pathway {i}", "NES": 2.0 - 0.05 * i, "NOM p-val": 0.0,
             "FDR q-val": 0.8, "Lead_genes": "MS4A1", "Library": "GO_BP"} for i in range(8)]
    rows += [{"Term": f"Down pathway {i}", "NES": -2.0 + 0.05 * i, "NOM p-val": 0.0,
              "FDR q-val": 0.001, "Lead_genes": "IFIT1", "Library": "Hallmark"} for i in range(7)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sf7.panel_C(ax, {"gsea_results": pd.DataFrame(rows)})
    alphas = {round(mcolors.to_rgba(p.get_facecolor())[3], 2) for p in ax.patches
              if p.get_width() > 0}
    full = {round(mcolors.to_rgba(p.get_facecolor())[3], 2) for p in ax.patches
            if p.get_width() < 0}
    assert max(alphas) < min(full)  # responder bars faded (n.s.), non-responder bars solid
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("n.s." in text for text in legend)
    plt.close(fig)


def test_supp_fig7_waterfall_fades_non_significant_genes(load_figure_module):
    """Same alpha trap in the melanoma gene waterfall: n.s. genes must be drawn faded."""
    sf7 = load_figure_module("supp/supp_fig7_cross_dataset_biology")
    genes = ["KRT13", "IGLC2", "IGLC3", "CD19", "MS4A1", "SMIM22", "S100A12", "CXCL10", "C3", "FABP4"]
    df = pd.DataFrame({
        "feature": genes,
        "beta_DiD": [2.1, 1.8, 1.7, 1.5, 1.4, -2.0, -1.9, -1.6, -1.5, -1.4],
        "p_DiD": [0.01, 0.2, 0.3, 0.4, 0.5, 0.006, 0.2, 0.3, 0.4, 0.5],
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    sf7.panel_B(ax, {"gene_results": df})
    alpha = {t.get_text(): round(p.get_facecolor()[3], 2)
             for t, p in zip(ax.get_yticklabels(), ax.patches)}
    assert alpha["KRT13"] == alpha["SMIM22"] == 0.9
    assert alpha["IGLC2"] == alpha["CXCL10"] == 0.35
    plt.close(fig)
