"""
Supplementary Figure 6 - Melanoma biological discovery + Cross-dataset consistency.

Panels:
  A  Gene-level volcano plot (melanoma DiD).
  B  Top genes ranked by effect size (waterfall, melanoma).
  C  GSEA enrichment bar chart (pathway enrichment, melanoma).
  D  Leading-edge gene overlap heatmap (melanoma).
  E  Cell-type-resolved DiD effect heatmap (melanoma).
  F  Gene-set score distributions (within-dataset z-score).
  G  All-pairs cross-dataset effect correlation (DiD/Δ labelled).
  H  Shared top genes with concordant direction (|β|>0.05 threshold).
  I  Gene-level effect distributions (DiD/Δ labelled).
  J  Exhaustion effects by cell type (proper two-sample SE for DiD).
  K  Effect heatmap across datasets (DiD/Δ labelled).
  L  Participant-level paired gene-set trajectories.
  M  Enrichment summary heatmap (within-dataset z-score).

Design-type handling:
  Two-arm datasets (Sade-Feldman, Stephenson, TNBC) use DiD estimand
  (treated Δ − control Δ).  Single-arm datasets (AML, CAR-T, Vaccine)
  use within-arm pre→post change (Δ).  Panels mixing both estimands
  label each dataset with (DiD) or (Δ) to avoid silent conflation.
"""

from __future__ import annotations

import gc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    add_log1p_cpm_layer,
    apply_style,
    clear_cache,
    despine,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_tnbc_zhang,
    get_vaccine,
    harmonize_response,
    save_panel,
    sig_display,
)
from ..main.figure4_biological_discovery_multi_dataset import (
    _clean_pathway_name,
    _compact_legend,
    _detect_gsea_columns,
    _is_likely_protein_coding,
    _prepare_bio_discovery_data,
    _shrink_colorbars,
    _swap_leading_edge_axes,
)

# ======================================================================
# Melanoma biological discovery panel helpers (supp figure A–E)
# ======================================================================

def _panel_A_signature_waterfall(ax, data: dict):
    """Fallback: signature DiD waterfall plot."""
    did_sig = data["did_sig"]
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=False).reset_index(drop=True)

    y_pos = np.arange(len(df))
    colors = [COLORS["treated"] if v > 0 else COLORS["control"]
              for v in df["beta_DiD"]]
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("Signature DiD Effects (Melanoma)", fontsize=11,
                 fontweight="bold")
    ax.invert_yaxis()
    despine(ax)


# ======================================================================
# Panel B -- GSEA Enrichment Bar Chart
# ======================================================================


def _panel_B_signature_waterfall(ax, data: dict):
    """Fallback: signature-level DiD effects bar chart."""
    did_sig = data["did_sig"]
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True)

    colors = [COLORS["treated"] if v > 0 else COLORS["control"]
              for v in df["beta_DiD"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("DiD Signature Effects", fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel C -- Leading-edge gene overlap heatmap
# ======================================================================


def _panel_C_did_summary(ax, data: dict):
    """Fallback: signature-level DiD effects bar chart."""
    did_sig = data["did_sig"]
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True)

    colors = [COLORS["treated"] if v > 0 else COLORS["control"]
              for v in df["beta_DiD"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("DiD Signature Effects", fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel D -- Signature DiD forest plot with bootstrap CIs
# ======================================================================


# Panel A — Gene-level volcano (Melanoma DiD)
def panel_A(ax, data: dict, *, composite: bool = False):
    """Volcano plot of gene-level DiD effects (Sade-Feldman).

    Labels prioritize protein-coding genes over pseudogenes/lncRNAs.
    When *composite* is True, fewer labels are drawn and adjustText
    is skipped to avoid cluttering the small composite axes.
    """
    gene_results = data["gene_results"]

    if gene_results is None or len(gene_results) == 0:
        ax.text(
            0.5, 0.5,
            "Gene-level results unavailable",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor=COLORS["gray"]),
        )
        ax.set_title("Gene-Level Volcano Plot", fontsize=11,
                     fontweight="bold")
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"

    p_col = "p_DiD"

    df = df.dropna(subset=[beta_col, p_col])

    # Standard volcano: nominal p on y-axis, colour by nominal p < 0.05.
    # Analytical (nonrobust) SEs provide gene-level resolution; bootstrap
    # inference is reserved for signature-level tests (Panel D).
    p_thresh = 0.05
    df["nlog10"] = -np.log10(df[p_col].clip(lower=1e-300))
    sig_mask = df[p_col] < p_thresh

    df["category"] = "ns"
    df.loc[sig_mask & (df[beta_col] > 0), "category"] = "up"
    df.loc[sig_mask & (df[beta_col] < 0), "category"] = "down"

    color_map = {
        "ns": COLORS["gray"],
        "up": COLORS["treated"],
        "down": COLORS["control"],
    }
    alpha_map = {"ns": 0.3, "up": 0.8, "down": 0.8}
    if composite:
        size_map = {"ns": 2, "up": 6, "down": 6}
    else:
        size_map = {"ns": 8, "up": 20, "down": 20}

    for cat in ["ns", "up", "down"]:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[beta_col], sub["nlog10"],
            c=color_map[cat], alpha=alpha_map[cat],
            s=size_map[cat], edgecolors="none", rasterized=True,
        )

    N_LABELS = 10
    labelled_genes: list[str] = []  # ordered by score (highest first)

    for sign in ("pos", "neg"):
        sub = df[df[beta_col] > 0].copy() if sign == "pos" else df[df[beta_col] < 0].copy()
        if len(sub) == 0:
            continue
        # Restrict to protein-coding genes only
        sub = sub[sub["feature"].apply(_is_likely_protein_coding)]
        if len(sub) == 0:
            continue

        # Force-include top 3 genes by -log10(p) in each direction
        # (ensures the most significant genes are always labelled)
        force_genes = set(
            sub.nlargest(min(3, len(sub)), "nlog10")["feature"].tolist()
        )

        # Combined score: rank-normalised |β| + rank-normalised -log10(p)
        # This naturally selects genes at volcano tips (high on both axes).
        sub = sub.copy()
        sub["_rank_beta"] = sub[beta_col].abs().rank(pct=True)
        sub["_rank_sig"] = sub["nlog10"].rank(pct=True)
        sub["_score"] = sub["_rank_beta"] + sub["_rank_sig"]

        candidates = sub.nlargest(min(N_LABELS * 3, len(sub)), "_score")

        # Deduplicate: skip genes too close to an already-selected one
        # (prevents overlapping arrows pointing to the same spot).
        x_range = df[beta_col].max() - df[beta_col].min()
        y_range = df["nlog10"].max() - df["nlog10"].min()
        min_dx = x_range * 0.025  # ~2.5% of axis range
        min_dy = y_range * 0.025
        selected_coords: list[tuple[float, float]] = []
        picks: list[str] = []

        # Add forced genes first
        for _, cand in candidates.iterrows():
            if cand["feature"] in force_genes and cand["feature"] not in picks:
                picks.append(cand["feature"])
                selected_coords.append((cand[beta_col], cand["nlog10"]))

        # Fill remaining slots with score-ranked candidates
        for _, cand in candidates.iterrows():
            if cand["feature"] in picks:
                continue
            cx, cy = cand[beta_col], cand["nlog10"]
            too_close = False
            for sx, sy in selected_coords:
                if abs(cx - sx) < min_dx and abs(cy - sy) < min_dy:
                    too_close = True
                    break
            if not too_close:
                picks.append(cand["feature"])
                selected_coords.append((cx, cy))
                if len(picks) >= N_LABELS:
                    break

        labelled_genes.extend(picks)

    labelled_set = set(labelled_genes)
    labelled_rows = df[df["feature"].isin(labelled_set)].copy()

    from adjustText import adjust_text as _adjust_text

    _lbl_fs = 3.5 if composite else 6.5
    _arrow_lw = 0.25 if composite else 0.4

    texts = []
    for _, row in labelled_rows.iterrows():
        t = ax.text(
            row[beta_col], row["nlog10"], row["feature"],
            fontsize=_lbl_fs, fontweight="bold", color="#444444",
            ha="center", va="center", zorder=5,
        )
        texts.append(t)

    x_span = df[beta_col].max() - df[beta_col].min()
    y_span = df["nlog10"].max() - df["nlog10"].min()
    _adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#888888", lw=_arrow_lw,
                        shrinkA=5, shrinkB=3),
        force_text=(3.0, 3.2),
        force_static=(3.5, 3.5),
        force_explode=(0.5, 0.6),
        expand=(2.3, 2.6),
        ensure_inside_axes=True,
        max_move=(x_span * 0.45, y_span * 0.45),
    )

    # Threshold line
    thresh_y = -np.log10(p_thresh)
    ax.axhline(thresh_y, color=COLORS["gray"], ls="--", lw=0.8, zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=0)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{DiD}}$)")
    ax.set_ylabel(r"$-\log_{10}$(p)")
    ax.set_title("Gene-Level Volcano (Melanoma DiD)", fontsize=11,
                 fontweight="bold")

    # Legend — no footnotes, no summary boxes
    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.8,
                       label="Responder ↑"),
        mpatches.Patch(color=COLORS["control"], alpha=0.8,
                       label="Non-responder ↑"),
        mpatches.Patch(color=COLORS["gray"], alpha=0.3,
                       label="Not significant"),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc="lower left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel C (new) -- Replicated pathways across cohorts
# ======================================================================


# Panel B — Top genes by effect size, waterfall (Melanoma)
def panel_B(ax, data: dict):
    """Top 30 protein-coding genes ranked by effect size (waterfall).

    Horizontal bar plot of the most extreme genes on each side,
    providing immediate biological interpretability.  Colour indicates
    direction *and* nominal significance (p < 0.05).
    """
    gene_results = data.get("gene_results")

    if gene_results is None or len(gene_results) == 0:
        # Fallback: signature DiD waterfall
        _panel_A_signature_waterfall(ax, data)
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"
    p_col = "p_DiD"

    df = df.dropna(subset=[beta_col, p_col])

    # Filter to protein-coding genes
    df = df[df["feature"].apply(_is_likely_protein_coding)]

    p_thresh = 0.05
    n_per_side = 15

    # Top positive and negative
    top_pos = df.nlargest(n_per_side, beta_col)
    top_neg = df.nsmallest(n_per_side, beta_col)
    selected = pd.concat([top_pos, top_neg]).drop_duplicates()
    selected = selected.sort_values(beta_col, ascending=True).reset_index(
        drop=True
    )

    y_pos = np.arange(len(selected))
    colors = []
    for _, row in selected.iterrows():
        sig = row[p_col] < p_thresh
        if row[beta_col] > 0:
            colors.append(
                COLORS["treated"] if sig else COLORS["treated"] + "55"
            )
        else:
            colors.append(
                COLORS["control"] if sig else COLORS["control"] + "55"
            )

    ax.barh(y_pos, selected[beta_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.3, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected["feature"].values, fontsize=4)

    ax.set_xlabel(r"Effect size ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("Top Genes by Effect Size — Melanoma DiD", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.9,
                       label="Responder ↑ (p < 0.05)"),
        mpatches.Patch(color=COLORS["treated"], alpha=0.35,
                       label="Responder ↑ (n.s.)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.9,
                       label="Non-responder ↑ (p < 0.05)"),
        mpatches.Patch(color=COLORS["control"], alpha=0.35,
                       label="Non-responder ↑ (n.s.)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# Panel C — GSEA enrichment bar chart (Melanoma)
def panel_C(ax, data: dict):
    """GSEA pathway enrichment bar chart: top pathways by NES, balanced up/down."""
    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_B_signature_waterfall(ax, data)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col = cols["nes"], cols["fdr"], cols["term"]

    if nes_col is None:
        _panel_B_signature_waterfall(ax, data)
        return

    # Convert to numeric
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Pathway selection is global (top by NES); no thematic keyword filter.

    # Balanced selection: top N up + top N down by |NES|
    n_show = 15
    df_pos = df[df[nes_col] > 0].nlargest(n_show // 2 + 1, nes_col)
    df_neg = df[df[nes_col] < 0].nsmallest(n_show - len(df_pos), nes_col)
    # If one direction is sparse, fill from the other
    if len(df_pos) + len(df_neg) < n_show:
        remainder = n_show - len(df_pos) - len(df_neg)
        already = set(df_pos.index) | set(df_neg.index)
        extra = (
            df[~df.index.isin(already)]
            .assign(_abs=df[nes_col].abs())
            .nlargest(remainder, "_abs")
        )
        df_selected = pd.concat([df_pos, df_neg, extra.drop(columns="_abs")])
    else:
        df_selected = pd.concat([df_pos, df_neg])
    df_selected = df_selected.drop_duplicates().sort_values(nes_col, ascending=True)

    # Fix #4: Clean pathway names AND disambiguate duplicates
    df_selected["pathway"] = df_selected[term_col].apply(_clean_pathway_name)
    # Disambiguate duplicate display names
    _seen = {}
    new_labels = []
    for idx, row in df_selected.iterrows():
        label = row["pathway"]
        if label in _seen:
            _seen[label] += 1
            lib = str(row.get("library", ""))
            if lib and lib != "averaged":
                label = f"{label} [{lib[:8]}]"
            else:
                label = f"{label} ({_seen[label]})"
        else:
            _seen[label] = 1
        new_labels.append(label)
    df_selected["pathway"] = new_labels

    # Color by direction and significance — use project palette
    # treated (blue) = Responder ↑, control (orange) = Non-responder ↑
    clr_up_sig = COLORS["treated"]
    clr_up_ns = COLORS["treated"] + "66"  # 40% alpha hex
    clr_dn_sig = COLORS["control"]
    clr_dn_ns = COLORS["control"] + "66"
    colors = []
    for _, row in df_selected.iterrows():
        sig = (
            fdr_col is not None
            and pd.notna(row.get(fdr_col))
            and row[fdr_col] < 0.25
        )
        if row[nes_col] > 0:
            colors.append(clr_up_sig if sig else clr_up_ns)
        else:
            colors.append(clr_dn_sig if sig else clr_dn_ns)

    y_pos = np.arange(len(df_selected))
    ax.barh(y_pos, df_selected[nes_col].values, color=colors, alpha=0.9,
            edgecolor="white", linewidth=0.5, height=0.7)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_selected["pathway"].values, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title("Pathway Enrichment", fontsize=11, fontweight="bold")

    # Build legend only for categories present
    def _is_sig(row):
        return (fdr_col and pd.notna(row.get(fdr_col))
                and row[fdr_col] < 0.25)

    has_up_sig = any(
        row[nes_col] > 0 and _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    has_up_ns = any(
        row[nes_col] > 0 and not _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    has_down_sig = any(
        row[nes_col] < 0 and _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    has_down_ns = any(
        row[nes_col] < 0 and not _is_sig(row)
        for _, row in df_selected.iterrows()
    )
    legend_handles = []
    if has_up_sig:
        legend_handles.append(mpatches.Patch(
            color=COLORS["treated"], alpha=0.9,
            label="Responder ↑ (FDR < 0.25)",
        ))
    if has_up_ns:
        legend_handles.append(mpatches.Patch(
            color=COLORS["treated"], alpha=0.4,
            label="Responder ↑ (n.s.)",
        ))
    if has_down_sig:
        legend_handles.append(mpatches.Patch(
            color=COLORS["control"], alpha=0.9,
            label="Non-responder ↑ (FDR < 0.25)",
        ))
    if has_down_ns:
        legend_handles.append(mpatches.Patch(
            color=COLORS["control"], alpha=0.4,
            label="Non-responder ↑ (n.s.)",
        ))
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
                  frameon=True, framealpha=0.9)
    despine(ax)


# Panel D — Leading-edge gene overlap heatmap (Melanoma)
def panel_D(ax, data: dict, *, composite: bool = False):
    """Leading-edge gene overlap heatmap across top enriched pathways.

    Information-dense design:
    - Tight imshow grid coloured by NES direction
    - Pathway labels coloured by NES (blue=Responder↑, orange=Non-responder↑)
    - Top marginal bar showing gene recurrence count
    - Hierarchical column clustering for gene co-occurrence
    - Capped to 8 pathways × 20 genes for readability

    When *composite* is True, the marginal bar and tight_layout are
    skipped so the panel can be embedded in a composite GridSpec figure.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    gsea_results = data["gsea_results"]

    if gsea_results is None or len(gsea_results) == 0:
        _panel_C_did_summary(ax, data)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col, lead_col = (
        cols["nes"], cols["fdr"], cols["term"], cols["lead"]
    )

    if nes_col is None or lead_col is None or lead_col not in df.columns:
        _panel_C_did_summary(ax, data)
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    # Select top pathways balanced across NES directions — match count in
    # panel A (GSEA bar chart) so every bar in A has a row in this heatmap.
    MAX_PW = 15
    work_df = df.assign(_abs=df[nes_col].abs())
    pos_df = work_df[work_df[nes_col] > 0].nlargest(MAX_PW, "_abs")
    neg_df = work_df[work_df[nes_col] < 0].nlargest(MAX_PW, "_abs")

    n_pos = min(len(pos_df), MAX_PW // 2)
    n_neg = min(len(neg_df), MAX_PW // 2)
    # Fill remaining slots from whichever side has more
    remaining = MAX_PW - n_pos - n_neg
    if remaining > 0:
        if len(pos_df) > n_pos:
            extra_pos = min(remaining, len(pos_df) - n_pos)
            n_pos += extra_pos
            remaining -= extra_pos
        if remaining > 0 and len(neg_df) > n_neg:
            n_neg += min(remaining, len(neg_df) - n_neg)

    selected = pd.concat([
        pos_df.head(n_pos),
        neg_df.head(n_neg),
    ]).drop(columns="_abs", errors="ignore")
    selected = selected.sort_values(nes_col, ascending=True)

    # Parse leading-edge genes
    pathway_genes: dict[str, set[str]] = {}
    pathway_nes: dict[str, float] = {}
    pathway_fdr: dict[str, float] = {}
    all_genes: set[str] = set()
    _seen_names: set[str] = set()
    for _, row in selected.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=32)
        if pname in _seen_names:
            lib = str(row.get("library", ""))
            pname = f"{pname} [{lib[:8]}]" if lib else f"{pname} (2)"
        _seen_names.add(pname)
        genes_str = str(row[lead_col])
        genes = [g.strip() for g in genes_str.replace(";", ",").split(",")
                 if g.strip()]
        genes = [g for g in genes if _is_likely_protein_coding(g)]
        pathway_genes[pname] = set(genes)
        pathway_nes[pname] = float(row[nes_col])
        if fdr_col is not None and pd.notna(row.get(fdr_col)):
            pathway_fdr[pname] = float(row[fdr_col])
        all_genes.update(genes)

    if not all_genes or not pathway_genes:
        _panel_C_did_summary(ax, data)
        return

    # Select informative genes — guarantee BOTH NES directions are
    # represented by selecting top genes PER direction then merging.
    # This avoids the problem where one direction's highly-overlapping
    # gene sets dominate a global top-N selection.
    pathways = list(pathway_genes.keys())
    pos_pathways = [p for p in pathways if pathway_nes.get(p, 0) > 0]
    neg_pathways = [p for p in pathways if pathway_nes.get(p, 0) <= 0]

    def _count_genes_in_group(pw_list):
        """Count gene occurrences within a group of pathways."""
        counts: dict[str, int] = {}
        for pw in pw_list:
            for g in pathway_genes.get(pw, set()):
                counts[g] = counts.get(g, 0) + 1
        return counts

    TOTAL_GENES = 20
    half = TOTAL_GENES // 2
    pos_counts = _count_genes_in_group(pos_pathways)
    neg_counts = _count_genes_in_group(neg_pathways)

    # Take top genes from each direction
    pos_genes = sorted(pos_counts.keys(),
                       key=lambda g: -pos_counts[g])[:half]
    neg_genes = sorted(neg_counts.keys(),
                       key=lambda g: -neg_counts[g])[:half]

    # Merge, removing duplicates (keep order)
    seen: set[str] = set()
    shared_genes: list[str] = []
    for g in pos_genes + neg_genes:
        if g not in seen:
            shared_genes.append(g)
            seen.add(g)

    # If one direction had fewer than half genes, fill from the other
    if len(shared_genes) < TOTAL_GENES:
        all_counts: dict[str, int] = {}
        for pw in pathways:
            for g in pathway_genes.get(pw, set()):
                all_counts[g] = all_counts.get(g, 0) + 1
        for g in sorted(all_counts.keys(), key=lambda g: -all_counts[g]):
            if g not in seen:
                shared_genes.append(g)
                seen.add(g)
            if len(shared_genes) >= TOTAL_GENES:
                break

    print(f"  Panel C: {len(pos_genes)} genes from NES>0 pathways, "
          f"{len(neg_genes)} genes from NES≤0 pathways")

    # Build binary matrix and prune zero rows/cols
    matrix = np.zeros((len(pathways), len(shared_genes)), dtype=int)
    for i, pw in enumerate(pathways):
        for j, g in enumerate(shared_genes):
            if g in pathway_genes[pw]:
                matrix[i, j] = 1
    # Prune zero rows (pathways with no genes in selection)
    row_ok = matrix.sum(axis=1) > 0
    matrix = matrix[row_ok]
    pathways = [p for p, k in zip(pathways, row_ok) if k]
    # Prune zero cols
    col_ok = matrix.sum(axis=0) > 0
    matrix = matrix[:, col_ok]
    shared_genes = [g for g, k in zip(shared_genes, col_ok) if k]

    n_pw_kept = sum(1 for p in pathways if pathway_nes.get(p, 0) > 0)
    n_neg_kept = sum(1 for p in pathways if pathway_nes.get(p, 0) <= 0)
    print(f"  Panel C: {n_pw_kept} NES>0 + {n_neg_kept} NES≤0 pathways "
          f"retained, {len(shared_genes)} genes")

    if matrix.size == 0 or not shared_genes:
        _panel_C_did_summary(ax, data)
        return

    n_pw, n_genes = matrix.shape

    # ── Hierarchical clustering of gene columns ──
    if n_genes >= 3:
        try:
            dist = pdist(matrix.T, metric="jaccard")
            dist = np.nan_to_num(dist, nan=1.0)
            Z = linkage(dist, method="average")
            gene_order = leaves_list(Z)
        except Exception:
            gene_order = np.arange(n_genes)
    else:
        gene_order = np.arange(n_genes)

    matrix = matrix[:, gene_order]
    shared_genes = [shared_genes[i] for i in gene_order]

    # Recompute column counts after clustering
    col_counts = matrix.sum(axis=0)

    # ── Sort pathways: NES>0 block on top, NES<0 on bottom ──
    pos_pws = [p for p in pathways if pathway_nes.get(p, 0) > 0]
    neg_pws = [p for p in pathways if pathway_nes.get(p, 0) <= 0]
    # Sort within each block by |NES|
    pos_pws.sort(key=lambda p: pathway_nes.get(p, 0))
    neg_pws.sort(key=lambda p: pathway_nes.get(p, 0))
    pathways_sorted = neg_pws + pos_pws
    row_idx = [pathways.index(p) for p in pathways_sorted]
    matrix = matrix[row_idx]
    pathways = pathways_sorted
    n_sep = len(neg_pws)  # separator position between blocks

    # Recompute column counts after reorder
    col_counts = matrix.sum(axis=0)

    # ── Colour constants ──
    BLUE = (0.122, 0.471, 0.706)   # steel blue (Responder ↑ / NES>0)
    ORANGE = (0.878, 0.478, 0.184)  # warm orange (Non-responder ↑ / NES<0)
    EMPTY_COLOR = (0.94, 0.94, 0.94)    # light gray for "not in leading edge"  # noqa: N806

    # ── Colour matrix: in leading edge (filled) vs not (empty) ──
    rgb = np.full((n_pw, n_genes, 3), 0.94)  # light gray for empty
    for i, pw in enumerate(pathways):
        nes_val = pathway_nes.get(pw, 0)
        fill = np.array(BLUE if nes_val > 0 else ORANGE)
        for j in range(n_genes):
            if matrix[i, j] == 1:
                rgb[i, j] = fill

    # ── Transpose: genes on Y-axis, pathways on X-axis ──
    rgb = np.transpose(rgb, (1, 0, 2))  # (n_genes, n_pw, 3)

    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower")

    # Thin white grid lines
    for i in range(n_genes + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
    for j in range(n_pw + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8, zorder=2)

    # Separator line between NES<0 and NES>0 blocks (now vertical)
    if n_sep > 0 and n_sep < n_pw:
        ax.axvline(n_sep - 0.5, color="black", linewidth=1.5, zorder=3)

    # X-axis: pathway labels, coloured by NES direction
    ax.set_xticks(range(n_pw))
    ax.set_xticklabels(pathways, rotation=35, ha="right", fontsize=5)
    for i, (pw, label) in enumerate(zip(pathways, ax.get_xticklabels())):
        label.set_color(BLUE if pathway_nes.get(pw, 0) > 0 else ORANGE)
        label.set_fontweight("bold")

    # Y-axis: gene labels
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(shared_genes, fontsize=6, style="italic")
    ax.tick_params(axis="both", length=0)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # Gene recurrence: how many pathways each gene appears in
    # col_counts was computed from original (n_pw × n_genes) matrix
    gene_counts = col_counts

    if not composite:
        # ── Right marginal bar: gene recurrence count ──
        fig = ax.get_figure()
        fig.tight_layout(rect=[0, 0, 0.90, 1])
        ax_pos = ax.get_position()
        bar_width = 0.04
        bar_ax = fig.add_axes([
            ax_pos.x1 + 0.02, ax_pos.y0,
            bar_width, ax_pos.height,
        ])
        bar_colors = ["#555555"] * n_genes
        bar_ax.barh(range(n_genes), gene_counts, height=0.7,
                    color=bar_colors, edgecolor="none")
        bar_ax.set_ylim(-0.5, n_genes - 0.5)
        bar_ax.set_xlim(0, max(gene_counts) + 0.5)
        bar_ax.set_yticks([])
        bar_ax.set_xlabel("# paths", fontsize=5.5, labelpad=5)
        bar_ax.tick_params(axis="x", labelsize=5.5, length=2)
        bar_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=3))
        for spine in ["top", "right", "left"]:
            bar_ax.spines[spine].set_visible(False)
        bar_ax.spines["bottom"].set_linewidth(0.5)
        ax.set_title("Leading-Edge Gene Overlap", fontsize=11,
                     fontweight="bold", pad=8)
    else:
        ax.set_title("Leading-Edge Gene Overlap", fontsize=11,
                     fontweight="bold")

    # Legend — inside the heatmap lower-right (gray empty region)
    legend_handles = [
        mpatches.Patch(facecolor=BLUE, label="Resp. ↑"),
        mpatches.Patch(facecolor=ORANGE, label="Non-resp. ↑"),
        mpatches.Patch(facecolor=EMPTY_COLOR, edgecolor="#CCCCCC",
                       label="Not in leading edge"),
    ]
    ax.legend(
        handles=legend_handles, fontsize=7, loc="lower right",
        frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
        handlelength=1.0, handleheight=0.7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


# Panel E — Cell-type DiD effect heatmap (Melanoma)
def panel_E(ax, data: dict):
    """Cell-type-resolved effect heatmap for top DiD genes.

    Rows = top genes by |β_DiD|, columns = cell types.
    Color = mean DiD-like effect per cell type (responder post-pre
    minus non-responder post-pre, using raw cell-level means).
    """
    gene_results = data.get("gene_results")
    adata = data.get("adata")

    if gene_results is None or adata is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.set_title("Cell-Type DiD Effects", fontsize=11,
                     fontweight="bold")
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"
    df = df.dropna(subset=[beta_col])

    # Restrict to protein-coding genes (skip RNU*, RNA5SP*, lncRNAs, etc.)
    df = df[df["feature"].apply(_is_likely_protein_coding)]

    # Select top 15 genes by |β_DiD| (balanced: top 8 pos + top 7 neg)
    n_per_dir = 8
    df_pos = df[df[beta_col] > 0].nlargest(n_per_dir, beta_col)
    df_neg = df[df[beta_col] < 0].nsmallest(n_per_dir - 1, beta_col)
    top_genes_df = pd.concat([df_pos, df_neg])
    top_genes = top_genes_df["feature"].tolist()

    # Restrict to genes present in adata
    available = [g for g in top_genes if g in adata.var_names]
    if len(available) == 0:
        ax.text(0.5, 0.5, "No top genes in adata",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Compute cell-type × gene DiD-like effects from raw cell-level data
    ct_col = "cell_type"
    if ct_col not in adata.obs.columns:
        ct_col = next(
            (c for c in adata.obs.columns if "cell" in c.lower()
             and "type" in c.lower()),
            None,
        )
    if ct_col is None:
        ax.text(0.5, 0.5, "No cell type column",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    layer_key = "log1p_tpm"
    cell_types = sorted(adata.obs[ct_col].dropna().unique())
    # Drop very rare cell types (<20 cells total)
    ct_counts = adata.obs[ct_col].value_counts()
    cell_types = [ct for ct in cell_types if ct_counts.get(ct, 0) >= 20]
    cell_types = [ct for ct in cell_types if "unassign" not in ct.lower()]

    effect_mat = pd.DataFrame(
        np.nan, index=available, columns=cell_types
    )

    sub_adata = adata[:, available].copy()
    if layer_key in sub_adata.layers:
        X = sub_adata.layers[layer_key]
    else:
        X = sub_adata.X

    obs = sub_adata.obs.copy()
    obs["_visit"] = obs["visit"]
    obs["_arm"] = obs["response_harmonized"]
    obs["_ct"] = obs[ct_col]

    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.toarray()
    expr_df = pd.DataFrame(X, index=obs.index, columns=available)
    expr_df["_visit"] = obs["_visit"].values
    expr_df["_arm"] = obs["_arm"].values
    expr_df["_ct"] = obs["_ct"].values

    for ct in cell_types:
        ct_mask = expr_df["_ct"] == ct
        ct_data = expr_df[ct_mask]
        for arm_label, visit_label in [
            ("Responder", "Pre"), ("Responder", "Post"),
            ("Non-responder", "Pre"), ("Non-responder", "Post"),
        ]:
            pass  # just checking groups exist

        for gene in available:
            try:
                means = {}
                for arm in ["Responder", "Non-responder"]:
                    for vis in ["Pre", "Post"]:
                        mask = (ct_data["_arm"] == arm) & (ct_data["_visit"] == vis)
                        vals = ct_data.loc[mask, gene]
                        means[(arm, vis)] = vals.mean() if len(vals) > 0 else np.nan

                # DiD = (R_post - R_pre) - (NR_post - NR_pre)
                r_delta = means[("Responder", "Post")] - means[("Responder", "Pre")]
                nr_delta = means[("Non-responder", "Post")] - means[("Non-responder", "Pre")]
                did_val = r_delta - nr_delta
                if np.isfinite(did_val):
                    effect_mat.loc[gene, ct] = did_val
            except Exception:
                pass

    # Sort genes by global β_DiD for consistent ordering
    gene_order = top_genes_df.set_index("feature").loc[available].sort_values(
        beta_col, ascending=True
    ).index.tolist()
    effect_mat = effect_mat.loc[gene_order]

    # Drop cell types with all NaN
    effect_mat = effect_mat.dropna(axis=1, how="all")

    if effect_mat.shape[1] == 0:
        ax.text(0.5, 0.5, "Insufficient cell-type data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COLORS["gray"])
        ax.axis("off")
        return

    # Plot heatmap
    vmax = np.nanmax(np.abs(effect_mat.values))
    vmax = max(vmax, 0.01)  # avoid degenerate scale

    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "did_div",
        [COLORS["control"], "#f0f0f0", COLORS["treated"]],
        N=256,
    )

    im = ax.imshow(
        effect_mat.values.astype(float),
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # Mask NaN cells with hatching
    nan_mask = np.isnan(effect_mat.values.astype(float))
    if nan_mask.any():
        masked = np.ma.array(np.ones_like(effect_mat.values, dtype=float),
                             mask=~nan_mask)
        ax.pcolormesh(
            np.arange(effect_mat.shape[1] + 1) - 0.5,
            np.arange(effect_mat.shape[0] + 1) - 0.5,
            masked,
            cmap=mcolors.ListedColormap(["#e8e8e8"]),
            vmin=0, vmax=1, zorder=0,
        )

    ax.set_xticks(np.arange(effect_mat.shape[1]))
    ax.set_xticklabels(effect_mat.columns, rotation=30, ha="right",
                       fontsize=6.5)
    ax.set_yticks(np.arange(effect_mat.shape[0]))
    ax.set_yticklabels(effect_mat.index, fontsize=7)
    ax.set_title("Cell-Type DiD Effects (Top Genes)", fontsize=11,
                 fontweight="bold")

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta\Delta$ expression", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ======================================================================
# Panel J — TNBC cell-type within-arm effect heatmap
# ======================================================================


FIGURE_NAME = "SuppFig6_cross_dataset_biology"

_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7", "CD3D", "FOXP3", "IL7R", "TOX",
]

_GENE_SETS = {
    "T cell exhaustion\n(PD-1/TIM-3/LAG-3)": ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TIGIT", "TOX", "ENTPD1"],
    "CD8+ cytotoxicity\n(granzyme/perforin)": ["GZMB", "PRF1", "GZMA", "GZMK", "NKG7", "GNLY", "FASLG"],
    "Pro-inflammatory\nactivation (IFNγ/TNF)": ["IFNG", "TNF", "IL2", "CD69", "IL2RA", "HLA-DRA"],
    "T cell identity\n(CD3/CD4/CD8)": ["CD3D", "CD3E", "CD4", "CD8A", "TCF7", "IL7R"],
}

_DATASET_CFG = {
    "Melanoma": {
        "design": "two_arm",
        "loader": get_sade_feldman,
        "harmonize": True,
        "layer": "log1p_tpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_treated": "Responder",
        "arm_control": "Non-responder",
        "visits": ("Pre", "Post"),
    },
    "AML": {
        "design": "single_arm",
        "loader": lambda: get_aml(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "Treatment",
        "visits": ("Pre", "Post"),
    },
    "CAR-T": {
        "design": "single_arm",
        "loader": lambda: get_cart(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        # Single-arm: every patient received CAR-T, so there is no arm to
        # filter on. (response now holds LtR/R/NR/Unknown from the loader, so the
        # old arm_filter="CAR-T" selected zero cells.)
        "arm_col": None,
        "arm_filter": None,
        "visits": ("Pre", "Post"),
    },
    "COVID-19": {
        "design": "two_arm",
        "loader": get_stephenson,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "Collection_Day",
        "arm_col": "severity",
        "arm_treated": "Severe",
        "arm_control": "Mild",
        "visits": ("D0", "D28"),
    },
    "Vaccine": {
        "design": "single_arm",
        "loader": get_vaccine,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": None,
        "visits": ("Pre", "Post"),
    },
    "TNBC": {
        "design": "two_arm",
        "loader": get_tnbc_zhang,
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "arm",
        "arm_treated": "anti-PDL1+Chemo",
        "arm_control": "Chemo",
        "visits": ("Pre", "Post"),
    },
}

_DS_PALETTE = dict(zip(_DATASET_CFG.keys(),
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]))

# Design-type label for legend annotations: DiD = two-arm difference-in-differences,
# Δ = single-arm pre/post change.
_DESIGN_LABEL: dict[str, str] = {
    "Melanoma": "DiD",
    "COVID-19": "DiD",
    "AML": "Δ",
    "CAR-T": "Δ",
    "Vaccine": "Δ",
    "TNBC": "DiD",
}


def _ds_label(name: str) -> str:
    """Return dataset name with design-type suffix for legends."""
    tag = _DESIGN_LABEL.get(name, "")
    return f"{name} ({tag})" if tag else name


def _to_array(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _score_gene_sets(adata, layer: str) -> dict[str, np.ndarray]:
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = _to_array(X)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    out = {}
    for gs_name, genes in _GENE_SETS.items():
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idx) < 2:
            continue
        vals = X[:, idx]
        z = (vals - np.nanmean(vals, axis=0)) / (np.nanstd(vals, axis=0) + 1e-8)
        out[gs_name] = np.nanmean(z, axis=1)
    return out


def _participant_delta(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    """Compute per-participant pre→post deltas.

    For single-arm datasets with ``arm_filter``, subset to that arm first.
    Returns a DataFrame with columns = features + ["participant_id", "arm"].

    Deltas are indexed by (participant, arm) to prevent arm misassignment
    when participant IDs are not strictly unique per arm stratum.
    """
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg.get("arm_col")
    arm_filter = cfg.get("arm_filter")
    pre_v, post_v = cfg["visits"]

    required = [pid_col, visit_col]
    if arm_col and arm_col in adata.obs.columns:
        required.append(arm_col)
    if not all(c in adata.obs.columns for c in required):
        return None

    if not features:
        return None

    # Single-arm: filter to treatment arm before computing deltas
    ad = adata
    if arm_filter and arm_col and arm_col in adata.obs.columns:
        ad = adata[adata.obs[arm_col] == arm_filter]

    X = _to_array(ad[:, features].layers[cfg["layer"]] if cfg["layer"] in ad.layers else ad[:, features].X)
    df = pd.DataFrame(X, columns=features, index=ad.obs_names)
    df[pid_col] = ad.obs[pid_col].values
    df[visit_col] = ad.obs[visit_col].values
    if arm_col and arm_col in ad.obs.columns:
        df["arm"] = ad.obs[arm_col].values
    else:
        df["arm"] = "All"

    # Group by (participant, visit, arm) to get unique pseudobulk per stratum
    pv = (
        df.groupby([pid_col, visit_col, "arm"], observed=True)[features]
        .mean()
        .reset_index()
    )
    pv = pv[pv[visit_col].isin([pre_v, post_v])].copy()

    # Index by (participant, arm) to ensure correct arm pairing
    pre = pv[pv[visit_col] == pre_v].set_index([pid_col, "arm"])
    post = pv[pv[visit_col] == post_v].set_index([pid_col, "arm"])
    common = pre.index.intersection(post.index)
    if len(common) < 3:
        return None

    delta = post.loc[common, features] - pre.loc[common, features]
    delta = delta.reset_index().rename(columns={pid_col: "participant_id"})
    return delta


def _participant_visit_gs(adata, cfg: dict, gs_scores: dict[str, np.ndarray]) -> pd.DataFrame | None:
    if not gs_scores:
        return None
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg["arm_col"]

    required = [pid_col, visit_col]
    if arm_col:
        required.append(arm_col)
    if not all(c in adata.obs.columns for c in required):
        return None

    cols = [pid_col, visit_col]
    if arm_col and arm_col in adata.obs.columns:
        cols.append(arm_col)
    obs = adata.obs[cols].copy()
    col_map = {pid_col: "participant_id", visit_col: "visit"}
    if arm_col:
        col_map[arm_col] = "arm"
    obs = obs.rename(columns=col_map)
    if "arm" not in obs.columns:
        obs["arm"] = "All"
    for k, v in gs_scores.items():
        obs[k] = v

    pv = obs.groupby(["participant_id", "visit", "arm"], observed=True).mean().reset_index()
    return pv


def _effect_vector(delta: pd.DataFrame, cfg: dict, features: list[str]) -> pd.Series:
    """Mean effect per feature.

    Two-arm: treated mean Δ − control mean Δ  (difference-in-differences).
    Single-arm: mean Δ across all participants in the treated arm.
    """
    if delta is None or delta.empty:
        return pd.Series(dtype=float)

    if cfg.get("design") == "two_arm":
        treated = cfg["arm_treated"]
        control = cfg["arm_control"]
        t = delta[delta["arm"] == treated][features].mean(axis=0)
        c = delta[delta["arm"] == control][features].mean(axis=0)
        return t - c

    # Single-arm: mean pre→post delta
    return delta[features].mean(axis=0)


def _load_all() -> dict[str, dict]:
    out = {}
    for name, cfg in _DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)

            layer = cfg["layer"]
            if layer == "log1p_cpm" and "log1p_cpm" not in adata.layers:
                if "counts" in adata.layers:
                    adata = add_log1p_cpm_layer(
                        adata, counts_layer="counts", out_layer="log1p_cpm",
                    )

            features = [f for f in _FEATURES if f in adata.var_names]
            gs_scores = _score_gene_sets(adata, cfg["layer"])
            delta = _participant_delta(adata, cfg, features)
            gs_pv = _participant_visit_gs(adata, cfg, gs_scores)
            effect = _effect_vector(delta, cfg, features) if delta is not None else pd.Series(dtype=float)

            out[name] = {
                "adata": adata,
                "cfg": cfg,
                "features": features,
                "gs_scores": gs_scores,
                "gs_pv": gs_pv,
                "delta": delta,
                "effect": effect,
            }
            n_part = 0 if delta is None else delta["participant_id"].nunique()
            print(f"  {name}: {adata.n_obs} cells, {n_part} paired participants")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return out


def _effect_matrix(data: dict[str, dict]) -> pd.DataFrame:
    cols = {}
    for name, ds in data.items():
        eff = ds.get("effect", pd.Series(dtype=float))
        if isinstance(eff, pd.Series) and len(eff) > 0:
            cols[name] = eff
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols)


def _panel_gs_distributions(ax, data: dict[str, dict]):
    rows = []
    for name, ds in data.items():
        pv = ds.get("gs_pv")
        if pv is None or pv.empty:
            continue
        gs_cols = [g for g in _GENE_SETS if g in pv.columns]
        if not gs_cols:
            continue
        # Participant-level means across visits to avoid cell count imbalance.
        per_pid = pv.groupby("participant_id", observed=True)[gs_cols].mean().reset_index()
        for gs in gs_cols:
            rows.extend({"Dataset": name, "Gene set": gs, "Score": float(v)} for v in per_pid[gs].values)

    if not rows:
        ax.text(0.5, 0.5, "No gene-set data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    # Map dataset names to design-labelled versions for legend
    df["Dataset"] = df["Dataset"].map(_ds_label)
    palette = {_ds_label(k): v for k, v in _DS_PALETTE.items()}
    sns.violinplot(data=df, x="Gene set", y="Score", hue="Dataset", palette=palette,
                   cut=0, linewidth=0.2, ax=ax)
    ax.set_title("Gene-set score distributions (within-dataset z-score)", fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.25)
    ax.legend(fontsize=4, frameon=True, ncol=2, loc="upper right")
    despine(ax)


def _panel_pairwise_corr(ax, data: dict[str, dict]):
    mat = _effect_matrix(data)
    if mat.empty or mat.shape[1] < 2:
        ax.text(0.5, 0.5, "Need >=2 datasets", ha="center", va="center", transform=ax.transAxes)
        return

    # Rename columns/index to include design-type label
    mat = mat.rename(columns=_ds_label)
    corr = pd.DataFrame(index=mat.columns, columns=mat.columns, dtype=float)
    for a in mat.columns:
        for b in mat.columns:
            if a == b:
                corr.loc[a, b] = 1.0
                continue
            common = mat[[a, b]].dropna()
            if len(common) < 3:
                corr.loc[a, b] = np.nan
            else:
                # Spearman rank correlation — more appropriate when comparing
                # effect estimates from different statistical frameworks
                # (DiD beta vs Hedges' g vs paired delta)
                rho, _ = sp_stats.spearmanr(common.iloc[:, 0], common.iloc[:, 1])
                corr.loc[a, b] = rho

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": r"Spearman $\rho$"}, ax=ax)
    ax.set_title("All-pairs cross-dataset effect correlation", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)


def _panel_concordant_top_genes(ax, data: dict[str, dict]):
    mat = _effect_matrix(data)
    if mat.empty:
        ax.text(0.5, 0.5, "No effects", ha="center", va="center", transform=ax.transAxes)
        return

    # Minimum |effect| threshold to count toward concordance — prevents
    # near-zero noisy estimates from inflating the concordance fraction.
    abs_thresh = 0.05

    stats_df = []
    for feat, row in mat.iterrows():
        vals = row.dropna().values
        if len(vals) < 2:
            continue
        # Only count effects above threshold for concordance
        sig = vals[np.abs(vals) > abs_thresh]
        if len(sig) < 2:
            continue
        pos = int(np.sum(sig > 0))
        neg = int(np.sum(sig < 0))
        concord = max(pos, neg) / len(sig)
        stats_df.append({
            "feature": feat,
            "mean_abs": float(np.mean(np.abs(vals))),
            "concordance": float(concord),
            "n_datasets": int(len(sig)),
            "direction": "up" if pos >= neg else "down",
        })

    if not stats_df:
        ax.text(0.5, 0.5, "No concordance data", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(stats_df).sort_values(["concordance", "mean_abs"], ascending=False).head(12)
    df = df.sort_values("mean_abs", ascending=True)
    colors = [COLORS["treated"] if d == "up" else COLORS["control"] for d in df["direction"]]

    ax.barh(df["feature"], df["mean_abs"], color=colors, alpha=0.85)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["mean_abs"] + 0.005, i, f"{r['concordance']:.0%}", va="center", fontsize=4)
    ax.set_xlabel("Mean |effect| across datasets", labelpad=1)
    ax.set_title("Shared top genes with concordant direction (|β|>0.05)", fontweight="bold")
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Upregulated"),
        mpatches.Patch(facecolor=COLORS["control"], label="Downregulated"),
    ]
    ax.legend(handles=handles, fontsize=4, frameon=True)
    despine(ax)


def _panel_gene_dist(ax, data: dict[str, dict]):
    rows = []
    for name, ds in data.items():
        eff = ds.get("effect")
        if eff is None or len(eff) == 0:
            continue
        for v in eff.dropna().values:
            rows.append({"Dataset": name, "Effect": float(v)})
    if not rows:
        ax.text(0.5, 0.5, "No effect distributions", ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.DataFrame(rows)
    df["Dataset"] = df["Dataset"].map(_ds_label)
    palette = {_ds_label(k): v for k, v in _DS_PALETTE.items()}
    sns.violinplot(data=df, x="Dataset", y="Effect", palette=palette, cut=0, inner="quartile",
                   linewidth=0.2, ax=ax)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_title("Gene-level effect distributions", fontweight="bold")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=_DS_PALETTE[n], label=_ds_label(n))
               for n in _DS_PALETTE if _ds_label(n) in df["Dataset"].values]
    ax.legend(handles=handles, fontsize=4, frameon=True, loc="lower right", ncol=2)
    despine(ax)


def _panel_exhaustion_by_celltype(ax, data: dict[str, dict]):
    """E: Exhaustion effects by cell type across datasets.

    Two-arm: effect = mean(treated Δ) − mean(control Δ),
             SE = sqrt(var_t/n_t + var_c/n_c)  (two-sample).
    Single-arm: effect = mean(Δ),
                SE = sd(Δ) / sqrt(n)  (one-sample).
    """
    ex_genes_full = ["PDCD1", "HAVCR2", "LAG3", "CTLA4", "TOX"]
    rows = []

    for ds_name, ds in data.items():
        adata = ds["adata"]
        cfg = ds["cfg"]
        ct_col = next((c for c in [
            "cell_type", "celltype", "cell_type_annot",
            "CellType", "cell_type_original", "cell_label",
            "clustnm", "leiden",
        ] if c in adata.obs.columns), None)
        ex_genes = [g for g in ex_genes_full if g in adata.var_names]
        if ct_col is None or len(ex_genes) < 2:
            continue

        # Use top 5 cell types with ≥50 cells to ensure reasonable estimates
        cts = adata.obs[ct_col].value_counts()
        top_ct = cts[cts >= 50].head(5).index.tolist()
        for ct in top_ct:
            sub = adata[adata.obs[ct_col] == ct]
            delta = _participant_delta(sub, cfg, ex_genes)
            if delta is None or len(delta) < 3:
                continue

            # Per-participant mean exhaustion score (average across genes)
            if cfg.get("design") == "two_arm":
                treated = cfg["arm_treated"]
                control = cfg["arm_control"]
                t_scores = delta[delta["arm"] == treated][ex_genes].mean(axis=1).values
                c_scores = delta[delta["arm"] == control][ex_genes].mean(axis=1).values
                if len(t_scores) < 2 or len(c_scores) < 2:
                    continue
                eff = float(np.mean(t_scores) - np.mean(c_scores))
                # Two-sample SE: sqrt(var_t/n_t + var_c/n_c)
                se = float(np.sqrt(
                    np.var(t_scores, ddof=1) / len(t_scores)
                    + np.var(c_scores, ddof=1) / len(c_scores)
                ))
            else:
                # Single-arm: one-sample mean and SE
                per_pid = delta[ex_genes].mean(axis=1).values
                eff = float(np.mean(per_pid))
                se = float(np.std(per_pid, ddof=1) / np.sqrt(len(per_pid))) if len(per_pid) > 1 else np.nan

            rows.append({"Dataset": ds_name, "Cell type": ct,
                        "Effect": float(eff),
                        "SE": float(se) if np.isfinite(se) else np.nan})

    if not rows:
        ax.text(0.5, 0.5, "No cell-type effects", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    # Color by dataset
    y = np.arange(len(df))
    colors = [_DS_PALETTE.get(d, "grey") for d in df["Dataset"]]
    ax.errorbar(df["Effect"], y, xerr=1.96 * df["SE"], fmt="none",
                ecolor="grey", capsize=2, lw=0.8, zorder=1)
    ax.scatter(df["Effect"], y, c=colors, s=12, edgecolors="white",
               linewidth=0.3, zorder=3)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Cell type']} [{r['Dataset']}]" for _, r in df.iterrows()],
                       fontsize=4)
    ax.set_xlabel("Exhaustion effect (treatment)")
    ax.set_title("T cell exhaustion effects by cell type", fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=_DS_PALETTE[n], label=_ds_label(n))
               for n in _DS_PALETTE if n in df["Dataset"].values]
    ax.legend(handles=handles, fontsize=4, loc="best", frameon=True)
    despine(ax)


def _panel_effect_heatmap(ax, data: dict[str, dict]):
    mat = _effect_matrix(data)
    if mat.empty:
        ax.text(0.5, 0.5, "No effect matrix", ha="center", va="center", transform=ax.transAxes)
        return
    # Top 15 most variable features across datasets
    vv = mat.var(axis=1, skipna=True).sort_values(ascending=False)
    top = vv.head(15).index.tolist()
    plot_df = mat.loc[top].rename(columns=_ds_label)
    sns.heatmap(plot_df, cmap="RdBu_r", center=0, linewidths=0.4, linecolor="white",
                annot=True, fmt=".2f", ax=ax,
                cbar_kws={"label": "Effect"})
    ax.set_title("Effect heatmap across datasets", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=4)


def _panel_paired_trajectories(ax, data: dict[str, dict]):
    exhaustion_col = next((k for k in _GENE_SETS if "exhaustion" in k.lower()), None)
    if exhaustion_col is None:
        ax.text(0.5, 0.5, "No exhaustion gene set", ha="center", va="center", transform=ax.transAxes)
        return

    x_tick = []
    x_tick_lab = []
    xpos = 0
    for name, ds in data.items():
        pv = ds.get("gs_pv")
        cfg = ds.get("cfg", {})
        if pv is None or pv.empty or exhaustion_col not in pv.columns:
            continue
        pre_v, post_v = cfg.get("visits", ("Pre", "Post"))
        for arm in pv["arm"].dropna().unique():
            sub = pv[pv["arm"] == arm]
            pre = sub[sub["visit"] == pre_v].set_index("participant_id")
            post = sub[sub["visit"] == post_v].set_index("participant_id")
            common = pre.index.intersection(post.index)
            if len(common) < 2:
                continue
            x0, x1 = xpos, xpos + 1
            for pid in common:
                y0 = float(pre.loc[pid, exhaustion_col])
                y1 = float(post.loc[pid, exhaustion_col])
                ax.plot([x0, x1], [y0, y1], color=_DS_PALETTE.get(name, "grey"), alpha=0.35, lw=0.8)
            med0 = np.median(pre.loc[common, exhaustion_col].values)
            med1 = np.median(post.loc[common, exhaustion_col].values)
            ax.plot([x0, x1], [med0, med1], color=_DS_PALETTE.get(name, "black"), lw=2.8, zorder=5)
            x_tick.extend([x0, x1])
            lbl = _ds_label(name)
            x_tick_lab.extend([f"{lbl}\n{arm}\n{pre_v}", f"{lbl}\n{arm}\n{post_v}"])
            xpos += 2.5

    if not x_tick:
        ax.text(0.5, 0.5, "No paired trajectory data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.set_xticks(x_tick)
    ax.set_xticklabels(x_tick_lab, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=3.5)
    ax.set_ylabel("Exhaustion score")
    ax.set_title("Participant-level paired trajectories (Exhaustion)", fontweight="bold")
    despine(ax)


def _panel_enrichment_heatmap(ax, data: dict[str, dict]):
    rows = []
    for name, ds in data.items():
        pv = ds.get("gs_pv")
        if pv is None or pv.empty:
            continue
        gs_cols = [g for g in _GENE_SETS if g in pv.columns]
        if not gs_cols:
            continue
        for arm in sorted(pv["arm"].dropna().unique()):
            for visit in sorted(pv["visit"].dropna().unique()):
                sub = pv[(pv["arm"] == arm) & (pv["visit"] == visit)]
                if sub.empty:
                    continue
                label = f"{_ds_label(name)}\n{arm}\n{visit}"
                means = sub[gs_cols].mean(axis=0)
                for gs in gs_cols:
                    rows.append({"Gene set": gs, "Group": label, "Score": float(means[gs])})

    if not rows:
        ax.text(0.5, 0.5, "No enrichment summary", ha="center", va="center", transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    piv = df.pivot(index="Gene set", columns="Group", values="Score")
    sns.heatmap(piv, cmap="RdBu_r", center=0, linewidths=0.3, linecolor="white",
                ax=ax, cbar_kws={"label": "Mean z-score\n(within-dataset)", "pad": 0.01})
    ax.set_title("Enrichment summary (within-dataset z-score)", fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=3.5, rotation=45)
    ax.tick_params(axis="y", labelsize=4)


def generate():
    """Create and save Supplementary Figure 6 panels (A–M) + composite."""
    print("Supplementary Figure 6: Melanoma Discovery + Cross-Dataset Consistency")

    # ── Load melanoma data (panels A–E) ──────────────────────────────
    print("  Loading melanoma biological discovery data...")
    data_mel = _prepare_bio_discovery_data()

    # ── Load cross-dataset data (panels F–M) ─────────────────────────
    data = _load_all()
    if not data:
        print("  No cross-dataset data available; skipping.")
        return

    # ── Individual panels ─────────────────────────────────────────────

    # Panels A–E: melanoma biological discovery
    mel_panels = [
        ("panel_A", panel_A,      (8, 6),  dict(composite=False)),
        ("panel_B", panel_B,    (8, 6),  {}),
        ("panel_C", panel_C,    (8, 6),  {}),
        ("panel_D", panel_D, (10, 7), {}),
        ("panel_E", panel_E,  (8, 6),  {}),
    ]
    for panel_name, fn, size, kwargs in mel_panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data_mel, **kwargs)
        if panel_name != "panel_D":
            fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    # Panels F–M: cross-dataset panels (previously A–H)
    cross_panels = [
        ("panel_F", _panel_gs_distributions,    (11.0, 5.8)),
        ("panel_G", _panel_pairwise_corr,        (6.8, 6.0)),
        ("panel_H", _panel_concordant_top_genes, (8.8, 5.8)),
        ("panel_I", _panel_gene_dist,            (7.2, 5.8)),
        ("panel_J", _panel_exhaustion_by_celltype,(7.8, 6.8)),
        ("panel_K", _panel_effect_heatmap,       (8.8, 6.0)),
        ("panel_L", _panel_paired_trajectories,  (12.0, 6.2)),
        ("panel_M", _panel_enrichment_heatmap,   (12.0, 6.5)),
    ]
    for panel_name, fn, size in cross_panels:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    # ==================================================================
    # Composite artboard  (180 mm × 215 mm)
    # ==================================================================
    #   Row  0: A | B | C  (melanoma: volcano | waterfall | GSEA)
    #   Row  2: D | E      (melanoma: leading-edge | cell-type HM)
    #   Row  4: F | G      (cross-dataset: gs-dist | pairwise corr)
    #   Row  6: H | I      (cross-dataset: concordant genes | gene dist)
    #   Row  8: J | K      (cross-dataset: exhaustion | effect heatmap)
    #   Row 10: L          (cross-dataset: paired trajectories)
    #   Row 12: M          (cross-dataset: enrichment heatmap)
    # ==================================================================
    print("  Building composite figure ...")

    _SMALL_RC = {
        "font.size": 4.5,
        "axes.titlesize": 5.0,
        "axes.labelsize": 4.5,
        "xtick.labelsize": 4.0,
        "ytick.labelsize": 4.0,
        "legend.fontsize": 3.5,
        "legend.title_fontsize": 3.5,
    }
    _MAX_FONT = 6

    def _cap_fontsize(fig_obj, maximum):
        for ax_i in fig_obj.get_axes():
            for txt in ([ax_i.title, ax_i.xaxis.label, ax_i.yaxis.label]
                        + ax_i.get_xticklabels() + ax_i.get_yticklabels()
                        + ax_i.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            leg = ax_i.get_legend()
            if leg:
                for txt in leg.get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
                t = leg.get_title()
                if t and t.get_fontsize() > maximum:
                    t.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    # 13 rows: 7 content rows interleaved with 6 spacer rows
    outer = fig_c.add_gridspec(
        13, 1,
        height_ratios=[
            0.65,   # row  0: A | B | C  — melanoma
            0.40,   # spacer
            0.50,   # row  2: D | E      — melanoma
            0.52,   # section spacer
            0.55,   # row  4: F | G
            0.39,   # spacer (slightly wider)
            0.43,   # row  6: H | I
            0.27,   # spacer
            0.85,   # row  8: J | K (taller)
            0.29,   # spacer
            0.30,   # row 10: L (full width)
            0.35,   # spacer
            0.30,   # row 12: M (full width)
        ],
        hspace=0.0,
        left=0.06, right=0.99, top=0.97, bottom=0.03,
    )

    # ── Row 0: A | B | C  (melanoma) ─────────────────────────────────
    gs0 = outer[0].subgridspec(1, 4, wspace=0.40,
                                width_ratios=[1.0, 0.85, 0.18, 0.7])
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])
    ax_c = fig_c.add_subplot(gs0[3])

    panel_A(ax_a, data_mel, composite=True)
    ax_a.tick_params(axis='y', labelsize=4)
    ax_a.xaxis.label.set_fontsize(4.5)
    ax_a.yaxis.label.set_fontsize(4.5)
    panel_B(ax_b, data_mel)
    ax_b.tick_params(axis='y', labelsize=4)
    _b_lbls = [t.get_text() for t in ax_b.get_yticklabels()]
    if _b_lbls:
        ax_b.set_yticklabels(
            [t if _k % 2 == 0 else "" for _k, t in enumerate(_b_lbls)],
            fontsize=4,
        )
    panel_C(ax_c, data_mel)
    ax_c.set_title(ax_c.get_title().replace("Melanoma", "").strip(" —–-") +
                   " — Melanoma", fontsize=5.0, fontweight="bold")
    ax_c.tick_params(axis='y', labelsize=3.5)
    ax_c.set_xticks([-2, 0, 2])

    # ── Row 2: D | E  (melanoma) ──────────────────────────────────────
    gs2 = outer[2].subgridspec(1, 4, wspace=0.30,
                                width_ratios=[0.10, 0.95, 0.25, 0.95])
    ax_d = fig_c.add_subplot(gs2[1])
    ax_e = fig_c.add_subplot(gs2[3])

    panel_D(ax_d, data_mel, composite=True)
    _swap_leading_edge_axes(ax_d)
    ax_d.set_title(ax_d.get_title().replace("Melanoma", "").strip(" —–-") +
                   " — Melanoma", fontsize=5.0, fontweight="bold")
    ax_d.tick_params(axis='y', labelsize=4)

    _axes_before_e = set(fig_c.get_axes())
    panel_E(ax_e, data_mel)
    _shrink_colorbars(fig_c, _axes_before_e, fs=3.5)
    for _cb_ax in set(fig_c.get_axes()) - _axes_before_e - {ax_e}:
        _cb_ax.tick_params(labelsize=4.0)
        _cb_ax.yaxis.label.set_fontsize(4.5)
    ax_e.set_title(ax_e.get_title().replace("Melanoma", "").strip(" —–-") +
                   " — Melanoma", fontsize=5.0, fontweight="bold")
    ax_e.tick_params(axis='x', labelsize=3.5)
    ax_e.tick_params(axis='y', labelsize=4)

    # ── Row 4: F | G ──────────────────────────────────────────────────
    gs4 = outer[4].subgridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.50)
    ax_f = fig_c.add_subplot(gs4[0])
    ax_g = fig_c.add_subplot(gs4[1])

    _panel_gs_distributions(ax_f, data)
    ax_f.tick_params(axis='y', labelsize=4)
    _axes_before_g = set(fig_c.get_axes())
    _panel_pairwise_corr(ax_g, data)
    ax_g.tick_params(axis='y', labelsize=4)
    for _cb_ax in set(fig_c.get_axes()) - _axes_before_g - {ax_g}:
        _cb_ax.tick_params(labelsize=4.0)
        _cb_ax.yaxis.label.set_fontsize(4.5)

    # ── Row 6: H | I ──────────────────────────────────────────────────
    gs6 = outer[6].subgridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.50)
    ax_h = fig_c.add_subplot(gs6[0])
    ax_i = fig_c.add_subplot(gs6[1])

    _panel_concordant_top_genes(ax_h, data)
    ax_h.tick_params(axis='y', labelsize=4)
    _panel_gene_dist(ax_i, data)
    ax_i.tick_params(axis='y', labelsize=4)
    ax_i.yaxis.label.set_fontsize(4.5)
    _leg_i = ax_i.get_legend()
    if _leg_i:
        _i_handles = _leg_i.legend_handles
        _i_labels = [t.get_text() for t in _leg_i.get_texts()]
        ax_i.legend(handles=_i_handles, labels=_i_labels,
                    fontsize=3.5, frameon=True, loc="upper left", ncol=3,
                    bbox_to_anchor=(0.0, 1.10))

    # ── Row 8: J | K ──────────────────────────────────────────────────
    gs8 = outer[8].subgridspec(1, 3, width_ratios=[0.0, 1.0, 1.2], wspace=0.45)
    ax_j = fig_c.add_subplot(gs8[1])
    ax_k = fig_c.add_subplot(gs8[2])

    _panel_exhaustion_by_celltype(ax_j, data)
    ax_j.tick_params(axis='y', labelsize=3)
    _axes_before_k = set(fig_c.get_axes())
    _panel_effect_heatmap(ax_k, data)
    for _cb_ax in set(fig_c.get_axes()) - _axes_before_k - {ax_k}:
        _cb_ax.tick_params(labelsize=4.0)
        _cb_ax.yaxis.label.set_fontsize(4.5)

    # ── Row 10: L (full width) ────────────────────────────────────────
    ax_l = fig_c.add_subplot(outer[10])
    _panel_paired_trajectories(ax_l, data)
    ax_l.tick_params(axis='y', labelsize=4)
    ax_l.yaxis.label.set_fontsize(4.5)

    # ── Row 12: M (shifted right with wider left margin) ──────────────
    gs12 = outer[12].subgridspec(1, 3, width_ratios=[0.14, 1.0, 0.0], wspace=0.0)
    ax_m = fig_c.add_subplot(gs12[1])
    _panel_enrichment_heatmap(ax_m, data)
    ax_m.xaxis.label.set_fontsize(4.5)
    ax_m.yaxis.label.set_fontsize(4.5)

    # ── Post-processing ───────────────────────────────────────────────
    for ax_pp in fig_c.get_axes():
        leg = ax_pp.get_legend()
        if leg:
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor("#CCCCCC")

    # Compact legends for melanoma panels
    _mel_bio_locs = {
        ax_a: "upper center", ax_b: "upper left",
        ax_c: "upper left",  ax_d: "lower left",
    }
    for ax_target, loc in _mel_bio_locs.items():
        _compact_legend(ax_target, loc, fs=4)

    # Shrink gene-label annotations in volcano/waterfall
    for _ax in [ax_a, ax_b]:
        for txt in _ax.texts:
            if txt.get_fontsize() > 4:
                txt.set_fontsize(max(txt.get_fontsize() * 0.50, 2.5))

    _cap_fontsize(fig_c, _MAX_FONT)

    # Normalise all axes titles to match G (rc axes.titlesize = 5.0)
    for _ax in fig_c.get_axes():
        if _ax.get_title():
            _ax.title.set_fontsize(5.0)

    # Bold panel labels A–M
    _lbl_fs = 7

    # Row 0: A B C  (A, B, C up)
    ax_a.text(-0.18, 1.10, "A", transform=ax_a.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_b.text(-0.26, 1.10, "B", transform=ax_b.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_c.text(-0.10, 1.10, "C", transform=ax_c.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # Row 2: D E  (D left and up; E up)
    ax_d.text(-0.30, 1.10, "D", transform=ax_d.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_e.text(-0.24, 1.10, "E", transform=ax_e.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # Row 4: F G  (unchanged)
    ax_f.text(-0.10, 1.10, "F", transform=ax_f.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_g.text(-0.16, 1.10, "G", transform=ax_g.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # Row 6: H I  (further up)
    ax_h.text(-0.10, 1.16, "H", transform=ax_h.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_i.text(-0.10, 1.16, "I", transform=ax_i.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # Row 8: J K  (unchanged)
    ax_j.text(-0.16, 1.10, "J", transform=ax_j.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    ax_k.text(-0.10, 1.10, "K", transform=ax_k.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # Row 10: L  (further up)
    ax_l.text(-0.05, 1.38, "L", transform=ax_l.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    # Row 12: M  (left)
    ax_m.text(-0.16, 1.16, "M", transform=ax_m.transAxes,
              fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    panel_dir = SUPP_OUTPUT / f"{FIGURE_NAME}_panels"
    panel_dir.mkdir(exist_ok=True)
    png_path = panel_dir / f"{FIGURE_NAME}.png"
    fig_c.savefig(str(png_path), format="png", dpi=600, facecolor="white")
    pdf_path = panel_dir / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", facecolor="white")
    print(f"    Saved panel: {FIGURE_NAME}")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # ── Cleanup ───────────────────────────────────────────────────────
    if "adata" in data_mel:
        del data_mel["adata"]
    data.clear()
    clear_cache()
    gc.collect()
    print("  SuppFig6 complete: 13 individual panels + combined (A–M)\n")


if __name__ == "__main__":
    apply_style()
    generate()
