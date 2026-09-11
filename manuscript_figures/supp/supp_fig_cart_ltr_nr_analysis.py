"""
Supplementary Figure — CAR-T LtR vs NR DiD Analysis
=====================================================

11-panel figure analogous to SuppFig3 (Melanoma/TNBC), using the CAR-T
dataset (GSE290722, ZUMA-1) with Long-term responders (LtR) vs Non-responders
(NR) as the two DiD arms.  R (Relapsed) patients are excluded because their
trajectory is distinct (initial response then relapse) and their response code
would be misread as "Responder" by the standard harmonizer.

Panels
------
A : Paired-participant verification (cells per participant × visit) [CAR-T].
B : Coefficient comparison: cell-level vs participant-level aggregation [CAR-T].
C : P-value comparison: −log₁₀ scale, illustrating pseudoreplication [CAR-T].
D : Leading-edge gene overlap heatmap (GSEA DiD, LtR vs NR) [CAR-T].
E : Cell-type-resolved DiD gene-expression heatmap [CAR-T].
F : Cell-type abundance DiD forest plot (LtR vs NR) [CAR-T].
G : Forest plot of DiD effects across all gene signatures [CAR-T].
H : Small-multiple interaction plots for the top 6 signatures [CAR-T].
I : Per-participant change heatmap across signatures [CAR-T].
J : Bar plot of mean Δ score (post − pre) by response group [CAR-T].
K : Cohen's d effect sizes (LtR − NR) on Δ scores [CAR-T].
"""

from __future__ import annotations

import gc
import re
import warnings
from contextlib import contextmanager

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_cart,
    load_or_run_gsea_did,
    save_panel,
    score_signatures,
    sig_display,
    verify_paired_participants,
)

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────

FIGURE_NAME = "SuppFig_CART_LtR_NR_analysis"
VISITS: tuple[str, str] = ("Pre", "Post")

_CART_LTR_ARM = "LtR"
_CART_NR_ARM = "NR"

COL_LTR = COLORS["treated"]    # blue  (LtR — Long-term responder)
COL_NR = COLORS["control"]     # orange (NR — Non-responder)
COL_GRAY = COLORS["gray"]

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_published",
    arm_treated=_CART_LTR_ARM,
    arm_control=_CART_NR_ARM,
    celltype_col="cell_type",
)

# ── Gene-name helper functions (adapted from figure4) ────────────────────

_NONCODING_PATTERN = re.compile(
    r"^(RNU\d|RNA5SP|RNY\d|RN7S[LK]|SNOR[AD]|MIR\d|LINC\d|LOC\d|"
    r"AC\d{6}|AL\d{6}|AP\d{6}|"
    r"RP\d+-|RP[SL]\d+P|CT[ABCD]-|XXbac-|KB-|LA16c-|GS\d-|"
    r"HIGD1AP|MTCO\d|RMVSL|BCRP|NAMA$|SLMO|"
    r"IGH[VDJ]|IGKV|IGLV|IGKJ|IGLJ|"
    r"TRB[VDJ]|TRA[VDJ]|TRG[VDJ]|TRD[VDJ]|"
    r"OR\d+[A-Z]|VN\d+R|"
    r"MT-|"
    r"RPS\d|RPL\d|"
    r"HCG\d|SPRR\d|"
    r"HLA-|"
    r"[A-Z]{1,2}\d{2}NC\d|"
    r"[A-Z]\d{5}\.\d)",
    re.IGNORECASE,
)
_PSEUDOGENE_SUFFIX = re.compile(r"P\d+$", re.IGNORECASE)


def _is_likely_protein_coding(gene: str) -> bool:
    if _NONCODING_PATTERN.match(gene) is not None:
        return False
    if _PSEUDOGENE_SUFFIX.search(gene) and len(gene) > 4:
        base = _PSEUDOGENE_SUFFIX.sub("", gene)
        if base and base[-1].isdigit():
            return False
    return True


def _detect_gsea_columns(df: pd.DataFrame) -> dict:
    cols = {"nes": None, "fdr": None, "term": None, "tag": None, "lead": None}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "nes":
            cols["nes"] = c
        elif cl in ("fdr q-val", "fdr"):
            cols["fdr"] = c
        elif cl == "term":
            cols["term"] = c
        elif cl.startswith("tag"):
            cols["tag"] = c
        elif cl in ("lead_genes", "leading_edge"):
            cols["lead"] = c
    if cols["nes"] is None:
        for c in df.columns:
            if c.lower() in ("nes", "normalized_enrichment_score"):
                cols["nes"] = c
                break
    if cols["fdr"] is None:
        for c in df.columns:
            if "fdr" in c.lower():
                cols["fdr"] = c
                break
    if cols["term"] is None:
        for c in df.columns:
            if c.lower() in ("name", "pathway"):
                cols["term"] = c
                break
        if cols["term"] is None:
            cols["term"] = df.columns[0]
    if cols["lead"] is None:
        for c in df.columns:
            if "lead" in c.lower() and "gene" in c.lower():
                cols["lead"] = c
                break
    return cols


def _clean_pathway_name(s: str, max_len: int = 55) -> str:
    s = str(s).replace("_", " ").title()
    s = re.sub(r"\s*\(Go:\d+\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*R-Hsa-\d+", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s[:max_len] + "…" if len(s) > max_len + 2 else s


# ── Data preparation ─────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Load CAR-T, filter to LtR/NR, score signatures, run DiD."""
    adata = get_cart()

    # Filter to LtR and NR only — exclude R (Relapsed) and Unknown
    mask = adata.obs["response_published"].isin([_CART_LTR_ARM, _CART_NR_ARM])
    adata = adata[mask].copy()

    # Restrict Post to 4wk only — the sole timepoint shared by both LtR and NR.
    # LtR patients have cells at 4wk, 6mo, and 12mo; NR patients only at 4wk.
    # Mixing timepoints into a single "Post" pseudobulk confounds the DiD
    # (different patients contribute different biological states).
    tp_mask = adata.obs["timepoint"].isin(["Leukapheresis", "4wk_post"])
    adata = adata[tp_mask].copy()

    layer = "log1p_norm" if "log1p_norm" in adata.layers else None
    adata, sig_cols = score_signatures(adata, layer=layer)

    pair_info = verify_paired_participants(
        adata.obs,
        visit_col=DESIGN.visit_col,
        visits=VISITS,
        participant_col=DESIGN.participant_col,
    )

    # Cell-level DiD (pseudoreplication demonstration for panels B–C)
    res_cell = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer=layer,
        standardize=True,
        aggregate="cell",
    )

    # Participant-level DiD with bootstrap CIs (primary results)
    did_res = did_table(
        adata,
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer=layer,
        standardize=True,
        aggregate="participant_visit",
        use_bootstrap=True,
        n_boot=999,
        seed=42,
    )

    # Pseudobulk: per-participant-visit means
    grp_cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col]
    obs_sub = adata.obs[grp_cols + sig_cols].copy()
    pb = (
        obs_sub
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    # Keep only paired participants
    visit_counts = pb.groupby(DESIGN.participant_col)[DESIGN.visit_col].nunique()
    paired_pids = visit_counts[visit_counts == 2].index
    pb = pb[pb[DESIGN.participant_col].isin(paired_pids)].copy()

    # Compute Δ = Post − Pre per participant
    pre = pb[pb[DESIGN.visit_col] == "Pre"].set_index(DESIGN.participant_col)
    post = pb[pb[DESIGN.visit_col] == "Post"].set_index(DESIGN.participant_col)
    common = pre.index.intersection(post.index)
    delta = post.loc[common, sig_cols].subtract(pre.loc[common, sig_cols])
    delta[DESIGN.arm_col] = pre.loc[common, DESIGN.arm_col]
    delta = delta.reset_index()

    n_ltr = (delta[DESIGN.arm_col] == _CART_LTR_ARM).sum()
    n_nr = (delta[DESIGN.arm_col] == _CART_NR_ARM).sum()
    print(f"  Paired participants: {len(delta)} (LtR={n_ltr}, NR={n_nr})")

    return {
        "adata": adata,
        "sig_cols": sig_cols,
        "layer": layer,
        "res_cell": res_cell,
        "pair_info": pair_info,
        "did_res": did_res,
        "pb": pb,
        "delta": delta,
    }


def _prepare_cart_bio_data(adata, layer: str | None, sig_cols: list[str]) -> dict:
    """Prepare data for panels D–F: GSEA, gene-level DiD, abundance DiD."""
    from sctrial import abundance_did

    # GSEA DiD (LtR vs NR × Pre→Post interaction)
    gsea_results = None
    did_sig = None
    try:
        did_sig = did_table(
            adata,
            features=sig_cols,
            design=DESIGN,
            visits=VISITS,
            layer=layer,
            standardize=True,
            aggregate="participant_visit",
        )
        gsea_results = load_or_run_gsea_did(
            adata, DESIGN, VISITS,
            layer=layer,
            dataset_name="CAR-T_LtRvNR",
        )
        print(f"  GSEA: {len(gsea_results) if gsea_results is not None else 0} pathways")
    except Exception as exc:
        print(f"  GSEA DiD failed: {exc}")

    # Gene-level DiD on top HVGs
    gene_results = None
    try:
        import scanpy as sc
        adata_genes = adata.copy()
        if layer is not None and layer in adata_genes.layers:
            sc.pp.highly_variable_genes(
                adata_genes, n_top_genes=2000, layer=layer, flavor="seurat",
            )
        else:
            sc.pp.highly_variable_genes(adata_genes, n_top_genes=2000, flavor="seurat")
        top_genes = adata_genes.var_names[adata_genes.var["highly_variable"]].tolist()
        print(f"  Gene-level DiD: {len(top_genes)} variable genes selected")
        gene_results = did_table(
            adata_genes,
            features=top_genes,
            design=DESIGN,
            visits=VISITS,
            layer=layer,
            standardize=True,
            aggregate="participant_visit",
        )
        gene_results = gene_results.dropna(subset=["beta_DiD", "p_DiD"])
        print(f"  Gene-level DiD: {len(gene_results)} genes with valid results")
    except Exception as exc:
        print(f"  Gene-level DiD failed: {exc}")

    # Cell-type abundance DiD
    res_abundance = None
    try:
        res_abundance = abundance_did(
            adata,
            design=DESIGN,
            visits=VISITS,
            use_bootstrap=True,
            n_boot=999,
            seed=42,
        )
        print(f"  Abundance DiD: {len(res_abundance)} cell types")
    except Exception as exc:
        print(f"  Abundance DiD failed: {exc}")

    return {
        "adata": adata,
        "did_sig": did_sig,
        "gsea_results": gsea_results,
        "gene_results": gene_results,
        "res_abundance": res_abundance,
    }


# ── Pseudobulk helpers ────────────────────────────────────────────────────

def _pseudobulk(adata, sig_col: str) -> pd.DataFrame:
    df = adata.obs[[
        DESIGN.participant_col, DESIGN.visit_col,
        DESIGN.arm_col, sig_col,
    ]].copy()
    return (
        df.groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )[sig_col]
        .mean()
        .reset_index()
    )


def _pseudobulk_all(adata, sig_cols: list[str]) -> pd.DataFrame:
    cols = [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col] + sig_cols
    df = adata.obs[cols].copy()
    return (
        df.groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )[sig_cols]
        .mean()
        .reset_index()
    )


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_sd = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_sd == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / pooled_sd
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return d * correction


# ── Panel A: Paired-participant verification ──────────────────────────────

def _panel_a_paired_verification(ax: plt.Axes, data: dict) -> None:
    adata = data["adata"]
    obs = adata.obs.copy()

    counts = (
        obs.groupby(
            [DESIGN.participant_col, DESIGN.visit_col, DESIGN.arm_col],
            observed=True,
        )
        .size()
        .reset_index(name="n_cells")
    )
    counts[DESIGN.visit_col] = pd.Categorical(
        counts[DESIGN.visit_col], categories=["Pre", "Post"], ordered=True,
    )
    counts = counts.sort_values([DESIGN.arm_col, DESIGN.participant_col, DESIGN.visit_col])

    participants = counts[DESIGN.participant_col].unique()
    pid_order = {pid: i for i, pid in enumerate(participants)}
    bar_width = 0.35

    for _, row in counts.iterrows():
        x_base = pid_order[row[DESIGN.participant_col]]
        offset = -bar_width / 2 if row[DESIGN.visit_col] == "Pre" else bar_width / 2
        color = COL_LTR if row[DESIGN.arm_col] == _CART_LTR_ARM else COL_NR
        alpha = 0.6 if row[DESIGN.visit_col] == "Pre" else 1.0
        ax.bar(x_base + offset, row["n_cells"], width=bar_width,
               color=color, alpha=alpha, edgecolor="white", linewidth=0.5)

    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(
        [f"P{i+1}" for i in range(len(participants))],
        rotation=90, ha="center", fontsize=5,
    )
    ax.set_xlabel("Participant", fontsize=9)
    ax.set_ylabel("Number of cells", fontsize=12)
    ax.set_title("Paired Participants: Cells per Visit (CAR-T)", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_LTR,
               markersize=5, markeredgewidth=0, label="LtR (Long-term resp.)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_NR,
               markersize=5, markeredgewidth=0, label="NR (Non-responder)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_GRAY,
               markersize=5, markeredgewidth=0, alpha=0.6, label="Pre"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_GRAY,
               markersize=5, markeredgewidth=0, label="Post"),
    ]
    ax.legend(handles=legend_handles, fontsize=16,
              loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=4, frameon=True, framealpha=0.9,
              handletextpad=0.3, columnspacing=0.8)

    pair_info = data["pair_info"]
    ax.text(
        0.02, 0.98,
        f"{pair_info['n_paired']}/{pair_info['n_total']} participants paired",
        transform=ax.transAxes, fontsize=5.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COL_GRAY, alpha=0.8),
    )
    despine(ax)


# ── Panel B: Coefficient comparison ──────────────────────────────────────

def _panel_b_beta_comparison(ax: plt.Axes, data: dict) -> None:
    res_cell = data["res_cell"].set_index("feature")
    res_part = data["did_res"].set_index("feature")
    common = res_cell.index.intersection(res_part.index)

    beta_cell = res_cell.loc[common, "beta_DiD"].values
    beta_part = res_part.loc[common, "beta_DiD"].values

    colors = [COL_LTR if b > 0 else COL_NR for b in beta_part]
    ax.scatter(beta_cell, beta_part, c=colors, s=20, edgecolors="white",
               linewidths=0.5, zorder=3)

    lim_lo = min(beta_cell.min(), beta_part.min(), -1.0) * 1.15
    lim_hi = max(beta_cell.max(), beta_part.max()) * 1.15
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color=COL_GRAY,
            lw=1, zorder=1, label="Identity")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.axhline(0, color=COL_GRAY, lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color=COL_GRAY, lw=0.5, ls=":", zorder=0)

    try:
        from adjustText import adjust_text
        texts = [ax.text(xv, yv, sig_display(feat), fontsize=8, alpha=0.9)
                 for feat, xv, yv in zip(common, beta_cell, beta_part)]
        adjust_text(
            texts, ax=ax, expand=(0.92, 0.95),
            arrowprops=dict(arrowstyle="-", color=COL_GRAY, lw=0.4, alpha=0.6),
            ensure_inside_axes=True,
        )
    except ImportError:
        for feat, xv, yv in zip(common, beta_cell, beta_part):
            ax.text(xv, yv, sig_display(feat), fontsize=5, alpha=0.7)

    r, p = stats.pearsonr(beta_cell, beta_part)
    ax.text(
        0.05, 0.95,
        f"r = {r:.2f}, p = {p:.1e}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COL_GRAY, alpha=0.8),
    )

    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (cell-level)", fontsize=12)
    ax.set_ylabel(r"$\beta_{\mathrm{DiD}}$ (participant-level)", fontsize=12)
    ax.set_title("Effect Size: Cell vs Participant Aggregation (CAR-T)", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=COL_LTR, label="LtR ↑ (positive effect)"),
        mpatches.Patch(facecolor=COL_NR, label="NR ↑ (negative effect)"),
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc="lower left",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: P-value inflation ────────────────────────────────────────────

def _panel_c_pvalue_inflation(ax: plt.Axes, data: dict) -> None:
    res_cell = data["res_cell"].set_index("feature")
    res_part = data["did_res"].set_index("feature")
    common = res_cell.index.intersection(res_part.index)

    df = pd.DataFrame({
        "feature": common,
        "p_cell": res_cell.loc[common, "p_DiD"].values,
        "p_part": res_part.loc[common, "p_DiD"].values,
    })
    df["nlog10_cell"] = -np.log10(df["p_cell"].clip(lower=1e-300))
    df["nlog10_part"] = -np.log10(df["p_part"].clip(lower=1e-300))
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("nlog10_cell", ascending=True).reset_index(drop=True)

    y_pos = np.arange(len(df))
    bar_h = 0.35

    ax.barh(y_pos - bar_h / 2, df["nlog10_cell"], height=bar_h,
            color=COLORS["highlight"], alpha=0.8, label="Cell-level", zorder=2)
    ax.barh(y_pos + bar_h / 2, df["nlog10_part"], height=bar_h,
            color=COL_LTR, alpha=0.8, label="Participant-level", zorder=2)

    thresh = -np.log10(0.05)
    ax.axvline(thresh, color=COL_GRAY, ls="--", lw=1, zorder=1)
    ax.text(thresh + 0.1, len(df) - 0.5, "p = 0.05", fontsize=8,
            va="bottom", color=COL_GRAY)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"], fontsize=4.5)
    ax.set_xlabel(r"$-\log_{10}(p)$")
    ax.set_title("P-value Inflation: Cell vs Participant Level (CAR-T)", fontsize=11,
                 fontweight="bold")

    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: Leading-edge gene overlap heatmap ────────────────────────────

def _panel_d_leading_edge(ax: plt.Axes, data_bio: dict, *, composite: bool = False) -> None:
    """Leading-edge gene overlap heatmap across top GSEA DiD pathways (LtR vs NR)."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    gsea_results = data_bio.get("gsea_results")

    if gsea_results is None or len(gsea_results) == 0:
        _panel_did_summary_fallback(ax, data_bio)
        return

    df = gsea_results.copy()
    cols = _detect_gsea_columns(df)
    nes_col, fdr_col, term_col, lead_col = (
        cols["nes"], cols["fdr"], cols["term"], cols["lead"]
    )

    if nes_col is None or lead_col is None or lead_col not in df.columns:
        _panel_did_summary_fallback(ax, data_bio)
        return

    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col is not None:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])

    MAX_PW = 15
    work_df = df.assign(_abs=df[nes_col].abs())
    pos_df = work_df[work_df[nes_col] > 0].nlargest(MAX_PW, "_abs")
    neg_df = work_df[work_df[nes_col] < 0].nlargest(MAX_PW, "_abs")

    n_pos = min(len(pos_df), MAX_PW // 2)
    n_neg = min(len(neg_df), MAX_PW // 2)
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

    pathway_genes: dict[str, set[str]] = {}
    pathway_nes: dict[str, float] = {}
    all_genes: set[str] = set()
    _seen_names: set[str] = set()
    for _, row in selected.iterrows():
        pname = _clean_pathway_name(str(row[term_col]), max_len=32)
        if pname in _seen_names:
            lib = str(row.get("library", ""))
            pname = f"{pname} [{lib[:8]}]" if lib else f"{pname} (2)"
        _seen_names.add(pname)
        genes_str = str(row[lead_col])
        genes = [g.strip() for g in genes_str.replace(";", ",").split(",") if g.strip()]
        genes = [g for g in genes if _is_likely_protein_coding(g)]
        pathway_genes[pname] = set(genes)
        pathway_nes[pname] = float(row[nes_col])
        all_genes.update(genes)

    if not all_genes or not pathway_genes:
        _panel_did_summary_fallback(ax, data_bio)
        return

    pathways = list(pathway_genes.keys())
    pos_pathways = [p for p in pathways if pathway_nes.get(p, 0) > 0]
    neg_pathways = [p for p in pathways if pathway_nes.get(p, 0) <= 0]

    def _count_genes_in_group(pw_list):
        counts: dict[str, int] = {}
        for pw in pw_list:
            for g in pathway_genes.get(pw, set()):
                counts[g] = counts.get(g, 0) + 1
        return counts

    TOTAL_GENES = 20
    half = TOTAL_GENES // 2
    pos_counts = _count_genes_in_group(pos_pathways)
    neg_counts = _count_genes_in_group(neg_pathways)

    pos_genes = sorted(pos_counts.keys(), key=lambda g: -pos_counts[g])[:half]
    neg_genes = sorted(neg_counts.keys(), key=lambda g: -neg_counts[g])[:half]

    seen: set[str] = set()
    shared_genes: list[str] = []
    for g in pos_genes + neg_genes:
        if g not in seen:
            shared_genes.append(g)
            seen.add(g)

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

    matrix = np.zeros((len(pathways), len(shared_genes)), dtype=int)
    for i, pw in enumerate(pathways):
        for j, g in enumerate(shared_genes):
            if g in pathway_genes[pw]:
                matrix[i, j] = 1

    row_ok = matrix.sum(axis=1) > 0
    matrix = matrix[row_ok]
    pathways = [p for p, k in zip(pathways, row_ok) if k]
    col_ok = matrix.sum(axis=0) > 0
    matrix = matrix[:, col_ok]
    shared_genes = [g for g, k in zip(shared_genes, col_ok) if k]

    if matrix.size == 0 or not shared_genes:
        _panel_did_summary_fallback(ax, data_bio)
        return

    n_pw, n_genes = matrix.shape

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
    col_counts = matrix.sum(axis=0)

    pos_pws = [p for p in pathways if pathway_nes.get(p, 0) > 0]
    neg_pws = [p for p in pathways if pathway_nes.get(p, 0) <= 0]
    pos_pws.sort(key=lambda p: pathway_nes.get(p, 0))
    neg_pws.sort(key=lambda p: pathway_nes.get(p, 0))
    pathways_sorted = neg_pws + pos_pws
    row_idx = [pathways.index(p) for p in pathways_sorted]
    matrix = matrix[row_idx]
    pathways = pathways_sorted
    n_sep = len(neg_pws)
    col_counts = matrix.sum(axis=0)

    BLUE = np.array(COL_LTR if isinstance(COL_LTR, (list, tuple)) else
                    plt.matplotlib.colors.to_rgb(COL_LTR))
    ORANGE = np.array(COL_NR if isinstance(COL_NR, (list, tuple)) else
                      plt.matplotlib.colors.to_rgb(COL_NR))
    EMPTY_COLOR = (0.94, 0.94, 0.94)

    rgb = np.full((n_pw, n_genes, 3), 0.94)
    for i, pw in enumerate(pathways):
        nes_val = pathway_nes.get(pw, 0)
        fill = BLUE if nes_val > 0 else ORANGE
        for j in range(n_genes):
            if matrix[i, j] == 1:
                rgb[i, j] = fill

    rgb = np.transpose(rgb, (1, 0, 2))
    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower")

    for i in range(n_genes + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8, zorder=2)
    for j in range(n_pw + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8, zorder=2)

    if n_sep > 0 and n_sep < n_pw:
        ax.axvline(n_sep - 0.5, color="black", linewidth=1.5, zorder=3)

    ax.set_xticks(range(n_pw))
    ax.set_xticklabels(pathways, rotation=35, ha="right", fontsize=5)
    for i, (pw, label) in enumerate(zip(pathways, ax.get_xticklabels())):
        label.set_color(
            plt.matplotlib.colors.to_hex(BLUE)
            if pathway_nes.get(pw, 0) > 0
            else plt.matplotlib.colors.to_hex(ORANGE)
        )
        label.set_fontweight("bold")

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(shared_genes, fontsize=6, style="italic")
    ax.tick_params(axis="both", length=0)

    if not composite:
        fig = ax.get_figure()
        fig.tight_layout(rect=[0, 0, 0.90, 1])
        ax_pos = ax.get_position()
        bar_ax = fig.add_axes([
            ax_pos.x1 + 0.02, ax_pos.y0, 0.04, ax_pos.height,
        ])
        bar_ax.barh(range(n_genes), col_counts, height=0.7,
                    color="#555555", edgecolor="none")
        bar_ax.set_ylim(-0.5, n_genes - 0.5)
        bar_ax.set_xlim(0, max(col_counts) + 0.5)
        bar_ax.set_yticks([])
        bar_ax.set_xlabel("# paths", fontsize=5.5, labelpad=5)
        bar_ax.tick_params(axis="x", labelsize=5.5, length=2)
        bar_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=3))
        for spine in ["top", "right", "left"]:
            bar_ax.spines[spine].set_visible(False)
        bar_ax.spines["bottom"].set_linewidth(0.5)

    ax.set_title("Leading-Edge Gene Overlap (CAR-T, LtR vs NR)", fontsize=11,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=plt.matplotlib.colors.to_hex(BLUE),
                       label=f"{_CART_LTR_ARM} ↑"),
        mpatches.Patch(facecolor=plt.matplotlib.colors.to_hex(ORANGE),
                       label=f"{_CART_NR_ARM} ↑"),
        mpatches.Patch(facecolor=EMPTY_COLOR, edgecolor="#CCCCCC",
                       label="Not in leading edge"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              handlelength=1.0, handleheight=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _panel_did_summary_fallback(ax: plt.Axes, data_bio: dict) -> None:
    """Fallback when GSEA results are unavailable: show signature DiD bars."""
    did_sig = data_bio.get("did_sig")
    if did_sig is None or len(did_sig) == 0:
        ax.text(0.5, 0.5, "GSEA / DiD data unavailable",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.axis("off")
        return
    df = did_sig.copy()
    df["display"] = df["feature"].map(sig_display)
    df = df.sort_values("beta_DiD", ascending=True)
    colors = [COL_LTR if v > 0 else COL_NR for v in df["beta_DiD"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["beta_DiD"].values, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=8)
    ax.set_xlabel(r"DiD coefficient ($\beta_{\mathrm{DiD}}$)")
    ax.set_title("DiD Signature Effects (CAR-T, LtR vs NR)", fontsize=11, fontweight="bold")
    despine(ax)


# ── Panel E: Cell-type-resolved DiD gene-expression heatmap ──────────────

def _panel_e_celltype_hm(ax: plt.Axes, data_bio: dict) -> None:
    """Gene × cell-type DiD heatmap for CAR-T (LtR vs NR)."""
    import scipy.sparse as sp

    gene_results = data_bio.get("gene_results")
    adata = data_bio.get("adata")

    if gene_results is None or adata is None or len(gene_results) == 0:
        ax.text(0.5, 0.5, "Gene-level DiD results unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COL_GRAY)
        ax.set_title("Cell-Type DiD Effects (CAR-T, LtR vs NR)", fontsize=11,
                     fontweight="bold")
        ax.axis("off")
        return

    df = gene_results.copy()
    beta_col = "beta_DiD"
    df = df.dropna(subset=[beta_col])
    df = df[df["feature"].apply(_is_likely_protein_coding)]

    n_per_dir = 8
    df_pos = df[df[beta_col] > 0].nlargest(n_per_dir, beta_col)
    df_neg = df[df[beta_col] < 0].nsmallest(n_per_dir - 1, beta_col)
    top_genes_df = pd.concat([df_pos, df_neg])
    top_genes = top_genes_df["feature"].tolist()

    available = [g for g in top_genes if g in adata.var_names]
    if len(available) == 0:
        ax.text(0.5, 0.5, "No top genes found in adata",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COL_GRAY)
        ax.axis("off")
        return

    ct_col = DESIGN.celltype_col
    if ct_col not in adata.obs.columns:
        ct_col = next(
            (c for c in adata.obs.columns
             if "cell" in c.lower() and "type" in c.lower()),
            None,
        )
    if ct_col is None:
        ax.text(0.5, 0.5, "No cell-type column in CAR-T data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COL_GRAY)
        ax.axis("off")
        return

    layer = "log1p_norm" if "log1p_norm" in adata.layers else None
    cell_types = sorted(adata.obs[ct_col].dropna().unique())
    ct_counts = adata.obs[ct_col].value_counts()
    cell_types = [ct for ct in cell_types if ct_counts.get(ct, 0) >= 20]
    cell_types = [ct for ct in cell_types if "unassign" not in ct.lower()]

    effect_mat = pd.DataFrame(np.nan, index=available, columns=cell_types)

    sub_adata = adata[:, available].copy()
    X = sub_adata.layers[layer] if layer and layer in sub_adata.layers else sub_adata.X
    if sp.issparse(X):
        X = X.toarray()

    obs = sub_adata.obs.copy()
    expr_df = pd.DataFrame(X, index=obs.index, columns=available)
    expr_df["_visit"] = obs[DESIGN.visit_col].values
    expr_df["_ct"] = obs[ct_col].values
    expr_df["_arm"] = obs[DESIGN.arm_col].values

    for ct in cell_types:
        ct_mask = expr_df["_ct"] == ct
        ct_data = expr_df[ct_mask]
        for gene in available:
            try:
                means = {}
                for arm in [_CART_LTR_ARM, _CART_NR_ARM]:
                    for vis in ["Pre", "Post"]:
                        mask = (ct_data["_arm"] == arm) & (ct_data["_visit"] == vis)
                        vals = ct_data.loc[mask, gene]
                        means[(arm, vis)] = vals.mean() if len(vals) > 0 else np.nan
                treated_delta = means[(_CART_LTR_ARM, "Post")] - means[(_CART_LTR_ARM, "Pre")]
                control_delta = means[(_CART_NR_ARM, "Post")] - means[(_CART_NR_ARM, "Pre")]
                did_val = treated_delta - control_delta
                if np.isfinite(did_val):
                    effect_mat.loc[gene, ct] = did_val
            except Exception:
                pass

    gene_order = (
        top_genes_df.set_index("feature")
        .loc[available]
        .sort_values(beta_col, ascending=True)
        .index.tolist()
    )
    effect_mat = effect_mat.loc[gene_order]
    effect_mat = effect_mat.dropna(axis=1, how="all")

    if effect_mat.shape[1] == 0:
        ax.text(0.5, 0.5, "Insufficient cell-type data for CAR-T",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=COL_GRAY)
        ax.axis("off")
        return

    import matplotlib.colors as mcolors
    vmax = max(np.nanmax(np.abs(effect_mat.values)), 0.01)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cart_div",
        [COL_NR, "#f0f0f0", COL_LTR],
        N=256,
    )
    im = ax.imshow(
        effect_mat.values.astype(float),
        aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(effect_mat.shape[1]))
    ax.set_xticklabels(effect_mat.columns, rotation=30, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(effect_mat.shape[0]))
    ax.set_yticklabels(effect_mat.index, fontsize=7)
    ax.set_title(f"Cell-Type DiD Effects: CAR-T\n(LtR vs NR)", fontsize=11,
                 fontweight="bold")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta\Delta$ expression (DiD)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ── Panel F: Cell-type abundance DiD forest plot ──────────────────────────

def _panel_f_abundance_did(ax: plt.Axes, data_bio: dict) -> None:
    df = data_bio.get("res_abundance")
    if df is None or len(df) == 0:
        ax.text(0.5, 0.5, "Abundance DiD unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=COL_GRAY)
        ax.axis("off")
        return

    df = df[~df["celltype"].str.lower().str.contains("unassign", na=False)]
    df = df.sort_values("beta_DiD").reset_index(drop=True)
    y_pos = np.arange(len(df))

    if "ci_lo_boot" in df.columns and "ci_hi_boot" in df.columns:
        ci_lo = df["ci_lo_boot"].values
        ci_hi = df["ci_hi_boot"].values
    else:
        ci_lo = (df["beta_DiD"] - 1.96 * df["se_DiD"]).values
        ci_hi = (df["beta_DiD"] + 1.96 * df["se_DiD"]).values

    for i, (_, row) in enumerate(df.iterrows()):
        color = COL_LTR if row["beta_DiD"] > 0 else COL_NR
        ax.hlines(y_pos[i], ci_lo[i], ci_hi[i],
                  color=color, linewidth=2.0, alpha=1.0, zorder=1)
        ax.scatter(row["beta_DiD"], y_pos[i], color=color, s=30,
                   edgecolors="white", linewidths=0.8, zorder=2)
        if row.get("FDR_DiD", 1.0) < 0.25:
            ax.text(ci_hi[i] + 0.02, y_pos[i], "*",
                    va="center", fontsize=10, fontweight="bold", color=color)

    ax.axvline(0, color="#333333", lw=0.9, ls="--", zorder=0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["celltype"].tolist(), fontsize=7)
    ax.set_xlabel("Abundance DiD (arcsin-sqrt fraction)", fontsize=8)
    ax.set_title(
        f"Cell-type abundance DiD effects — CAR-T\n"
        f"({_CART_LTR_ARM} vs {_CART_NR_ARM})",
        fontsize=9, fontweight="bold",
    )
    ax.set_ylim(-0.6, len(df) - 0.4)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_LTR, markersize=4,
               label=f"{_CART_LTR_ARM} ↑"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_NR, markersize=4,
               label=f"{_CART_NR_ARM} ↑"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel G: Forest plot ──────────────────────────────────────────────────

def _panel_g_forest(ax: plt.Axes, data: dict) -> None:
    did_res = data["did_res"]
    df = did_res.sort_values("beta_DiD").reset_index(drop=True)
    y_pos = np.arange(len(df))

    analytical_lo = df["beta_DiD"] - 1.96 * df["se_DiD"]
    analytical_hi = df["beta_DiD"] + 1.96 * df["se_DiD"]
    if "ci_lo_boot" in df.columns and "ci_hi_boot" in df.columns:
        ci_lo = df["ci_lo_boot"].fillna(analytical_lo)
        ci_hi = df["ci_hi_boot"].fillna(analytical_hi)
    else:
        ci_lo = analytical_lo
        ci_hi = analytical_hi

    for i, (_, row) in enumerate(df.iterrows()):
        color = COL_LTR if row["beta_DiD"] > 0 else COL_NR
        ax.hlines(y_pos[i], ci_lo.iloc[i], ci_hi.iloc[i],
                  color=color, linewidth=1.2, alpha=1.0, zorder=1)
        ax.scatter(row["beta_DiD"], y_pos[i], color=color, s=12,
                   edgecolors="white", linewidths=0.5, alpha=1.0, zorder=2)

    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", zorder=0, alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([sig_display(f) for f in df["feature"]], fontsize=4.5)
    ax.set_xlabel(r"DiD coefficient ($\beta$, standardised)", fontsize=11)
    ax.set_title("DiD Effects Across Signatures (CAR-T, LtR vs NR)", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(-0.6, len(df) - 0.4)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_LTR, markersize=5,
               label=r"LtR $\uparrow$"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COL_NR, markersize=5,
               label=r"NR $\uparrow$"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc="lower right",
              ncol=1, frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
              handletextpad=0.4, borderpad=0.5)
    despine(ax)


# ── Panel H: Small-multiple interaction plots ─────────────────────────────

def _panel_h_interaction_grid(
    fig: plt.Figure,
    gs_parent: gridspec.SubplotSpec,
    data: dict,
    n_sigs: int = 6,
    *,
    inner_hspace: float = 0.70,
    inner_wspace: float = 0.35,
    inner_top: float = 1.0,
    group_title: str = "",
) -> tuple[list[plt.Axes], plt.Axes | None]:
    adata = data["adata"]
    did_res = data["did_res"]

    rank_p = did_res.get("p_DiD_boot", did_res["p_DiD"]).fillna(did_res["p_DiD"]) \
        if "p_DiD_boot" in did_res.columns else did_res["p_DiD"]
    top = did_res.assign(_rank_p=rank_p).sort_values("_rank_p").head(n_sigs).drop(columns="_rank_p")

    nrows, ncols = 2, 3
    ax_hdr: plt.Axes | None = None
    if inner_top < 1.0:
        _gs_split = gs_parent.subgridspec(2, 1, height_ratios=[1.0 - inner_top, inner_top], hspace=0.0)
        ax_hdr = fig.add_subplot(_gs_split[0])
        ax_hdr.axis("off")
        if group_title:
            ax_hdr.text(0.5, 1.05, group_title, transform=ax_hdr.transAxes,
                        ha="center", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
        gs_inner = _gs_split[1].subgridspec(nrows, ncols, hspace=inner_hspace, wspace=inner_wspace)
    else:
        gs_inner = gs_parent.subgridspec(nrows, ncols, hspace=inner_hspace, wspace=inner_wspace)
    axes = []

    arm_colors = {_CART_LTR_ARM: COL_LTR, _CART_NR_ARM: COL_NR}
    x_map = {VISITS[0]: 0.0, VISITS[1]: 1.0}

    for idx, (_, row) in enumerate(top.iterrows()):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs_inner[r, c])
        axes.append(ax)

        sig_col = row["feature"]
        pb = _pseudobulk(adata, sig_col)

        for arm, arm_df in pb.groupby(DESIGN.arm_col, observed=True):
            color = arm_colors.get(arm, COL_GRAY)
            for _, pid_df in arm_df.groupby(DESIGN.participant_col, observed=True):
                pid_df = pid_df.sort_values(DESIGN.visit_col, key=lambda s: s.map(x_map))
                if len(pid_df) == 2:
                    ax.plot(
                        pid_df[DESIGN.visit_col].map(x_map),
                        pid_df[sig_col],
                        color=color, alpha=0.22, linewidth=0.8, zorder=1,
                    )

        group_means = (
            pb.groupby([DESIGN.arm_col, DESIGN.visit_col], observed=True)
            [sig_col].mean().reset_index()
        )
        for arm, gdf in group_means.groupby(DESIGN.arm_col, observed=True):
            color = arm_colors.get(arm, COL_GRAY)
            gdf = gdf.sort_values(DESIGN.visit_col, key=lambda s: s.map(x_map))
            ax.plot(
                gdf[DESIGN.visit_col].map(x_map), gdf[sig_col],
                color=color, linewidth=2.0, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(VISITS, fontsize=9)
        ax.set_xlim(-0.35, 1.35)
        ax.tick_params(axis="y", labelsize=8)

        p_val = row.get("p_DiD_boot", np.nan)
        if pd.isna(p_val):
            p_val = row["p_DiD"]
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.1e}"
        ax.set_title(f"{sig_display(sig_col)}\n{p_str}", fontsize=6, fontweight="bold", pad=-2)
        despine(ax)

    legend_handles = [
        Line2D([0], [0], color=COL_LTR, linewidth=2.5, marker="o",
               markersize=6, markeredgecolor="white", label="LtR"),
        Line2D([0], [0], color=COL_NR, linewidth=2.5, marker="o",
               markersize=6, markeredgecolor="white", label="NR"),
        Line2D([0], [0], color=COL_GRAY, linewidth=0.8, alpha=0.4, label="Individual"),
    ]
    axes[-1].legend(handles=legend_handles, fontsize=16,
                    loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=3, frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    return axes, ax_hdr


# ── Panel I: Per-participant change heatmap ───────────────────────────────

def _panel_i_heatmap(ax: plt.Axes, data: dict) -> None:
    adata = data["adata"]
    sig_cols = data["sig_cols"]
    did_res = data["did_res"]

    pb = _pseudobulk_all(adata, sig_cols)

    pre_mask = pb[DESIGN.visit_col] == VISITS[0]
    post_mask = pb[DESIGN.visit_col] == VISITS[1]

    pre_num = pb.loc[pre_mask].groupby(DESIGN.participant_col, observed=True)[sig_cols].mean()
    post_num = pb.loc[post_mask].groupby(DESIGN.participant_col, observed=True)[sig_cols].mean()
    pre_arm = pb.loc[pre_mask].groupby(DESIGN.participant_col, observed=True)[DESIGN.arm_col].first()
    pre = pre_num.join(pre_arm)
    common_pids = sorted(set(pre.index) & set(post_num.index))

    if len(common_pids) == 0:
        ax.text(0.5, 0.5, "No paired participants", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        return

    delta = pd.DataFrame(
        post_num.loc[common_pids, sig_cols].values
        - pre.loc[common_pids, sig_cols].values,
        index=common_pids,
        columns=[sig_display(c) for c in sig_cols],
    )
    arms = pre.loc[common_pids, DESIGN.arm_col]

    mean_delta = delta.mean(axis=1)
    sort_df = pd.DataFrame({
        "arm_order": arms.map({_CART_LTR_ARM: 0, _CART_NR_ARM: 1}).values,
        "mean_delta": -mean_delta.values,
    }, index=common_pids)
    ordered_pids = sort_df.sort_values(["arm_order", "mean_delta"]).index.tolist()
    delta = delta.loc[ordered_pids]
    arms = arms.loc[ordered_pids]

    col_order = [sig_display(f) for f in
                 did_res.sort_values("beta_DiD")["feature"]]
    col_order = [c for c in col_order if c in delta.columns]
    delta = delta[col_order]

    vmax = np.nanpercentile(np.abs(delta.values), 95)
    vmax = max(vmax, 0.1)

    im = ax.imshow(delta.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    sidebar_w = 0.35
    _sb_left = -0.5 - sidebar_w
    for i, pid in enumerate(ordered_pids):
        arm = arms.loc[pid]
        color = COL_LTR if arm == _CART_LTR_ARM else COL_NR
        ax.add_patch(plt.Rectangle((_sb_left, i - 0.5), sidebar_w, 1.0,
                                   color=color, clip_on=False))
    ax.set_xlim(_sb_left - 0.15, len(delta.columns) - 0.5)

    n_ltr = sum(1 for p in ordered_pids if arms.loc[p] == _CART_LTR_ARM)
    if 0 < n_ltr < len(ordered_pids):
        ax.axhline(n_ltr - 0.5, color="white", linewidth=2.5, zorder=5)

    ax.set_xticks(np.arange(len(delta.columns)))
    ax.set_xticklabels(delta.columns, fontsize=4.5, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(delta.index)))
    ax.set_yticklabels([str(pid) for pid in ordered_pids], fontsize=8.5,
                       fontfamily="monospace", fontweight="medium")
    for i, tick in enumerate(ax.get_yticklabels()):
        arm = arms.iloc[i]
        tick.set_color(COL_LTR if arm == _CART_LTR_ARM else COL_NR)
    ax.set_title("Per-participant Score Change (Post − Pre) (CAR-T)", fontsize=12,
                 fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Score Δ", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_LTR,
               markersize=9, markeredgewidth=0, label="LtR"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COL_NR,
               markersize=9, markeredgewidth=0, label="NR"),
    ]
    ax.legend(handles=legend_handles, fontsize=10,
              loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.5)


# ── Panel J: Mean Δ by response group ────────────────────────────────────

def _panel_j_delta_by_response(ax: plt.Axes, data: dict) -> None:
    delta = data["delta"]
    sig_cols = data["sig_cols"]

    ltr_mask = delta[DESIGN.arm_col] == _CART_LTR_ARM
    nr_mask = delta[DESIGN.arm_col] == _CART_NR_ARM

    means_ltr = delta.loc[ltr_mask, sig_cols].mean()
    means_nr = delta.loc[nr_mask, sig_cols].mean()
    sems_ltr = delta.loc[ltr_mask, sig_cols].sem()
    sems_nr = delta.loc[nr_mask, sig_cols].sem()

    order = means_ltr.sort_values().index
    display_names = [sig_display(s) for s in order]
    y_pos = np.arange(len(order))
    bar_h = 0.35

    ax.barh(y_pos + bar_h / 2, means_ltr[order].values, height=bar_h,
            color=COL_LTR, alpha=0.85,
            xerr=sems_ltr[order].values, capsize=1, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.4}, label="LtR", edgecolor="none")
    ax.barh(y_pos - bar_h / 2, means_nr[order].values, height=bar_h,
            color=COL_NR, alpha=0.85,
            xerr=sems_nr[order].values, capsize=1, ecolor=COL_GRAY,
            error_kw={"linewidth": 0.4}, label="NR", edgecolor="none")

    ax.axvline(0, ls=":", color=COL_GRAY, lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=4.5)
    ax.set_xlabel("Mean Δ score (Post − Pre)")
    ax.set_title("Signature Changes by Response (CAR-T)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=16, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel K: Cohen's d ────────────────────────────────────────────────────

def _panel_k_cohens_d(ax: plt.Axes, data: dict) -> None:
    delta = data["delta"]
    sig_cols = data["sig_cols"]

    records = []
    for col in sig_cols:
        x = delta.loc[delta[DESIGN.arm_col] == _CART_LTR_ARM, col].dropna().values
        y = delta.loc[delta[DESIGN.arm_col] == _CART_NR_ARM, col].dropna().values
        d = _cohens_d(x, y)
        records.append({"feature": col, "display": sig_display(col), "d": d})

    df = pd.DataFrame(records).dropna(subset=["d"]).sort_values("d")
    y_pos = np.arange(len(df))
    colors = [COL_LTR if v > 0 else COL_NR for v in df["d"].values]

    ax.hlines(y_pos, 0, df["d"].values, colors=colors, lw=1.2, zorder=2)
    ax.scatter(df["d"].values, y_pos, c=colors, s=12,
               edgecolor="white", linewidth=0.4, zorder=3)

    ax.axvline(0, ls=":", color=COL_GRAY, lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["display"].values, fontsize=4.5)
    ax.set_xlabel("Cohen's d (LtR − NR)")
    ax.set_title("Effect Size of Response Separation (CAR-T)", fontsize=10,
                 fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=COL_LTR, label="LtR ↑"),
        mpatches.Patch(facecolor=COL_NR, label="NR ↑"),
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Font helpers ──────────────────────────────────────────────────────────

_BIG_FONT_RC = {
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}
_MIN_FONT = 11


@contextmanager
def _big_fonts():
    prev = {k: plt.rcParams[k] for k in _BIG_FONT_RC}
    plt.rcParams.update(_BIG_FONT_RC)
    try:
        yield
    finally:
        plt.rcParams.update(prev)


def _enforce_min_fontsize(fig, minimum: float = _MIN_FONT) -> None:
    for ax in fig.get_axes():
        for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                    + ax.get_xticklabels() + ax.get_yticklabels()
                    + ax.texts):
            if txt.get_fontsize() < minimum:
                txt.set_fontsize(minimum)
        if ax.get_legend():
            for txt in ax.get_legend().get_texts():
                if txt.get_fontsize() < minimum:
                    txt.set_fontsize(minimum)
    for txt in fig.texts:
        if txt.get_fontsize() < minimum:
            txt.set_fontsize(minimum)


# ── Composite generation ──────────────────────────────────────────────────

def generate() -> None:
    print("Supp Fig CAR-T (LtR vs NR): DiD Analysis")
    data = _prepare_data()
    data_bio = _prepare_cart_bio_data(data["adata"], data["layer"], data["sig_cols"])

    with _big_fonts():
        # Panels A–C
        for panel_name, func, size, extra in [
            ("panel_A_paired_verification", _panel_a_paired_verification, (11, 6), {}),
            ("panel_B_beta_comparison",     _panel_b_beta_comparison,     (11, 6), {}),
            ("panel_C_pvalue_inflation",    _panel_c_pvalue_inflation,    (8, 6),  {}),
        ]:
            fig, ax = plt.subplots(figsize=size)
            func(ax, data)
            _enforce_min_fontsize(fig)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

        # Panel D: Leading edge
        fig_d, ax_d_solo = plt.subplots(figsize=(12, 8))
        _panel_d_leading_edge(ax_d_solo, data_bio, composite=False)
        _enforce_min_fontsize(fig_d)
        save_panel(fig_d, "panel_D_leading_edge", FIGURE_NAME, SUPP_OUTPUT)

        # Panel E: Cell-type DiD heatmap
        fig_e, ax_e_solo = plt.subplots(figsize=(12, 7))
        _panel_e_celltype_hm(ax_e_solo, data_bio)
        _enforce_min_fontsize(fig_e)
        fig_e.tight_layout()
        save_panel(fig_e, "panel_E_celltype_hm", FIGURE_NAME, SUPP_OUTPUT)

        # Panel F: Abundance DiD
        fig_f, ax_f_solo = plt.subplots(figsize=(10, 7))
        _panel_f_abundance_did(ax_f_solo, data_bio)
        _enforce_min_fontsize(fig_f)
        fig_f.tight_layout()
        save_panel(fig_f, "panel_F_abundance_did", FIGURE_NAME, SUPP_OUTPUT)

        # Panel G: Forest plot
        fig_g, ax_g_solo = plt.subplots(figsize=(12, 6))
        _panel_g_forest(ax_g_solo, data)
        _enforce_min_fontsize(fig_g)
        fig_g.tight_layout()
        save_panel(fig_g, "panel_G_forest", FIGURE_NAME, SUPP_OUTPUT)

        # Panel H: Interaction grid
        fig_h = plt.figure(figsize=(14, 5.5))
        gs_h = fig_h.add_gridspec(1, 1)[0, 0]
        _panel_h_interaction_grid(fig_h, gs_h, data, n_sigs=6, inner_top=0.92)
        fig_h.suptitle("Participant-Level Trajectories (CAR-T, LtR vs NR)",
                        fontsize=11, fontweight="bold", y=0.97)
        _enforce_min_fontsize(fig_h)
        fig_h.tight_layout(rect=[0, 0, 1, 0.90])
        save_panel(fig_h, "panel_H_interaction_grid", FIGURE_NAME, SUPP_OUTPUT)

        # Panel I: Heatmap
        fig_i, ax_i_solo = plt.subplots(figsize=(12, 8))
        _panel_i_heatmap(ax_i_solo, data)
        _enforce_min_fontsize(fig_i)
        fig_i.tight_layout()
        save_panel(fig_i, "panel_I_heatmap", FIGURE_NAME, SUPP_OUTPUT)

        # Panels J, K
        for panel_name, func in [
            ("panel_J_delta_by_response", _panel_j_delta_by_response),
            ("panel_K_cohens_d",          _panel_k_cohens_d),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5.5))
            func(ax, data)
            _enforce_min_fontsize(fig)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    # ── Combined artboard (180 × 245 mm) ─────────────────────────────────
    _SMALL_RC = {
        "font.size":             5,
        "axes.titlesize":        5.5,
        "axes.labelsize":        5,
        "xtick.labelsize":       4.5,
        "ytick.labelsize":       4.5,
        "legend.fontsize":       4,
        "legend.title_fontsize": 4,
    }
    _MAX_FONT_COMPOSITE = 6

    def _cap_fontsize(fig, maximum):
        for ax in fig.get_axes():
            for txt in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                        + ax.get_xticklabels() + ax.get_yticklabels()
                        + ax.texts):
                if txt.get_fontsize() > maximum:
                    txt.set_fontsize(maximum)
            if ax.get_legend():
                for txt in ax.get_legend().get_texts():
                    if txt.get_fontsize() > maximum:
                        txt.set_fontsize(maximum)
        for txt in fig.texts:
            if txt.get_fontsize() > maximum:
                txt.set_fontsize(maximum)

    _prev_rc = {k: plt.rcParams[k] for k in _SMALL_RC}
    plt.rcParams.update(_SMALL_RC)

    _mm = 1.0 / 25.4
    fig_c = plt.figure(figsize=(180 * _mm, 245 * _mm))

    # 9-row gridspec matching supp_fig3 layout:
    #   Row 0 (content): A | B
    #   Row 1 (spacer):  narrow
    #   Row 2 (content): C | D | E  ← CAR-T bio panels (leading edge, celltype hm)
    #   Row 3 (spacer):  narrow
    #   Row 4 (content): F (top-left) + G (bottom-left) | H (right, spans both)
    #   Row 5 (spacer):  wider
    #   Row 6 (content): I | J
    #   Row 7 (spacer):  narrow
    #   Row 8 (content): K (centred)
    outer = fig_c.add_gridspec(
        9, 1,
        height_ratios=[1.0, 1.05, 1.8, 1.05, 3.8, 0.75, 1.8, 1.05, 1.8],
        hspace=0,
        left=0.10, right=0.95, top=0.97, bottom=0.05,
    )

    # Row 0: A | B
    gs0 = outer[0].subgridspec(1, 2, wspace=0.28, width_ratios=[1, 1.4])
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    # Row 2: C | D | E  (p-value inflation | leading edge | celltype hm)
    gs1 = outer[2].subgridspec(1, 3, wspace=0.95, width_ratios=[0.72, 1, 0.95])
    ax_c = fig_c.add_subplot(gs1[0])
    ax_d = fig_c.add_subplot(gs1[1])
    ax_e = fig_c.add_subplot(gs1[2])

    # Row 4: F (top-left) + G (bottom-left) | H (right, spans both)
    gs2 = outer[4].subgridspec(2, 2, width_ratios=[1, 1.6], hspace=0.50, wspace=0.40)
    ax_f = fig_c.add_subplot(gs2[0, 0])      # Panel F: abundance DiD
    ax_g_comp = fig_c.add_subplot(gs2[1, 0]) # Panel G: forest plot

    # Row 6: I | J
    gs3 = outer[6].subgridspec(1, 2, wspace=0.55)
    ax_i = fig_c.add_subplot(gs3[0])
    ax_j = fig_c.add_subplot(gs3[1])

    # Row 8: K (centred)
    gs4 = outer[8].subgridspec(1, 3, width_ratios=[0.6, 1.8, 0.6], wspace=0.40)
    ax_k = fig_c.add_subplot(gs4[1])

    # Draw all panels
    _panel_a_paired_verification(ax_a, data)
    _panel_b_beta_comparison(ax_b, data)
    _panel_c_pvalue_inflation(ax_c, data)
    _panel_d_leading_edge(ax_d, data_bio, composite=True)
    _panel_e_celltype_hm(ax_e, data_bio)
    _panel_f_abundance_did(ax_f, data_bio)
    _panel_g_forest(ax_g_comp, data)
    axes_h, ax_h_hdr = _panel_h_interaction_grid(
        fig_c, gs2[:, 1], data, n_sigs=6,
        inner_hspace=0.80, inner_wspace=0.30, inner_top=0.88,
        group_title="Participant-Level Trajectories (CAR-T, LtR vs NR)",
    )
    _panel_i_heatmap(ax_i, data)
    _panel_j_delta_by_response(ax_j, data)
    _panel_k_cohens_d(ax_k, data)

    # Move legends inside for space
    _inside = {
        ax_b:      "lower left",
        ax_c:      "lower right",
        ax_f:      "upper left",
        ax_g_comp: "upper left",
        ax_j:      "lower right",
        ax_k:      "lower right",
    }
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=4.5, loc=loc,
                frameon=True, framealpha=0.85,
                handlelength=1, handletextpad=0.3,
                borderpad=0.3, labelspacing=0.2,
            )

    # Panel A: legend inside upper-right
    leg_a = ax_a.get_legend()
    if leg_a:
        handles_a = leg_a.legend_handles
        labels_a = [t.get_text() for t in leg_a.get_texts()]
        leg_a.remove()
        ax_a.legend(handles=handles_a, labels=labels_a,
                    fontsize=4.5, loc="upper right",
                    bbox_to_anchor=(1.0, 1.10),
                    frameon=True, framealpha=0.85,
                    handlelength=1, handletextpad=0.3,
                    borderpad=0.3, labelspacing=0.2, ncol=2)

    # Panel H: consolidated legend
    if axes_h:
        leg_h = axes_h[-1].get_legend()
        if leg_h:
            leg_h.remove()
        mid_top_ax = axes_h[1]
        _hh = [
            Line2D([0], [0], color=COL_LTR, linewidth=1.5, marker="o",
                   markersize=3, markeredgecolor="white", label="LtR"),
            Line2D([0], [0], color=COL_NR, linewidth=1.5, marker="o",
                   markersize=3, markeredgecolor="white", label="NR"),
            Line2D([0], [0], color=COL_GRAY, linewidth=0.6, alpha=0.4, label="Individual"),
        ]
        mid_top_ax.legend(handles=_hh, fontsize=4.5,
                          loc="upper center", bbox_to_anchor=(0.5, -0.28),
                          ncol=3, frameon=True, framealpha=0.95, edgecolor="#CCCCCC")

    # Panel B: shrink annotation text
    for txt in ax_b.texts:
        txt.set_fontsize(max(txt.get_fontsize() * 0.55, 3.0))

    # Panel C: nudge "p = 0.05" annotation
    for txt in ax_c.texts:
        if "0.05" in txt.get_text():
            x, y = txt.get_position()
            txt.set_position((x, y - 2))

    # Panel I: replace below-legend with y-axis group labels
    from matplotlib.colors import to_rgba
    from matplotlib.transforms import blended_transform_factory

    leg_i = ax_i.get_legend()
    if leg_i:
        leg_i.remove()
    _trans_i = blended_transform_factory(ax_i.transAxes, ax_i.transData)
    _ltr_rgba = to_rgba(COL_LTR)
    fig_c.canvas.draw_idle()
    _n_ltr_i = sum(
        1 for t in ax_i.get_yticklabels()
        if np.allclose(to_rgba(t.get_color()), _ltr_rgba, atol=0.02)
    )
    _n_total_i = len(ax_i.get_yticklabels())
    _n_nr_i = _n_total_i - _n_ltr_i
    if _n_ltr_i > 0:
        ax_i.text(-0.16, (_n_ltr_i - 1) / 2, "LtR",
                  transform=_trans_i, color=COL_LTR,
                  fontsize=4, fontweight="bold",
                  ha="right", va="center", rotation=90, clip_on=False)
    if _n_nr_i > 0:
        ax_i.text(-0.16, _n_ltr_i + (_n_nr_i - 1) / 2, "NR",
                  transform=_trans_i, color=COL_NR,
                  fontsize=4, fontweight="bold",
                  ha="right", va="center", rotation=90, clip_on=False)

    # Shrink colorbar for panel I
    for _child_ax in fig_c.get_axes():
        if _child_ax.get_ylabel() == "Score Δ":
            _child_ax.set_ylabel("Score Δ", fontsize=3.5, labelpad=1)
            _child_ax.tick_params(labelsize=3, pad=1)
            break

    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    # Panel labels
    _lbl_fs = 7
    _lbl_pos = {
        "C": (-0.55, 1.22),
        "D": (-0.15, 1.22),
        "E": (-0.28, 1.22),
        "F": (-0.15, 1.22),
        "G": (-0.15, 1.22),
    }
    _lbl_default = (-0.15, 1.12)
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"),
        (ax_d, "D"), (ax_e, "E"), (ax_f, "F"),
        (ax_g_comp, "G"), (ax_i, "I"), (ax_j, "J"), (ax_k, "K"),
    ]:
        _lx, _ly = _lbl_pos.get(lbl, _lbl_default)
        ax.text(_lx, _ly, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")
    if ax_h_hdr is not None:
        ax_h_hdr.text(-0.15, 1.68, "H", transform=ax_h_hdr.transAxes,
                      fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
                      clip_on=False)

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    del data["adata"]
    del data
    gc.collect()
    print(f"  {FIGURE_NAME} complete: 11 individual panels + combined (A–K)\n")


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()
    generate()
