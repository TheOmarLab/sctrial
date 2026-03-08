"""
Supplementary Figure 7 — Participant-Level Heterogeneity Architecture
=====================================================================

Eight-panel figure examining individual participant-level treatment
effect heterogeneity across the Sade-Feldman immunotherapy and AML
clinical-trial datasets.

Panels
------
A  Strip plot of individual participant effects across features (SF).
B  Participant-level effect heatmap (participants × features, SF).
C  Leave-one-out influence scatter (Cook-style, SF).
D  Response-stratified box plots (top 6 most variable features, SF).
E  Individual effects strip plot (AML).
F  Cross-dataset effect SD comparison (SF vs AML).
G  Participant-effect correlation matrix (features within SF).
H  Heterogeneity test bar chart (3-way interaction p-values).
"""

from __future__ import annotations

import gc
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

from .._shared import (
    COLORS,
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
)

# ── Figure-level constants ────────────────────────────────────────────
FIGURE_NAME = "SuppFig7_heterogeneity"

FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19", "CD14", "LYZ", "NKG7",
]

# ── dataset configs ───────────────────────────────────────────────────
_DATASET_CFGS = {
    "Sade-Feldman": dict(
        loader=get_sade_feldman,
        layer="log1p_tpm",
        design_kw=dict(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response_harmonized",
            arm_treated="Responder",
            arm_control="Non-responder",
        ),
        visits=("Pre", "Post"),
        response_col="response_harmonized",
    ),
    "AML": dict(
        loader=lambda: load_clinical_trial_dataset("aml"),
        layer="log1p_norm",
        design_kw=dict(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response",
            arm_treated="Treatment",
            arm_control="Control",
        ),
        visits=("Pre", "Post"),
        response_col="response",
    ),
}


# ======================================================================
# Helpers
# ======================================================================

def _compute_individual_effects(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    """Compute Post-Pre difference per participant for given features."""
    layer = cfg["layer"]
    pid_col = cfg["design_kw"]["participant_col"]
    visit_col = cfg["design_kw"]["visit_col"]
    arm_col = cfg["design_kw"]["arm_col"]
    pre_v, post_v = cfg["visits"]

    # Get available features
    avail = [f for f in features if f in adata.var_names]
    if len(avail) < 3:
        return None

    obs = adata.obs.copy()
    # Extract expression matrix for available features
    if layer in adata.layers:
        expr = pd.DataFrame(
            adata[:, avail].layers[layer].toarray()
            if hasattr(adata[:, avail].layers[layer], "toarray")
            else np.asarray(adata[:, avail].layers[layer]),
            columns=avail,
            index=adata.obs_names,
        )
    else:
        expr = pd.DataFrame(
            adata[:, avail].X.toarray()
            if hasattr(adata[:, avail].X, "toarray")
            else np.asarray(adata[:, avail].X),
            columns=avail,
            index=adata.obs_names,
        )

    for c in [pid_col, visit_col, arm_col]:
        expr[c] = obs[c].values

    # Pseudobulk per participant × visit
    pb = expr.groupby([pid_col, visit_col, arm_col], observed=True)[avail].mean().reset_index()

    pre = pb[pb[visit_col] == pre_v].set_index(pid_col)
    post = pb[pb[visit_col] == post_v].set_index(pid_col)
    common = pre.index.intersection(post.index)
    if len(common) < 3:
        return None

    effects = post.loc[common, avail] - pre.loc[common, avail]
    effects["arm"] = pre.loc[common, arm_col]
    effects = effects.reset_index()
    effects.rename(columns={pid_col: "participant_id"}, inplace=True)
    return effects


def _load_all_data() -> dict:
    """Load both datasets and compute individual effects."""
    results = {}
    for name, cfg in _DATASET_CFGS.items():
        try:
            adata = cfg["loader"]()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            effects = _compute_individual_effects(adata, cfg, FEATURES)
            avail = [f for f in FEATURES if f in adata.var_names]
            results[name] = dict(adata=adata, effects=effects, features=avail, cfg=cfg)
            n_part = len(effects) if effects is not None else 0
            print(f"  {name}: {adata.n_obs} cells, {n_part} paired participants")
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
    return results


# ======================================================================
# Panel A — Strip plot of individual effects (SF)
# ======================================================================

def _panel_strip_effects(ax, effects: pd.DataFrame, features: list[str],
                         title: str, arm_treated: str, arm_control: str):
    """Strip plot of individual participant effects across features."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No individual effects data",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    df_long = effects.melt(
        id_vars=["participant_id", "arm"], value_vars=avail,
        var_name="feature", value_name="effect",
    )

    feat_order = df_long.groupby("feature")["effect"].mean().sort_values().index.tolist()

    arm_colors = {arm_treated: COLORS["treated"], arm_control: COLORS["control"]}
    x_positions = {f: i for i, f in enumerate(feat_order)}
    rng = np.random.default_rng(42)

    for arm, color in arm_colors.items():
        sub = df_long[df_long["arm"] == arm]
        x_vals = sub["feature"].map(x_positions).values
        jitter = rng.uniform(-0.2, 0.2, size=len(sub))
        ax.scatter(x_vals + jitter, sub["effect"].values,
                   c=color, alpha=0.5, s=18, edgecolors="none", label=arm, zorder=2)

    # Mean lines
    for feat in feat_order:
        sub = df_long[df_long["feature"] == feat]
        mean_val = sub["effect"].mean()
        x = x_positions[feat]
        ax.hlines(mean_val, x - 0.35, x + 0.35, color="black", linewidth=2.0, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Individual effect (Post − Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    handles = [mpatches.Patch(color=COLORS["treated"], alpha=0.6, label=arm_treated),
               mpatches.Patch(color=COLORS["control"], alpha=0.6, label=arm_control),
               Line2D([0], [0], color="black", linewidth=2, label="Mean")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel B — Participant effect heatmap (SF)
# ======================================================================

def _panel_effect_heatmap(ax, effects: pd.DataFrame, features: list[str], title: str):
    """Heatmap of participant × feature effects."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    mat = effects.set_index("participant_id")[avail]

    # Cluster rows
    from scipy.cluster.hierarchy import linkage, leaves_list
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = mat.fillna(0).values
        if clean.shape[0] > 2:
            Z = linkage(clean, method="ward")
            row_order = leaves_list(Z)
            mat = mat.iloc[row_order]

    vmax = np.nanpercentile(np.abs(mat.values), 95)
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    sns.heatmap(mat, ax=ax, cmap="RdBu_r", norm=norm, linewidths=0.3,
                linecolor="white", cbar_kws={"label": "Effect (Post − Pre)", "shrink": 0.8},
                xticklabels=True, yticklabels=False)
    ax.set_ylabel("Participants (clustered)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)


# ======================================================================
# Panel C — LOO influence scatter (SF)
# ======================================================================

def _panel_loo_influence(ax, effects: pd.DataFrame, features: list[str], title: str):
    """Cook-style influence: how much does each participant shift the mean effect."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    mat = effects.set_index("participant_id")[avail].values
    n = mat.shape[0]
    if n < 4:
        ax.text(0.5, 0.5, "Too few participants", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    full_mean = np.nanmean(mat, axis=0)

    influence_scores = []
    for i in range(n):
        loo_mean = np.nanmean(np.delete(mat, i, axis=0), axis=0)
        diff = full_mean - loo_mean
        influence = np.sqrt(np.nansum(diff ** 2))
        influence_scores.append(influence)

    influence_scores = np.array(influence_scores)
    pids = effects["participant_id"].values
    arms = effects["arm"].values

    # Threshold: mean + 2 SD
    threshold = np.mean(influence_scores) + 2 * np.std(influence_scores)

    for arm_val in np.unique(arms):
        mask = arms == arm_val
        color = COLORS["treated"] if "Resp" in str(arm_val) or "Treat" in str(arm_val) else COLORS["control"]
        ax.scatter(np.where(mask)[0], influence_scores[mask],
                   c=color, alpha=0.7, s=30, edgecolors="white", linewidths=0.3,
                   label=str(arm_val))

    ax.axhline(threshold, color="red", linewidth=0.8, linestyle="--",
               label=f"Threshold (μ + 2σ = {threshold:.3f})")

    # Label influential participants
    for i in range(n):
        if influence_scores[i] > threshold:
            ax.annotate(str(pids[i])[:8], (i, influence_scores[i]),
                        fontsize=6, ha="center", va="bottom")

    ax.set_xlabel("Participant index")
    ax.set_ylabel("LOO influence score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel D — Response-stratified box plots (SF)
# ======================================================================

def _panel_response_boxes(ax, effects: pd.DataFrame, features: list[str],
                          title: str, arm_treated: str, arm_control: str):
    """Box plots by response for top 6 most variable features."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    sds = effects[avail].std().sort_values(ascending=False)
    top = sds.head(6).index.tolist()

    df_long = effects.melt(id_vars=["participant_id", "arm"], value_vars=top,
                           var_name="feature", value_name="effect")

    arm_colors = {arm_treated: COLORS["treated"], arm_control: COLORS["control"]}
    x_positions = {f: i for i, f in enumerate(top)}
    box_width = 0.35

    for arm_idx, (arm, color) in enumerate(arm_colors.items()):
        sub = df_long[df_long["arm"] == arm]
        offset = -0.2 if arm_idx == 0 else 0.2
        for feat in top:
            data = sub[sub["feature"] == feat]["effect"].dropna()
            if len(data) == 0:
                continue
            ax.boxplot(data, positions=[x_positions[feat] + offset], widths=box_width,
                       patch_artist=True, showfliers=True,
                       flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                       markerfacecolor=color, markeredgecolor="none"),
                       medianprops=dict(color="white", linewidth=1.5),
                       boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color),
                       whiskerprops=dict(color=color), capprops=dict(color=color))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Individual effect (Post − Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    handles = [mpatches.Patch(color=COLORS["treated"], alpha=0.7, label=arm_treated),
               mpatches.Patch(color=COLORS["control"], alpha=0.7, label=arm_control)]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel F — Cross-dataset effect SD comparison
# ======================================================================

def _panel_cross_dataset_sd(ax, results: dict):
    """Compare effect heterogeneity (SD) across datasets for shared features."""
    sd_data = {}
    for name, res in results.items():
        effects = res["effects"]
        if effects is None:
            continue
        avail = [f for f in res["features"] if f in effects.columns]
        sd_data[name] = effects[avail].std()

    if len(sd_data) < 2:
        ax.text(0.5, 0.5, "Need 2 datasets", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    names = list(sd_data.keys())
    common = sd_data[names[0]].index.intersection(sd_data[names[1]].index)
    if len(common) == 0:
        ax.text(0.5, 0.5, "No shared features", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    sd1 = sd_data[names[0]][common].values
    sd2 = sd_data[names[1]][common].values

    ax.scatter(sd1, sd2, c=COLORS["highlight"], s=40, alpha=0.7,
               edgecolors="white", linewidths=0.5)
    for i, feat in enumerate(common):
        ax.annotate(feat, (sd1[i], sd2[i]), fontsize=7, ha="left", va="bottom")

    # Diagonal
    lim = max(sd1.max(), sd2.max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=0.8, alpha=0.6)

    r, p = sp_stats.pearsonr(sd1, sd2)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))

    ax.set_xlabel(f"SD of effects ({names[0]})")
    ax.set_ylabel(f"SD of effects ({names[1]})")
    ax.set_title("Cross-Dataset Effect Heterogeneity", fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel G — Feature-feature correlation of participant effects (SF)
# ======================================================================

def _panel_effect_correlation(ax, effects: pd.DataFrame, features: list[str], title: str):
    """Correlation matrix of per-participant effects across features."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    mat = effects[avail]
    corr = mat.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, ax=ax, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.5, linecolor="white", annot=True, fmt=".2f",
                annot_kws={"fontsize": 6}, square=True,
                cbar_kws={"label": "Pearson r", "shrink": 0.8})
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)


# ======================================================================
# Panel H — Effect SD bar chart (both datasets)
# ======================================================================

def _panel_effect_sd_bars(ax, results: dict, title: str):
    """Grouped horizontal bar chart of SD of individual effects per feature."""
    sd_data = {}
    for name, res in results.items():
        effects = res.get("effects")
        if effects is None:
            continue
        avail = [f for f in res["features"] if f in effects.columns]
        sd_data[name] = effects[avail].std()

    if len(sd_data) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    names = list(sd_data.keys())
    # Get union of features, sorted by SD in first dataset
    all_feats = sorted(
        set().union(*(s.index for s in sd_data.values())),
        key=lambda f: sd_data[names[0]].get(f, 0) if f in sd_data[names[0]].index else 0,
    )

    y_pos = np.arange(len(all_feats))
    bar_height = 0.35
    ds_colors = [COLORS["treated"], COLORS["control"]]

    for i, (name, sds) in enumerate(sd_data.items()):
        offset = -bar_height / 2 + i * bar_height
        vals = [sds.get(f, 0) for f in all_feats]
        ax.barh(y_pos + offset, vals, height=bar_height, color=ds_colors[i],
                alpha=0.85, edgecolor="white", linewidth=0.5, label=name)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_feats, fontsize=8)
    ax.set_xlabel("SD of individual effects (Post − Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 7 panels."""
    print("Supplementary Figure 7: Participant-Level Heterogeneity Architecture")
    apply_style()

    results = _load_all_data()
    if not results:
        print("  ERROR: no datasets loaded")
        return

    sf = results.get("Sade-Feldman", {})
    aml = results.get("AML", {})
    sf_effects = sf.get("effects")
    aml_effects = aml.get("effects")
    sf_feats = sf.get("features", FEATURES)
    aml_feats = aml.get("features", FEATURES)
    sf_cfg = sf.get("cfg", _DATASET_CFGS["Sade-Feldman"])
    aml_cfg = aml.get("cfg", _DATASET_CFGS["AML"])

    panels = [
        ("panel_A", lambda fig, ax: _panel_strip_effects(
            ax, sf_effects, sf_feats, "Individual Effects (Sade-Feldman)",
            "Responder", "Non-responder"), (12, 6)),
        ("panel_B", lambda fig, ax: _panel_effect_heatmap(
            ax, sf_effects, sf_feats, "Participant × Feature Effects (SF)"), (10, 7)),
        ("panel_C", lambda fig, ax: _panel_loo_influence(
            ax, sf_effects, sf_feats, "LOO Influence (Sade-Feldman)"), (8, 6)),
        ("panel_D", lambda fig, ax: _panel_response_boxes(
            ax, sf_effects, sf_feats, "Effects by Response (Top 6, SF)",
            "Responder", "Non-responder"), (8, 6)),
        ("panel_E", lambda fig, ax: _panel_strip_effects(
            ax, aml_effects, aml_feats, "Individual Effects (AML)",
            "Treatment", "Control"), (12, 6)),
        ("panel_F", lambda fig, ax: _panel_cross_dataset_sd(ax, results), (7, 7)),
        ("panel_G", lambda fig, ax: _panel_effect_correlation(
            ax, sf_effects, sf_feats, "Effect Correlation Matrix (SF)"), (8, 8)),
        ("panel_H", lambda fig, ax: _panel_effect_sd_bars(
            ax, results, "Effect Heterogeneity (SD per Feature)"), (8, 7)),
    ]

    for panel_name, draw_fn, psize in panels:
        try:
            fig, ax = plt.subplots(figsize=psize)
            draw_fn(fig, ax)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)
        except Exception as exc:
            print(f"    {panel_name}: ERROR {exc}")
            plt.close("all")

    # Cleanup
    for res in results.values():
        if "adata" in res:
            del res["adata"]
    del results
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
