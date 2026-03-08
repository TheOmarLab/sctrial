"""
Supplementary Figure 8 — Temporal Structure & Visit-Window Robustness
=====================================================================

Eight-panel figure examining temporal dynamics and visit-window
robustness across the Sade-Feldman immunotherapy and AML datasets.

Panels
------
A  Pre-vs-Post mean expression scatter (SF, per feature).
B  Pre-vs-Post mean expression scatter (AML, per feature).
C  Within-arm Pre→Post change (SF, treated vs control, paired).
D  Within-arm Pre→Post change (AML, treated vs control, paired).
E  Effect size vs baseline expression (SF).
F  Effect size vs baseline expression (AML).
G  Visit-window cell count distribution (SF, per participant × visit).
H  Cross-dataset Pre/Post fold-change comparison.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
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
FIGURE_NAME = "SuppFig8_temporal"

FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19", "CD14", "LYZ", "NKG7",
]

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
    ),
}


# ======================================================================
# Helpers
# ======================================================================

def _get_pseudobulk(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    """Return pseudobulk DataFrame: participant × visit × arm × features."""
    layer = cfg["layer"]
    pid_col = cfg["design_kw"]["participant_col"]
    visit_col = cfg["design_kw"]["visit_col"]
    arm_col = cfg["design_kw"]["arm_col"]

    avail = [f for f in features if f in adata.var_names]
    if len(avail) < 3:
        return None

    if layer in adata.layers:
        X = adata[:, avail].layers[layer]
    else:
        X = adata[:, avail].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)

    expr = pd.DataFrame(X, columns=avail, index=adata.obs_names)
    for c in [pid_col, visit_col, arm_col]:
        expr[c] = adata.obs[c].values

    pb = expr.groupby([pid_col, visit_col, arm_col], observed=True)[avail].mean().reset_index()
    pb.rename(columns={pid_col: "participant_id", visit_col: "visit", arm_col: "arm"}, inplace=True)
    return pb


def _load_all() -> dict:
    """Load datasets and compute pseudobulk."""
    results = {}
    for name, cfg in _DATASET_CFGS.items():
        try:
            adata = cfg["loader"]()
            if name == "Sade-Feldman":
                adata = harmonize_response(adata)
            avail = [f for f in FEATURES if f in adata.var_names]
            pb = _get_pseudobulk(adata, cfg, FEATURES)
            results[name] = dict(adata=adata, pb=pb, features=avail, cfg=cfg)
            print(f"  {name}: {adata.n_obs} cells, {len(avail)} features")
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
    return results


# ======================================================================
# Panel A/B — Pre vs Post scatter
# ======================================================================

def _panel_pre_post_scatter(ax, pb: pd.DataFrame, features: list[str],
                            cfg: dict, title: str):
    """Scatter of mean expression Pre vs Post, one point per feature."""
    if pb is None or len(pb) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    pre_v, post_v = cfg["visits"]
    avail = [f for f in features if f in pb.columns]

    pre_means = pb[pb["visit"] == pre_v][avail].mean()
    post_means = pb[pb["visit"] == post_v][avail].mean()

    ax.scatter(pre_means.values, post_means.values, c=COLORS["highlight"],
               s=40, alpha=0.7, edgecolors="white", linewidths=0.5)
    for i, feat in enumerate(avail):
        ax.annotate(feat, (pre_means[feat], post_means[feat]),
                    fontsize=7, ha="left", va="bottom")

    # Diagonal
    lim_lo = min(pre_means.min(), post_means.min()) * 0.9
    lim_hi = max(pre_means.max(), post_means.max()) * 1.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="gray", linewidth=0.8)

    r, p = sp_stats.pearsonr(pre_means.values, post_means.values)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))

    ax.set_xlabel(f"Mean expression ({pre_v})")
    ax.set_ylabel(f"Mean expression ({post_v})")
    ax.set_title(title, fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel C/D — Within-arm paired changes
# ======================================================================

def _panel_within_arm_change(ax, pb: pd.DataFrame, features: list[str],
                             cfg: dict, title: str):
    """Paired Pre→Post changes for treated and control arms."""
    if pb is None or len(pb) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    pre_v, post_v = cfg["visits"]
    arm_treated = cfg["design_kw"]["arm_treated"]
    arm_control = cfg["design_kw"]["arm_control"]
    avail = [f for f in features if f in pb.columns]

    changes = {}
    for arm_label, arm_val in [("Treated", arm_treated), ("Control", arm_control)]:
        arm_pb = pb[pb["arm"] == arm_val]
        pre = arm_pb[arm_pb["visit"] == pre_v].set_index("participant_id")
        post = arm_pb[arm_pb["visit"] == post_v].set_index("participant_id")
        common = pre.index.intersection(post.index)
        if len(common) > 0:
            diff = post.loc[common, avail].mean() - pre.loc[common, avail].mean()
            changes[arm_label] = diff

    if len(changes) == 0:
        ax.text(0.5, 0.5, "No paired data", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    y_pos = np.arange(len(avail))
    bar_height = 0.35

    if "Treated" in changes:
        ax.barh(y_pos - bar_height / 2, changes["Treated"][avail].values,
                height=bar_height, color=COLORS["treated"], alpha=0.85,
                label=arm_treated, edgecolor="white", linewidth=0.5)
    if "Control" in changes:
        ax.barh(y_pos + bar_height / 2, changes["Control"][avail].values,
                height=bar_height, color=COLORS["control"], alpha=0.85,
                label=arm_control, edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avail, fontsize=8)
    ax.set_xlabel("Mean change (Post − Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel E/F — Effect size vs baseline expression
# ======================================================================

def _panel_effect_vs_baseline(ax, pb: pd.DataFrame, features: list[str],
                              cfg: dict, title: str):
    """Scatter: effect vs Pre-treatment mean expression.

    Uses DiD when both arms have paired data.  Falls back to treated-arm
    Pre→Post change when the control arm lacks paired participants.
    """
    if pb is None or len(pb) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    pre_v, post_v = cfg["visits"]
    arm_treated = cfg["design_kw"]["arm_treated"]
    arm_control = cfg["design_kw"]["arm_control"]
    avail = [f for f in features if f in pb.columns]

    # Check whether control arm has paired Pre/Post participants
    ctrl_pb = pb[pb["arm"] == arm_control]
    ctrl_pre_ids = set(ctrl_pb[ctrl_pb["visit"] == pre_v]["participant_id"])
    ctrl_post_ids = set(ctrl_pb[ctrl_pb["visit"] == post_v]["participant_id"])
    ctrl_paired = ctrl_pre_ids & ctrl_post_ids
    use_did = len(ctrl_paired) > 0

    effects = []
    baselines = []
    for feat in avail:
        t_pre = pb[(pb["arm"] == arm_treated) & (pb["visit"] == pre_v)][feat].mean()
        t_post = pb[(pb["arm"] == arm_treated) & (pb["visit"] == post_v)][feat].mean()
        if use_did:
            c_pre = pb[(pb["arm"] == arm_control) & (pb["visit"] == pre_v)][feat].mean()
            c_post = pb[(pb["arm"] == arm_control) & (pb["visit"] == post_v)][feat].mean()
            eff = (t_post - t_pre) - (c_post - c_pre)
            baseline = (t_pre + c_pre) / 2
        else:
            eff = t_post - t_pre
            baseline = t_pre
        effects.append(eff)
        baselines.append(baseline)

    effects = np.array(effects)
    baselines = np.array(baselines)

    ax.scatter(baselines, effects, c=COLORS["highlight"], s=40, alpha=0.7,
               edgecolors="white", linewidths=0.5)
    for i, feat in enumerate(avail):
        ax.annotate(feat, (baselines[i], effects[i]), fontsize=7, ha="left", va="bottom")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    mask = np.isfinite(baselines) & np.isfinite(effects)
    if mask.sum() > 3:
        r, p = sp_stats.pearsonr(baselines[mask], effects[mask])
        ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))

    ax.set_xlabel("Baseline expression (Pre mean)")
    y_label = "DiD effect size" if use_did else "Treated Δ (Post − Pre)"
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel G — Cell count per participant × visit
# ======================================================================

def _panel_cell_counts(ax, adata, cfg: dict, title: str):
    """Distribution of cell counts per participant × visit."""
    pid_col = cfg["design_kw"]["participant_col"]
    visit_col = cfg["design_kw"]["visit_col"]
    arm_col = cfg["design_kw"]["arm_col"]
    arm_treated = cfg["design_kw"]["arm_treated"]
    arm_control = cfg["design_kw"]["arm_control"]

    counts = adata.obs.groupby([pid_col, visit_col, arm_col], observed=True).size().reset_index(name="n_cells")
    counts.rename(columns={pid_col: "participant_id", visit_col: "visit", arm_col: "arm"}, inplace=True)

    arms = {arm_treated: COLORS["treated"], arm_control: COLORS["control"]}
    visits = cfg["visits"]

    x_pos = 0
    tick_positions = []
    tick_labels = []

    for visit in visits:
        for arm, color in arms.items():
            sub = counts[(counts["visit"] == visit) & (counts["arm"] == arm)]
            if len(sub) == 0:
                x_pos += 1
                continue
            bp = ax.boxplot(sub["n_cells"], positions=[x_pos], widths=0.6,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                            markerfacecolor=color, markeredgecolor="none"),
                            medianprops=dict(color="white", linewidth=1.5),
                            boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color),
                            whiskerprops=dict(color=color), capprops=dict(color=color))
            tick_positions.append(x_pos)
            arm_short = "T" if arm == arm_treated else "C"
            tick_labels.append(f"{visit}\n({arm_short})")
            x_pos += 1
        x_pos += 0.5  # gap between visits

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Cells per participant")
    ax.set_title(title, fontsize=11, fontweight="bold")

    handles = [mpatches.Patch(color=COLORS["treated"], alpha=0.7, label=arm_treated),
               mpatches.Patch(color=COLORS["control"], alpha=0.7, label=arm_control)]
    ax.legend(handles=handles, fontsize=8, loc="upper right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel H — Cross-dataset fold change comparison
# ======================================================================

def _panel_cross_dataset_fc(ax, results: dict):
    """Compare Pre→Post log fold-change across datasets (treated arm only)."""
    fc_data = {}
    for name, res in results.items():
        pb = res.get("pb")
        cfg = res["cfg"]
        if pb is None:
            continue
        pre_v, post_v = cfg["visits"]
        arm_treated = cfg["design_kw"]["arm_treated"]
        avail = [f for f in res["features"] if f in pb.columns]

        treated_pb = pb[pb["arm"] == arm_treated]
        pre_mean = treated_pb[treated_pb["visit"] == pre_v][avail].mean()
        post_mean = treated_pb[treated_pb["visit"] == post_v][avail].mean()
        # Log2 fold change (add small pseudocount)
        fc = np.log2((post_mean + 0.01) / (pre_mean + 0.01))
        fc_data[name] = fc

    if len(fc_data) < 2:
        ax.text(0.5, 0.5, "Need 2 datasets", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    names = list(fc_data.keys())
    common = fc_data[names[0]].index.intersection(fc_data[names[1]].index)
    common = [f for f in common if np.isfinite(fc_data[names[0]][f]) and np.isfinite(fc_data[names[1]][f])]
    if len(common) == 0:
        ax.text(0.5, 0.5, "No shared features", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    fc1 = np.array([fc_data[names[0]][f] for f in common])
    fc2 = np.array([fc_data[names[1]][f] for f in common])

    ax.scatter(fc1, fc2, c=COLORS["highlight"], s=40, alpha=0.7,
               edgecolors="white", linewidths=0.5)
    for i, feat in enumerate(common):
        ax.annotate(feat, (fc1[i], fc2[i]), fontsize=7, ha="left", va="bottom")

    # Diagonal
    lim = max(np.abs(fc1).max(), np.abs(fc2).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", linewidth=0.8, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    r, p = sp_stats.pearsonr(fc1, fc2)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))

    ax.set_xlabel(f"log₂FC (treated, {names[0]})")
    ax.set_ylabel(f"log₂FC (treated, {names[1]})")
    ax.set_title("Cross-Dataset Fold Change (Treated Arm)", fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 8 panels."""
    print("Supplementary Figure 8: Temporal Structure & Visit-Window Robustness")
    apply_style()

    results = _load_all()
    if not results:
        print("  ERROR: no datasets loaded")
        return

    sf = results.get("Sade-Feldman", {})
    aml = results.get("AML", {})

    panels = [
        ("panel_A", lambda fig, ax: _panel_pre_post_scatter(
            ax, sf.get("pb"), sf.get("features", []), sf["cfg"],
            "Pre vs Post Expression (SF)"), (7, 7)),
        ("panel_B", lambda fig, ax: _panel_pre_post_scatter(
            ax, aml.get("pb"), aml.get("features", []), aml["cfg"],
            "Pre vs Post Expression (AML)"), (7, 7)),
        ("panel_C", lambda fig, ax: _panel_within_arm_change(
            ax, sf.get("pb"), sf.get("features", []), sf["cfg"],
            "Within-Arm Change (SF)"), (8, 7)),
        ("panel_D", lambda fig, ax: _panel_within_arm_change(
            ax, aml.get("pb"), aml.get("features", []), aml["cfg"],
            "Within-Arm Change (AML)"), (8, 7)),
        ("panel_E", lambda fig, ax: _panel_effect_vs_baseline(
            ax, sf.get("pb"), sf.get("features", []), sf["cfg"],
            "Effect vs Baseline (SF)"), (7, 7)),
        ("panel_F", lambda fig, ax: _panel_effect_vs_baseline(
            ax, aml.get("pb"), aml.get("features", []), aml["cfg"],
            "Effect vs Baseline (AML)"), (7, 7)),
        ("panel_G", lambda fig, ax: _panel_cell_counts(
            ax, sf["adata"], sf["cfg"],
            "Cell Counts per Participant × Visit (SF)"), (8, 6)),
        ("panel_H", lambda fig, ax: _panel_cross_dataset_fc(ax, results), (7, 7)),
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
