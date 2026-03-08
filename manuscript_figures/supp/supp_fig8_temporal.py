"""
Supplementary Figure 8 — Temporal Structure & Visit-Window Robustness
=====================================================================

Eight-panel figure examining temporal dynamics and visit-window
robustness across the Sade-Feldman immunotherapy, AML, and CAR-T
datasets.

Panels
------
A  Pre-vs-Post mean expression scatter (SF, per feature).
B  Pre-vs-Post mean expression scatter (AML, per feature).
C  Within-arm Pre->Post change (SF, treated vs control, paired).
D  Within-arm Pre->Post change (AML + CAR-T, side by side).
E  Effect size vs baseline expression (SF).
F  Effect size vs baseline expression (AML).
G  Visit-window cell count distribution (SF, per participant x visit).
H  Cross-dataset Pre/Post fold-change comparison (SF vs AML vs CAR-T).
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from adjustText import adjust_text
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

# -- Figure-level constants ------------------------------------------------
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
    "CAR-T": dict(
        loader=lambda: load_clinical_trial_dataset("cart"),
        layer="log1p_norm",
        design_kw=dict(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response",
            arm_treated="CAR-T",
            arm_control="CAR-T",
        ),
        visits=("Pre", "Post"),
    ),
}


# ======================================================================
# Helpers
# ======================================================================

def _get_pseudobulk(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    """Return pseudobulk DataFrame: participant x visit x arm x features."""
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
# Panel A/B -- Pre vs Post scatter
# ======================================================================

def _panel_pre_post_scatter(ax, pb: pd.DataFrame, features: list[str],
                            cfg: dict, title: str):
    """Scatter of mean expression Pre vs Post, one point per feature.

    When the dynamic range spans >10x (e.g. AML with dominant LYZ),
    uses log1p-scaled axes so low-expression genes are readable.
    """
    if pb is None or len(pb) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    pre_v, post_v = cfg["visits"]
    avail = [f for f in features if f in pb.columns]

    pre_means = pb[pb["visit"] == pre_v][avail].mean()
    post_means = pb[pb["visit"] == post_v][avail].mean()

    # Detect and separate outlier genes (e.g. LYZ >> everything else)
    all_vals = np.concatenate([pre_means.values, post_means.values])
    sorted_vals = np.sort(all_vals)
    has_outlier = (sorted_vals[-1] / (sorted_vals[-3] + 1e-6)) > 5

    # Identify outlier genes and separate them
    outlier_feats = []
    main_feats = list(avail)
    if has_outlier:
        threshold = sorted_vals[-3] * 3
        for f in avail:
            if pre_means[f] > threshold or post_means[f] > threshold:
                outlier_feats.append(f)
        main_feats = [f for f in avail if f not in outlier_feats]

    # Plot main genes — zoom axes to their range for readability
    x_main = np.array([pre_means[f] for f in main_feats])
    y_main = np.array([post_means[f] for f in main_feats])

    ax.scatter(x_main, y_main, c=COLORS["highlight"],
               s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
               zorder=3)

    # Label main genes with adjustText
    texts = []
    for feat in main_feats:
        texts.append(ax.text(pre_means[feat], post_means[feat], feat,
                             fontsize=7, ha="left", va="bottom"))
    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    force_text=(2.0, 2.0), force_points=(2.0, 2.0),
                    expand=(1.5, 1.5))

    # Zoom axes to main gene cluster; annotate outliers as text note
    if outlier_feats:
        pad = max(x_main.ptp(), y_main.ptp()) * 0.15
        ax.set_xlim(x_main.min() - pad, x_main.max() + pad * 3)
        ax.set_ylim(y_main.min() - pad, y_main.max() + pad * 3)
        # Add text note for outlier genes off-plot
        out_strs = []
        for feat in outlier_feats:
            out_strs.append(
                f"{feat}: Pre={pre_means[feat]:.2f}, Post={post_means[feat]:.2f}"
            )
        ax.text(0.97, 0.03, "Outlier (off-plot):\n" + "\n".join(out_strs),
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(facecolor="#fff3cd", alpha=0.9, edgecolor="#ffc107",
                          linewidth=0.8, boxstyle="round,pad=0.4"))

    # Diagonal
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lim_lo = min(xlims[0], ylims[0])
    lim_hi = max(xlims[1], ylims[1])
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="gray",
            linewidth=0.8)

    r, p = sp_stats.pearsonr(pre_means.values, post_means.values)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                      linewidth=0.5, boxstyle="round,pad=0.3"))

    xlabel = f"Mean expression ({pre_v})"
    ylabel = f"Mean expression ({post_v})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel C/D -- Within-arm paired changes
# ======================================================================

def _panel_within_arm_change(ax, pb: pd.DataFrame, features: list[str],
                             cfg: dict, title: str, *,
                             note: str | None = None):
    """Paired Pre->Post changes for treated and control arms."""
    if pb is None or len(pb) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
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
                height=bar_height, color=COLORS["treated"], alpha=0.9,
                label=arm_treated, edgecolor="white", linewidth=0.5)
    if "Control" in changes:
        ax.barh(y_pos + bar_height / 2, changes["Control"][avail].values,
                height=bar_height, color=COLORS["control"], alpha=0.9,
                label=arm_control, edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avail, fontsize=8)
    ax.set_xlabel("Mean change (Post \u2212 Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)

    # Optional annotation note (e.g. missing arm info)
    if note is not None:
        ax.text(0.98, 0.02, note, transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom",
                fontstyle="italic", color=COLORS["gray"],
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    despine(ax)


def _panel_within_arm_change_multi(ax, results: dict, dataset_names: list[str],
                                   features: list[str], title: str):
    """Within-arm change for multiple datasets side-by-side.

    Each dataset gets its own group of bars. When only one arm is
    available the bar still renders with a note.
    """
    avail_global = features  # all requested features; per-dataset filtered below

    # Collect per-dataset changes
    all_changes: list[tuple[str, str, pd.Series]] = []  # (dataset, arm_label, series)
    notes: list[str] = []

    for ds_name in dataset_names:
        res = results.get(ds_name)
        if res is None:
            continue
        pb = res.get("pb")
        cfg = res["cfg"]
        if pb is None:
            continue

        pre_v, post_v = cfg["visits"]
        arm_treated = cfg["design_kw"]["arm_treated"]
        arm_control = cfg["design_kw"]["arm_control"]
        avail = [f for f in avail_global if f in pb.columns]

        # Deduplicate arms for single-arm datasets
        arms = [("Treated", arm_treated)]
        if arm_control != arm_treated:
            arms.append(("Control", arm_control))

        for arm_label, arm_val in arms:
            arm_pb = pb[pb["arm"] == arm_val]
            pre = arm_pb[arm_pb["visit"] == pre_v].set_index("participant_id")
            post = arm_pb[arm_pb["visit"] == post_v].set_index("participant_id")
            common = pre.index.intersection(post.index)
            if len(common) > 0:
                diff = post.loc[common, avail].mean() - pre.loc[common, avail].mean()
                # Pad missing features with NaN
                diff_full = pd.Series(np.nan, index=avail_global)
                for f in avail:
                    diff_full[f] = diff[f]
                display_label = f"{ds_name} ({arm_label})"
                all_changes.append((ds_name, display_label, diff_full))
            else:
                if arm_label == "Control":
                    notes.append(f"{ds_name}: no paired Control data")

    if len(all_changes) == 0:
        ax.text(0.5, 0.5, "No paired data", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    n_bars = len(all_changes)
    y_pos = np.arange(len(avail_global))
    bar_height = 0.7 / n_bars

    palette = [COLORS["treated"], COLORS["control"], COLORS["highlight"],
               COLORS["success"], COLORS["neutral"], COLORS["gray"]]

    for idx, (ds_name, label, series) in enumerate(all_changes):
        offset = (idx - n_bars / 2 + 0.5) * bar_height
        color = palette[idx % len(palette)]
        vals = series[avail_global].values
        ax.barh(y_pos + offset, vals, height=bar_height, color=color,
                alpha=0.9, label=label, edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avail_global, fontsize=8)
    ax.set_xlabel("Mean change (Post \u2212 Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9)

    if notes:
        note_text = "; ".join(notes)
        ax.text(0.98, 0.02, note_text, transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom",
                fontstyle="italic", color=COLORS["gray"],
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    despine(ax)


# ======================================================================
# Panel E/F -- Effect size vs baseline expression
# ======================================================================

def _panel_effect_vs_baseline(ax, pb: pd.DataFrame, features: list[str],
                              cfg: dict, title: str):
    """Scatter: effect vs Pre-treatment mean expression.

    Uses DiD when both arms have paired data.  Falls back to treated-arm
    Pre->Post change when the control arm lacks paired participants.
    """
    if pb is None or len(pb) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
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
    use_did = len(ctrl_paired) > 0 and arm_treated != arm_control

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

    # Detect and separate outlier genes by baseline expression
    sorted_bl = np.sort(baselines)
    has_outlier = (sorted_bl[-1] / (sorted_bl[-3] + 1e-6)) > 5

    outlier_idx = []
    main_idx = list(range(len(avail)))
    if has_outlier:
        bl_threshold = sorted_bl[-3] * 3
        outlier_idx = [i for i in range(len(avail)) if baselines[i] > bl_threshold]
        main_idx = [i for i in range(len(avail)) if i not in outlier_idx]

    # Plot main genes
    if main_idx:
        ax.scatter(baselines[main_idx], effects[main_idx],
                   c=COLORS["highlight"], s=60, alpha=0.8,
                   edgecolors="white", linewidths=0.5, zorder=3)

    # Label main genes with adjustText
    texts = []
    for i in main_idx:
        texts.append(ax.text(baselines[i], effects[i], avail[i], fontsize=7))
    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    force_text=(2.0, 2.0), force_points=(2.0, 2.0),
                    expand=(1.5, 1.5))

    # Zoom axes to main gene cluster; annotate outliers as text note
    if outlier_idx:
        main_bl = baselines[main_idx]
        main_eff = effects[main_idx]
        pad_x = max(main_bl.ptp(), 0.01) * 0.2
        pad_y = max(main_eff.ptp(), 0.01) * 0.2
        ax.set_xlim(main_bl.min() - pad_x, main_bl.max() + pad_x * 4)
        ax.set_ylim(main_eff.min() - pad_y * 2, main_eff.max() + pad_y * 2)
        # Annotate outlier genes as text note
        out_strs = []
        for i in outlier_idx:
            out_strs.append(
                f"{avail[i]}: baseline={baselines[i]:.2f}, "
                f"effect={effects[i]:.2f}"
            )
        ax.text(0.97, 0.03, "Outlier (off-plot):\n" + "\n".join(out_strs),
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                bbox=dict(facecolor="#fff3cd", alpha=0.9, edgecolor="#ffc107",
                          linewidth=0.8, boxstyle="round,pad=0.4"))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    mask = np.isfinite(baselines) & np.isfinite(effects)
    if mask.sum() > 3:
        r, p = sp_stats.pearsonr(baselines[mask], effects[mask])
        ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                          linewidth=0.5, boxstyle="round,pad=0.3"))

    xlabel = "Baseline expression (Pre mean)"
    ax.set_xlabel(xlabel)
    y_label = "DiD effect size" if use_did else "Treated Δ (Post − Pre)"
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    despine(ax)


# ======================================================================
# Panel G -- Cell count per participant x visit
# ======================================================================

def _panel_cell_counts(ax, adata, cfg: dict, title: str):
    """Distribution of cell counts per participant x visit."""
    pid_col = cfg["design_kw"]["participant_col"]
    visit_col = cfg["design_kw"]["visit_col"]
    arm_col = cfg["design_kw"]["arm_col"]
    arm_treated = cfg["design_kw"]["arm_treated"]
    arm_control = cfg["design_kw"]["arm_control"]

    counts = (adata.obs.groupby([pid_col, visit_col, arm_col], observed=True)
              .size().reset_index(name="n_cells"))
    counts.rename(columns={pid_col: "participant_id", visit_col: "visit",
                           arm_col: "arm"}, inplace=True)

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
            bp = ax.boxplot(
                sub["n_cells"], positions=[x_pos], widths=0.6,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor=color, markeredgecolor="none"),
                medianprops=dict(color="white", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.7, edgecolor=color),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
            )
            tick_positions.append(x_pos)
            arm_short = "T" if arm == arm_treated else "C"
            tick_labels.append(f"{visit}\n({arm_short})")
            x_pos += 1
        x_pos += 0.5  # gap between visits

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Cells per participant")
    ax.set_title(title, fontsize=11, fontweight="bold")

    handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.7, label=arm_treated),
        mpatches.Patch(color=COLORS["control"], alpha=0.7, label=arm_control),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel H -- Cross-dataset fold change comparison (3 datasets)
# ======================================================================

def _panel_cross_dataset_fc(ax, results: dict):
    """Compare Pre->Post log fold-change across all available datasets.

    Plots pairwise comparisons for all dataset pairs sharing features.
    With 3 datasets, shows the first two as x/y axes and overlays the
    third as a secondary comparison.
    """
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
        ax.text(0.5, 0.5, "Need 2+ datasets", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    names = list(fc_data.keys())
    # Primary comparison: first two datasets
    common = fc_data[names[0]].index.intersection(fc_data[names[1]].index)
    common = [f for f in common
              if np.isfinite(fc_data[names[0]][f])
              and np.isfinite(fc_data[names[1]][f])]
    if len(common) == 0:
        ax.text(0.5, 0.5, "No shared features", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    fc1 = np.array([fc_data[names[0]][f] for f in common])
    fc2 = np.array([fc_data[names[1]][f] for f in common])

    ax.scatter(fc1, fc2, c=COLORS["treated"], s=70, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"{names[0]} vs {names[1]}")

    # If third dataset available, overlay as second comparison (vs first)
    if len(names) >= 3:
        common3 = fc_data[names[0]].index.intersection(fc_data[names[2]].index)
        common3 = [f for f in common3
                   if np.isfinite(fc_data[names[0]][f])
                   and np.isfinite(fc_data[names[2]][f])]
        if len(common3) > 0:
            fc1_3 = np.array([fc_data[names[0]][f] for f in common3])
            fc3 = np.array([fc_data[names[2]][f] for f in common3])
            ax.scatter(fc1_3, fc3, c=COLORS["control"], s=70, alpha=0.85,
                       edgecolors="white", linewidths=0.5, zorder=3,
                       marker="D",
                       label=f"{names[0]} vs {names[2]}")

    # Use adjustText for gene labels
    texts = []
    for i, feat in enumerate(common):
        texts.append(ax.text(fc1[i], fc2[i], feat, fontsize=7))
    if len(names) >= 3 and len(common3) > 0:
        for i, feat in enumerate(common3):
            if feat not in common:  # avoid duplicate labels
                texts.append(ax.text(fc1_3[i], fc3[i], feat, fontsize=7))
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Reference lines
    all_vals = list(fc1) + list(fc2)
    if len(names) >= 3 and len(common3) > 0:
        all_vals += list(fc1_3) + list(fc3)
    lim = max(abs(v) for v in all_vals) * 1.2
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray",
            linewidth=1.0, alpha=0.6)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.3)
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.3)

    r, p = sp_stats.pearsonr(fc1, fc2)
    ax.text(0.05, 0.95, f"r = {r:.2f}\np = {p:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                      linewidth=0.5, boxstyle="round,pad=0.3"))

    ax.set_xlabel(f"log\u2082FC (treated, {names[0]})")
    ax.set_ylabel(f"log\u2082FC (treated, {names[1]})")
    ax.set_title("Cross-Dataset Fold Change (Treated Arm)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
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
        # A: Pre vs Post scatter -- Sade-Feldman
        ("panel_A", lambda fig, ax: _panel_pre_post_scatter(
            ax, sf.get("pb"), sf.get("features", []), sf["cfg"],
            "Pre vs Post Expression (Sade-Feldman)"), (7, 7)),

        # B: Pre vs Post scatter -- AML
        ("panel_B", lambda fig, ax: _panel_pre_post_scatter(
            ax, aml.get("pb"), aml.get("features", []), aml["cfg"],
            "Pre vs Post Expression (AML)"), (7, 7)),

        # C: Within-arm change -- Sade-Feldman (both arms)
        ("panel_C", lambda fig, ax: _panel_within_arm_change(
            ax, sf.get("pb"), sf.get("features", []), sf["cfg"],
            "Within-Arm Change (Sade-Feldman)"), (8, 7)),

        # D: Within-arm change -- AML + CAR-T combined
        ("panel_D", lambda fig, ax: _panel_within_arm_change_multi(
            ax, results, ["AML", "CAR-T"],
            [f for f in FEATURES
             if any(f in results.get(ds, {}).get("features", [])
                    for ds in ["AML", "CAR-T"])],
            "Within-Arm Change (AML + CAR-T)"), (9, 7)),

        # E: Effect vs baseline -- Sade-Feldman (DiD)
        ("panel_E", lambda fig, ax: _panel_effect_vs_baseline(
            ax, sf.get("pb"), sf.get("features", []), sf["cfg"],
            "Effect vs Baseline (Sade-Feldman)"), (7, 7)),

        # F: Effect vs baseline -- AML (Treated delta)
        ("panel_F", lambda fig, ax: _panel_effect_vs_baseline(
            ax, aml.get("pb"), aml.get("features", []), aml["cfg"],
            "Effect vs Baseline (AML)"), (7, 7)),

        # G: Cell counts -- Sade-Feldman
        ("panel_G", lambda fig, ax: _panel_cell_counts(
            ax, sf["adata"], sf["cfg"],
            "Cell Counts per Participant \u00d7 Visit (Sade-Feldman)"), (8, 6)),

        # H: Cross-dataset fold change (SF vs AML vs CAR-T)
        ("panel_H", lambda fig, ax: _panel_cross_dataset_fc(ax, results),
         (8, 7)),
    ]

    for panel_name, draw_fn, psize in panels:
        try:
            fig, ax = plt.subplots(figsize=psize)
            draw_fn(fig, ax)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)
        except Exception as exc:
            print(f"    {panel_name}: ERROR {exc}")
            import traceback
            traceback.print_exc()
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
