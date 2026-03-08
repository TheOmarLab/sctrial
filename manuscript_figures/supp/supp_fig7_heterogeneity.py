"""
Supplementary Figure 7 -- Participant-Level Heterogeneity Architecture
=====================================================================

Ten-panel figure (A--J) examining individual participant-level treatment
effect heterogeneity across the Sade-Feldman immunotherapy, AML,
CAR-T, and Melanoma clinical-trial datasets.

Panels
------
A  Strip plot of individual participant effects across features (SF).
B  Participant-level effect heatmap (participants x features, SF).
C  Leave-one-out influence scatter (Cook-style, SF).
D  Response-stratified box plots (top 6 most variable features, SF).
E  Individual effects strip plot (AML).
F  Variance decomposition: between-group vs within-group (SF).
G  Cross-dataset heterogeneity scatter (effect SD, multi-dataset).
H  Inter-feature correlation matrix of participant effects (SF).
I  Simpson diversity of effect directions per feature (SF).
J  Effect heterogeneity bars (SD by feature, all datasets).
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

# -- Figure-level constants ------------------------------------------------
FIGURE_NAME = "SuppFig7_heterogeneity"

FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19", "CD14", "LYZ", "NKG7",
]

# -- Dataset configs -------------------------------------------------------
_DATASET_CFGS = {
    "Sade-Feldman": dict(
        loader=get_sade_feldman,
        layer="log1p_tpm",
        harmonize=True,
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
        harmonize=False,
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
    "CAR-T": dict(
        loader=lambda: load_clinical_trial_dataset("cart"),
        layer="log1p_norm",
        harmonize=False,
        design_kw=dict(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response",
            arm_treated="CAR-T",
            arm_control="CAR-T",
        ),
        visits=("Pre", "Post"),
        response_col="response",
    ),
    "Melanoma": dict(
        loader=lambda: load_clinical_trial_dataset("melanoma"),
        layer="log1p_tpm",
        harmonize=False,
        design_kw=dict(
            participant_col="participant_id",
            visit_col="visit",
            arm_col="response",
            arm_treated="Post_Treatment",
            arm_control="Treatment_Naive",
        ),
        visits=("Pre", "Post"),
        response_col="response",
    ),
}

# Extra colours for multi-dataset panels
_DS_COLORS = {
    "Sade-Feldman": COLORS["treated"],
    "AML": COLORS["control"],
    "CAR-T": COLORS["success"],
    "Melanoma": COLORS["neutral"],
}
_DS_MARKERS = {
    "Sade-Feldman": "o",
    "AML": "s",
    "CAR-T": "D",
    "Melanoma": "^",
}


# ======================================================================
# Helpers
# ======================================================================

def _compute_individual_effects(
    adata, cfg: dict, features: list[str]
) -> pd.DataFrame | None:
    """Compute Post-Pre difference per participant for given features.

    For single-arm studies (e.g. CAR-T where arm_treated == arm_control),
    all participants are pooled and the column ``arm`` is set to the
    single arm label.
    """
    layer = cfg["layer"]
    pid_col = cfg["design_kw"]["participant_col"]
    visit_col = cfg["design_kw"]["visit_col"]
    arm_col = cfg["design_kw"]["arm_col"]
    pre_v, post_v = cfg["visits"]

    avail = [f for f in features if f in adata.var_names]
    if len(avail) < 3:
        return None

    obs = adata.obs.copy()
    if layer in adata.layers:
        raw = adata[:, avail].layers[layer]
        expr = pd.DataFrame(
            raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw),
            columns=avail,
            index=adata.obs_names,
        )
    else:
        raw = adata[:, avail].X
        expr = pd.DataFrame(
            raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw),
            columns=avail,
            index=adata.obs_names,
        )

    for c in [pid_col, visit_col, arm_col]:
        expr[c] = obs[c].values

    pb = (
        expr.groupby([pid_col, visit_col, arm_col], observed=True)[avail]
        .mean()
        .reset_index()
    )

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
    """Load all datasets and compute individual effects."""
    results = {}
    for name, cfg in _DATASET_CFGS.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)
            effects = _compute_individual_effects(adata, cfg, FEATURES)
            avail = [f for f in FEATURES if f in adata.var_names]
            results[name] = dict(
                adata=adata, effects=effects, features=avail, cfg=cfg
            )
            n_part = len(effects) if effects is not None else 0
            print(f"  {name}: {adata.n_obs} cells, {n_part} paired participants")
        except Exception as exc:
            print(f"  {name}: SKIP ({exc})")
    return results


# ======================================================================
# Panel A -- Strip plot of individual effects (SF)
# ======================================================================

def _panel_strip_effects(
    ax,
    effects: pd.DataFrame,
    features: list[str],
    title: str,
    arm_treated: str,
    arm_control: str,
):
    """Strip plot of individual participant effects across features."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No individual effects data",
                transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    df_long = effects.melt(
        id_vars=["participant_id", "arm"],
        value_vars=avail,
        var_name="feature",
        value_name="effect",
    )
    feat_order = (
        df_long.groupby("feature")["effect"].mean().sort_values().index.tolist()
    )

    is_single_arm = arm_treated == arm_control
    if is_single_arm:
        arm_colors = {arm_treated: COLORS["treated"]}
    else:
        arm_colors = {arm_treated: COLORS["treated"], arm_control: COLORS["control"]}

    # Only keep arms that actually have data (paired participants)
    arms_with_data = set(df_long["arm"].unique())
    arm_colors = {a: c for a, c in arm_colors.items() if a in arms_with_data}

    x_positions = {f: i for i, f in enumerate(feat_order)}
    rng = np.random.default_rng(42)

    for arm, color in arm_colors.items():
        sub = df_long[df_long["arm"] == arm]
        x_vals = sub["feature"].map(x_positions).values
        jitter = rng.uniform(-0.2, 0.2, size=len(sub))
        ax.scatter(
            x_vals + jitter,
            sub["effect"].values,
            c=color,
            alpha=0.55,
            s=22,
            edgecolors="none",
            label=arm,
            zorder=2,
        )

    for feat in feat_order:
        sub = df_long[df_long["feature"] == feat]
        mean_val = sub["effect"].mean()
        x = x_positions[feat]
        ax.hlines(
            mean_val, x - 0.35, x + 0.35,
            color="black", linewidth=2.0, zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Individual effect (Post \u2212 Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    handles = []
    for arm, color in arm_colors.items():
        handles.append(mpatches.Patch(color=color, alpha=0.6, label=arm))
    handles.append(Line2D([0], [0], color="black", linewidth=2, label="Mean"))
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel B -- Participant effect heatmap (SF)
# ======================================================================

def _panel_effect_heatmap(ax, effects: pd.DataFrame, features: list[str], title: str):
    """Heatmap of participant x feature effects with hierarchical clustering."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    mat = effects.set_index("participant_id")[avail]

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

    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        norm=norm,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Effect (Post \u2212 Pre)", "shrink": 0.8},
        xticklabels=True,
        yticklabels=False,
    )
    ax.set_ylabel("Participants (clustered)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)


# ======================================================================
# Panel C -- LOO influence scatter (SF)
# ======================================================================

def _panel_loo_influence(ax, effects: pd.DataFrame, features: list[str], title: str):
    """Cook-style LOO influence: how much each participant shifts the mean."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    mat = effects.set_index("participant_id")[avail].values
    n = mat.shape[0]
    if n < 4:
        ax.text(0.5, 0.5, "Too few participants",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    full_mean = np.nanmean(mat, axis=0)

    influence_scores = np.empty(n)
    for i in range(n):
        loo_mean = np.nanmean(np.delete(mat, i, axis=0), axis=0)
        diff = full_mean - loo_mean
        influence_scores[i] = np.sqrt(np.nansum(diff ** 2))

    pids = effects["participant_id"].values
    arms = effects["arm"].values
    threshold = np.mean(influence_scores) + 2 * np.std(influence_scores)

    for arm_val in np.unique(arms):
        mask = arms == arm_val
        color = (
            COLORS["treated"]
            if ("Resp" in str(arm_val) or "Treat" in str(arm_val))
            else COLORS["control"]
        )
        ax.scatter(
            np.where(mask)[0],
            influence_scores[mask],
            c=color,
            alpha=0.8,
            s=40,
            edgecolors="white",
            linewidths=0.3,
            label=str(arm_val),
        )

    ax.axhline(
        threshold, color="red", linewidth=0.8, linestyle="--",
        label=f"Threshold (\u03bc + 2\u03c3 = {threshold:.3f})",
    )

    for i in range(n):
        if influence_scores[i] > threshold:
            ax.annotate(
                str(pids[i])[:8],
                (i, influence_scores[i]),
                fontsize=7,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("Participant index")
    ax.set_ylabel("LOO influence score")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel D -- Response-stratified box plots (SF)
# ======================================================================

def _panel_response_boxes(
    ax,
    effects: pd.DataFrame,
    features: list[str],
    title: str,
    arm_treated: str,
    arm_control: str,
):
    """Box plots by response for top 6 most variable features."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    sds = effects[avail].std().sort_values(ascending=False)
    top = sds.head(6).index.tolist()

    df_long = effects.melt(
        id_vars=["participant_id", "arm"],
        value_vars=top,
        var_name="feature",
        value_name="effect",
    )

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
            ax.boxplot(
                data,
                positions=[x_positions[feat] + offset],
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker="o",
                    markersize=3,
                    alpha=0.4,
                    markerfacecolor=color,
                    markeredgecolor="none",
                ),
                medianprops=dict(color="white", linewidth=1.5),
                boxprops=dict(facecolor=color, alpha=0.8, edgecolor=color),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Individual effect (Post \u2212 Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")

    handles = [
        mpatches.Patch(color=COLORS["treated"], alpha=0.8, label=arm_treated),
        mpatches.Patch(color=COLORS["control"], alpha=0.8, label=arm_control),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel F -- Variance decomposition (between-group vs within-group, SF)
# ======================================================================

def _panel_variance_decomposition(ax, effects: pd.DataFrame, features: list[str],
                                  title: str, arm_treated: str, arm_control: str):
    """Stacked bar: fraction of total variance that is between-group vs within-group."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    # Need two actual groups
    unique_arms = effects["arm"].unique()
    if len(unique_arms) < 2:
        ax.text(0.5, 0.5, "Need two arms for decomposition",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.axis("off")
        return

    between_fracs = []
    within_fracs = []
    feat_labels = []

    for feat in avail:
        vals = effects[feat].dropna()
        if len(vals) < 4:
            continue
        total_var = vals.var(ddof=1)
        if total_var == 0 or np.isnan(total_var):
            continue

        grand_mean = vals.mean()
        ss_between = 0.0
        ss_within = 0.0
        for arm_val in unique_arms:
            group = effects.loc[effects["arm"] == arm_val, feat].dropna()
            n_g = len(group)
            if n_g == 0:
                continue
            g_mean = group.mean()
            ss_between += n_g * (g_mean - grand_mean) ** 2
            ss_within += ((group - g_mean) ** 2).sum()

        ss_total = ss_between + ss_within
        if ss_total == 0:
            continue
        frac_between = ss_between / ss_total
        frac_within = ss_within / ss_total
        between_fracs.append(frac_between)
        within_fracs.append(frac_within)
        feat_labels.append(feat)

    if len(feat_labels) == 0:
        ax.text(0.5, 0.5, "No variance data", transform=ax.transAxes,
                ha="center", va="center")
        ax.axis("off")
        return

    # Sort by between-group fraction descending
    order = np.argsort(between_fracs)[::-1]
    feat_labels = [feat_labels[i] for i in order]
    between_fracs = [between_fracs[i] for i in order]
    within_fracs = [within_fracs[i] for i in order]

    y_pos = np.arange(len(feat_labels))

    ax.barh(
        y_pos,
        between_fracs,
        height=0.7,
        color=COLORS["highlight"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Between-group",
    )
    ax.barh(
        y_pos,
        within_fracs,
        left=between_fracs,
        height=0.7,
        color=COLORS["gray"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Within-group",
    )

    # Annotate percentage on bars
    for i in range(len(feat_labels)):
        bfrac = between_fracs[i]
        if bfrac > 0.12:
            ax.text(
                bfrac / 2, y_pos[i], f"{bfrac:.0%}",
                ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_xlabel("Fraction of total variance")
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=True, framealpha=0.9)
    ax.invert_yaxis()
    despine(ax)


# ======================================================================
# Panel G -- Cross-dataset heterogeneity scatter (multi-dataset)
# ======================================================================

def _panel_cross_dataset_sd(ax, results: dict):
    """Scatter of effect SD across datasets for shared features.

    Uses Sade-Feldman as x-axis reference, plots all other datasets
    on y-axis.  If only 2 datasets exist, falls back to simple pairwise.
    """
    sd_data = {}
    for name, res in results.items():
        eff = res.get("effects")
        if eff is None:
            continue
        avail = [f for f in res["features"] if f in eff.columns]
        if len(avail) < 3:
            continue
        sd_data[name] = eff[avail].std()

    if len(sd_data) < 2:
        ax.text(0.5, 0.5, "Need >= 2 datasets",
                transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    ref_name = "Sade-Feldman" if "Sade-Feldman" in sd_data else list(sd_data.keys())[0]
    ref_sd = sd_data[ref_name]
    others = {k: v for k, v in sd_data.items() if k != ref_name}

    try:
        from adjustText import adjust_text
        _has_adjustText = True
    except ImportError:
        _has_adjustText = False

    all_texts = []
    global_max = 0.0
    for other_name, other_sd in others.items():
        common = ref_sd.index.intersection(other_sd.index)
        if len(common) == 0:
            continue
        s1 = ref_sd[common].values
        s2 = other_sd[common].values
        color = _DS_COLORS.get(other_name, COLORS["neutral"])
        marker = _DS_MARKERS.get(other_name, "o")
        ax.scatter(
            s1, s2,
            c=color,
            marker=marker,
            s=50,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            label=other_name,
            zorder=3,
        )
        # Collect text annotations for adjustText
        for i, feat in enumerate(common):
            txt = ax.text(
                s1[i], s2[i], feat,
                fontsize=7, ha="left", va="bottom", zorder=4,
            )
            all_texts.append(txt)
        local_max = max(s1.max(), s2.max())
        if local_max > global_max:
            global_max = local_max

        # Pearson r for each other dataset
        if len(common) >= 3:
            r, p = sp_stats.pearsonr(s1, s2)
            # Place text inside legend or annotation
            ax.scatter([], [], c="none", label=f"  r={r:.2f}, p={p:.3f}")

    # Repel overlapping labels
    if _has_adjustText and all_texts:
        adjust_text(
            all_texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
            expand=(1.3, 1.5),
            force_text=(0.8, 1.0),
            force_points=(0.5, 0.5),
        )

    lim = global_max * 1.15
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(f"SD of effects ({ref_name})")
    ax.set_ylabel("SD of effects (other datasets)")
    ax.set_title("Cross-Dataset Effect Heterogeneity", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel H -- Feature-feature correlation matrix (SF)
# ======================================================================

def _panel_effect_correlation(ax, effects: pd.DataFrame, features: list[str],
                              title: str):
    """Lower-triangle correlation matrix of per-participant effects."""
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    mat = effects[avail]
    corr = mat.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)


# ======================================================================
# Panel I -- Simpson diversity of effect directions (SF)
# ======================================================================

def _panel_effect_direction_diversity(ax, effects: pd.DataFrame,
                                      features: list[str], title: str):
    """Horizontal diverging bar: fraction of participants with positive vs
    negative effects for each feature, illustrating directional heterogeneity.
    """
    if effects is None or len(effects) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    pos_fracs = []
    neg_fracs = []
    feat_labels = []

    for feat in avail:
        vals = effects[feat].dropna()
        if len(vals) < 3:
            continue
        n_pos = (vals > 0).sum()
        n_neg = (vals < 0).sum()
        n_total = n_pos + n_neg
        if n_total == 0:
            continue
        feat_labels.append(feat)
        pos_fracs.append(n_pos / n_total)
        neg_fracs.append(n_neg / n_total)

    if len(feat_labels) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    # Sort by fraction positive (ascending = most negative-dominated on top)
    order = np.argsort(pos_fracs)
    feat_labels = [feat_labels[i] for i in order]
    pos_fracs = [pos_fracs[i] for i in order]
    neg_fracs = [neg_fracs[i] for i in order]

    y_pos = np.arange(len(feat_labels))
    bar_height = 0.7

    # Negative bars extend left (plotted as negative width from 0)
    ax.barh(
        y_pos,
        [-nf for nf in neg_fracs],
        height=bar_height,
        color=COLORS["control"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Negative (Post < Pre)",
    )
    # Positive bars extend right
    ax.barh(
        y_pos,
        pos_fracs,
        height=bar_height,
        color=COLORS["treated"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Positive (Post > Pre)",
    )

    # Annotate percentages
    for i in range(len(feat_labels)):
        pf = pos_fracs[i]
        nf = neg_fracs[i]
        if pf > 0.08:
            ax.text(pf / 2, y_pos[i], f"{pf:.0%}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")
        if nf > 0.08:
            ax.text(-nf / 2, y_pos[i], f"{nf:.0%}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_xlabel("Fraction of participants")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Panel J -- Effect heterogeneity bars (SD, all datasets)
# ======================================================================

def _panel_effect_sd_bars(ax, results: dict, title: str):
    """Grouped horizontal bar chart of SD of individual effects, all datasets."""
    sd_data = {}
    for name, res in results.items():
        eff = res.get("effects")
        if eff is None:
            continue
        avail = [f for f in res["features"] if f in eff.columns]
        if len(avail) < 3:
            continue
        sd_data[name] = eff[avail].std()

    if len(sd_data) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return

    names = list(sd_data.keys())
    # Union of features sorted by mean SD across datasets
    all_feats_set: set[str] = set()
    for s in sd_data.values():
        all_feats_set.update(s.index)

    def _mean_sd(f):
        vals = [sd_data[n].get(f, 0) for n in names if f in sd_data[n].index]
        return np.mean(vals) if vals else 0.0

    all_feats = sorted(all_feats_set, key=_mean_sd)

    y_pos = np.arange(len(all_feats))
    n_ds = len(names)
    bar_height = 0.8 / n_ds

    for i, name in enumerate(names):
        sds = sd_data[name]
        offset = -0.4 + bar_height / 2 + i * bar_height
        vals = [sds.get(f, 0) for f in all_feats]
        color = _DS_COLORS.get(name, COLORS["gray"])
        ax.barh(
            y_pos + offset,
            vals,
            height=bar_height,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=name,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_feats, fontsize=8)
    ax.set_xlabel("SD of individual effects (Post \u2212 Pre)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ======================================================================
# Generate
# ======================================================================

def generate():
    """Create and save Supplementary Figure 7 panels (A--J)."""
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
        # A: Individual effects strip (SF)
        (
            "panel_A",
            lambda fig, ax: _panel_strip_effects(
                ax, sf_effects, sf_feats,
                "Individual Effects (Sade-Feldman)",
                "Responder", "Non-responder",
            ),
            (12, 6),
        ),
        # B: Clustered heatmap (SF)
        (
            "panel_B",
            lambda fig, ax: _panel_effect_heatmap(
                ax, sf_effects, sf_feats,
                "Participant \u00d7 Feature Effects (SF)",
            ),
            (10, 7),
        ),
        # C: LOO influence (SF)
        (
            "panel_C",
            lambda fig, ax: _panel_loo_influence(
                ax, sf_effects, sf_feats,
                "LOO Influence (Sade-Feldman)",
            ),
            (8, 6),
        ),
        # D: Response-stratified boxes (SF)
        (
            "panel_D",
            lambda fig, ax: _panel_response_boxes(
                ax, sf_effects, sf_feats,
                "Effects by Response (Top 6, SF)",
                "Responder", "Non-responder",
            ),
            (8, 6),
        ),
        # E: Individual effects strip (AML)
        (
            "panel_E",
            lambda fig, ax: _panel_strip_effects(
                ax, aml_effects, aml_feats,
                "Individual Effects (AML)",
                "Treatment", "Control",
            ),
            (12, 6),
        ),
        # F: Variance decomposition (SF)
        (
            "panel_F",
            lambda fig, ax: _panel_variance_decomposition(
                ax, sf_effects, sf_feats,
                "Variance Decomposition (SF)",
                "Responder", "Non-responder",
            ),
            (8, 7),
        ),
        # G: Cross-dataset heterogeneity scatter
        (
            "panel_G",
            lambda fig, ax: _panel_cross_dataset_sd(ax, results),
            (8, 7),
        ),
        # H: Inter-feature correlation matrix (SF)
        (
            "panel_H",
            lambda fig, ax: _panel_effect_correlation(
                ax, sf_effects, sf_feats,
                "Effect Correlation Matrix (SF)",
            ),
            (8, 8),
        ),
        # I: Simpson diversity of effect directions (SF)
        (
            "panel_I",
            lambda fig, ax: _panel_effect_direction_diversity(
                ax, sf_effects, sf_feats,
                "Effect Direction Diversity (SF)",
            ),
            (8, 7),
        ),
        # J: Effect SD bars (all datasets)
        (
            "panel_J",
            lambda fig, ax: _panel_effect_sd_bars(
                ax, results,
                "Effect Heterogeneity (SD, All Datasets)",
            ),
            (8, 8),
        ),
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
