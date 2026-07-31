"""
Supplementary Figure 7 - Participant heterogeneity and temporal dynamics.

Panels:
  Row 1 — TNBC (Zhang):
  A  TNBC participant x feature heatmap.
  B  TNBC variance decomposition.
  C  TNBC direction-diversity profile.

  Row 2 — Melanoma (Sade-Feldman):
  D  Sade-Feldman participant x feature heatmap.
  E  Sade-Feldman variance decomposition.
  F  Sade-Feldman direction-diversity profile.

  Row 3 — Cross-dataset + temporal:
  G  Cross-dataset SD bars.
  H  Cross-dataset heterogeneity scatter.
  I  TNBC within-arm change profile (both arms).

  Row 4 — Single-arm + concordance:
  J  Single-arm datasets within-arm profile.
  K  Cross-dataset treated-arm fold-change concordance.
"""

from __future__ import annotations

import gc

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

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
)

FIGURE_NAME = "SuppFig7_heterogeneity_temporal"

FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19", "CD14", "LYZ", "NKG7", "IL7R",
]

_DATASET_CFG = {
    "Melanoma": {
        "design": "two_arm",
        "loader": get_sade_feldman,
        "harmonize": True,
        "layer": "log1p_tpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response_harmonized",
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
        "arm_treated": "Treatment",
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
        "arm_treated": "CAR-T",
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
        "arm_treated": None,
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

# Design-type label for legend annotations
_DESIGN_LABEL: dict[str, str] = {
    "Melanoma": "DiD",
    "COVID-19": "DiD",
    "TNBC": "DiD",
    "AML": "Δ",
    "CAR-T": "Δ",
    "Vaccine": "Δ",
}


def _ds_label(name: str) -> str:
    """Return dataset name with design-type suffix for legends."""
    tag = _DESIGN_LABEL.get(name, "")
    return f"{name} ({tag})" if tag else name

_DS_COLORS = dict(zip(_DATASET_CFG.keys(),
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#a6761d"]))
_DS_MARKERS = {
    "Melanoma": "o",
    "AML": "s",
    "CAR-T": "D",
    "COVID-19": "P",
    "Vaccine": "X",
    "TNBC": "^",
}


def _to_array(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _participant_delta(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    """Compute per-participant pre→post deltas.

    For single-arm datasets with ``arm_filter``, subset to that arm first
    so that participant IDs are unique per arm stratum.  Arm is carried
    through the groupby so that deltas are keyed by (participant, arm).
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
            features = [f for f in FEATURES if f in adata.var_names]
            delta = _participant_delta(adata, cfg, features)
            out[name] = {
                "adata": adata,
                "cfg": cfg,
                "features": features,
                "delta": delta,
            }
            n_paired = 0 if delta is None else delta["participant_id"].nunique()
            print(f"  {name}: {adata.n_obs} cells, {n_paired} paired participants")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return out


def _draw_heatmap_row_annotations(ax, pid_order, annotations, n_features, composite):
    """Colored row-annotation strips drawn to the left of a seaborn heatmap.

    Each annotation dict: {mapping: {pid: label}, palette: {label: color}, label: str}.
    Index-0 is the outermost (leftmost) strip.
    """
    strip_w = 0.35
    gap_between = 0.08
    gap_to_heatmap = 0.15
    n_strips = len(annotations)

    all_handles: list = []
    seen: set = set()

    for s_idx, ann in enumerate(annotations):
        # outermost strip (s_idx=0) sits furthest left
        x_right = -(gap_to_heatmap + s_idx * (strip_w + gap_between))
        x_left = x_right - strip_w
        mapping = ann["mapping"]
        palette = ann["palette"]

        for j, pid in enumerate(pid_order):
            lbl = mapping.get(pid)
            color = palette.get(lbl, "#DDDDDD") if lbl is not None else "#DDDDDD"
            ax.add_patch(plt.Rectangle(
                (x_left, j), strip_w, 1,
                facecolor=color, edgecolor="white",
                linewidth=0.2, zorder=4, clip_on=False,
            ))

        for lbl, color in palette.items():
            if lbl not in seen:
                all_handles.append(mpatches.Patch(facecolor=color, label=lbl, linewidth=0))
                seen.add(lbl)

    total_left = gap_to_heatmap + n_strips * strip_w + (n_strips - 1) * gap_between
    ax.set_xlim(-total_left, n_features)

    _leg_fs = 5 if composite else 6
    ax.legend(handles=all_handles, fontsize=_leg_fs, frameon=True,
              loc="lower center", ncol=2,
              handlelength=0.7, handleheight=0.7,
              borderpad=0.3, labelspacing=0.15,
              bbox_to_anchor=(0.5, 1.01),
              bbox_transform=ax.transAxes)


def _panel_heatmap(ax, effects: pd.DataFrame, features: list[str], title: str,
                   composite: bool = False, annotations=None):
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    # Sort rows by the first annotation that specifies a group_order
    if annotations:
        _sort_ann = next((a for a in annotations if a.get("group_order")), None)
        if _sort_ann:
            _mapping = _sort_ann["mapping"]
            _rank = {g: i for i, g in enumerate(_sort_ann["group_order"])}
            effects = effects.copy()
            effects["_sort"] = effects["participant_id"].map(
                lambda p: _rank.get(_mapping.get(p), len(_rank))
            )
            effects = effects.sort_values("_sort").drop(columns=["_sort"])

    # NaN masking instead of fillna(0) — zeros would fabricate absent data
    mat = effects.set_index("participant_id")[features]
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else float(vmax)

    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        mask=mat.isna(),
        cbar_kws={"label": "Effect (Post - Pre)", "shrink": 0.8},
        xticklabels=True,
        yticklabels=False,
        linewidths=0.2,
        linecolor="white",
    )
    _xtk_fs = 5 if composite else 8
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=_xtk_fs)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Participants")

    if annotations:
        _draw_heatmap_row_annotations(ax, mat.index.tolist(), annotations, len(features), composite)
    else:
        ax.yaxis.set_label_coords(-0.06, 0.5)


def _panel_variance_decomp(ax, effects: pd.DataFrame, features: list[str], title: str,
                           composite: bool = False):
    """One-way ANOVA decomposition: SS_between / SS_total per feature."""
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    arms = effects["arm"].dropna().unique().tolist()
    if len(arms) < 2:
        ax.text(0.5, 0.5, "Need two arms", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    rows = []
    for feat in features:
        vals = effects[["arm", feat]].dropna()
        if len(vals) < 4:
            continue
        grand = vals[feat].mean()
        ss_total = float(((vals[feat] - grand) ** 2).sum())
        ss_between = 0.0
        for arm in arms:
            g = vals[vals["arm"] == arm][feat]
            if g.empty:
                continue
            ss_between += len(g) * (g.mean() - grand) ** 2
        if ss_total <= 0:
            continue
        eta_sq = ss_between / ss_total
        rows.append({"feature": feat, "between": eta_sq, "within": 1.0 - eta_sq})

    if not rows:
        ax.text(0.5, 0.5, "No variance decomposition", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    df = pd.DataFrame(rows).sort_values("between", ascending=False)
    y = np.arange(len(df))
    ax.barh(y, df["between"], color=COLORS["highlight"], alpha=0.85, label="Between-arm (η²)")
    ax.barh(y, df["within"], left=df["between"], color=COLORS["gray"], alpha=0.8, label="Within-arm")
    ax.set_yticks(y)
    _xtk_fs = 5 if composite else 8
    ax.set_yticklabels(df["feature"], fontsize=_xtk_fs)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of total SS (η²)")
    ax.set_title(title, fontweight="bold")
    if composite:
        ax.legend(fontsize=5, frameon=True, loc="lower right", ncol=2)
    else:
        ax.legend(fontsize=8, frameon=True)
    ax.invert_yaxis()
    despine(ax)


def _panel_direction_diversity(ax, effects: pd.DataFrame, features: list[str],
                               title: str, *, effect_threshold: float = 0.05,
                               composite: bool = False, legend_loc: str | None = None):
    """Direction-diversity bar chart with effect-size threshold."""
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    rows = []
    for feat in features:
        vals = effects[feat].dropna()
        if len(vals) < 3:
            continue
        pos = (vals > effect_threshold).sum()
        neg = (vals < -effect_threshold).sum()
        if pos + neg == 0:
            continue
        rows.append({"feature": feat, "pos": pos / (pos + neg), "neg": neg / (pos + neg)})

    if not rows:
        ax.text(0.5, 0.5, "No direction data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    df = pd.DataFrame(rows)
    df["imbalance"] = (df["pos"] - df["neg"]).abs()
    df = df.sort_values("imbalance", ascending=True)
    y = np.arange(len(df))
    ax.barh(y, -df["neg"], color=COLORS["control"], alpha=0.85, label="Negative")
    ax.barh(y, df["pos"], color=COLORS["treated"], alpha=0.85, label="Positive")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    _ytk_fs = 5 if composite else 8
    ax.set_yticklabels(df["feature"], fontsize=_ytk_fs)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Fraction of participants")
    ax.set_title(title, fontweight="bold")
    _leg_fs = 5 if composite else 8
    _leg_loc = legend_loc if legend_loc else ("lower right" if composite else "best")
    ax.legend(fontsize=_leg_fs, frameon=True, loc=_leg_loc)
    despine(ax)


def _panel_sd_bars(ax, data: dict[str, dict], composite: bool = False):
    sd_map = {}
    for name, ds in data.items():
        eff = ds.get("delta")
        feats = ds.get("features", [])
        if eff is None or eff.empty:
            continue
        sd_map[name] = eff[feats].std(axis=0)

    if not sd_map:
        ax.text(0.5, 0.5, "No SD data", ha="center", va="center", transform=ax.transAxes)
        return

    all_feats = sorted(set().union(*[s.index.tolist() for s in sd_map.values()]))
    mean_sd = pd.Series({f: np.mean([sd_map[n].get(f, np.nan) for n in sd_map]) for f in all_feats}).dropna()
    top_feats = mean_sd.sort_values(ascending=False).head(12).index.tolist()

    y = np.arange(len(top_feats))
    width = 0.75 / len(sd_map)
    for i, name in enumerate(sd_map):
        vals = [sd_map[name].get(f, np.nan) for f in top_feats]
        ax.barh(y + i * width - 0.35 + width / 2, vals, height=width,
                color=_DS_COLORS.get(name, COLORS["gray"]), alpha=0.85,
                label=_ds_label(name))

    ax.set_yticks(y)
    _ytk_fs = 5 if composite else 8
    ax.set_yticklabels(top_feats, fontsize=_ytk_fs)
    ax.set_xlabel("SD of participant effects")
    ax.set_title("Cross-dataset heterogeneity magnitude", fontweight="bold")
    ax.legend(fontsize=5 if composite else 7, frameon=True)
    despine(ax)


def _panel_sd_scatter(ax, data: dict[str, dict], composite: bool = False):
    sd_map = {}
    for name, ds in data.items():
        eff = ds.get("delta")
        feats = ds.get("features", [])
        if eff is None or eff.empty:
            continue
        sd_map[name] = eff[feats].std(axis=0)

    if len(sd_map) < 2:
        ax.text(0.5, 0.5, "Need >=2 datasets", ha="center", va="center", transform=ax.transAxes)
        return

    ref = "Melanoma" if "Melanoma" in sd_map else list(sd_map.keys())[0]
    ref_sd = sd_map[ref]
    max_lim = 0.0

    for name, s in sd_map.items():
        if name == ref:
            continue
        common = ref_sd.index.intersection(s.index)
        if len(common) < 2:
            continue
        x = ref_sd[common].values
        y = s[common].values
        max_lim = max(max_lim, float(np.nanmax(np.r_[x, y])))
        _s = 18 if composite else 45
        _lw = 0.3 if composite else 0.4
        ax.scatter(x, y, s=_s, alpha=0.85, marker=_DS_MARKERS.get(name, "o"),
                   color=_DS_COLORS.get(name, COLORS["neutral"]), edgecolors="white",
                   linewidth=_lw, label=_ds_label(name))

    if max_lim <= 0:
        max_lim = 1.0
    lim = max_lim * 1.1
    ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"SD ({ref})")
    ax.set_ylabel("SD (other dataset)")
    ax.set_title("Cross-dataset SD concordance", fontweight="bold")
    if composite:
        ax.legend(fontsize=5, frameon=True, markerscale=1.0,
                  loc="upper left", ncol=1)
    else:
        ax.legend(fontsize=7, frameon=True)
    despine(ax)


def _panel_within_arm_profile(ax, effects: pd.DataFrame, features: list[str], title: str,
                              composite: bool = False):
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        return

    feat_order = effects[features].abs().mean().sort_values(ascending=False).head(12).index.tolist()
    x = np.arange(len(feat_order))

    for arm in effects["arm"].dropna().unique():
        sub = effects[effects["arm"] == arm][feat_order]
        if sub.empty:
            continue
        mu = sub.mean(axis=0).values
        se = sub.std(axis=0, ddof=1).values / np.sqrt(max(sub.shape[0], 1))
        color = COLORS["treated"] if ("Resp" in str(arm) or "Treat" in str(arm) or "CAR" in str(arm) or "PDL1" in str(arm)) else COLORS["control"]
        ax.errorbar(x, mu, yerr=1.96 * se, marker="o", lw=1.6, ms=4, color=color, label=str(arm))

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    _xtk_fs = 5 if composite else 8
    ax.set_xticklabels(feat_order, rotation=35, ha="right", fontsize=_xtk_fs)
    ax.set_ylabel("Mean participant effect")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=5 if composite else 7, frameon=True)
    despine(ax)


def _panel_aml_cart_profile(ax, data: dict[str, dict], composite: bool = False):
    # Include only single-arm datasets
    datasets = [d for d in data if _DATASET_CFG.get(d, {}).get("design") == "single_arm"]
    if not datasets:
        datasets = [d for d in ["AML", "CAR-T"] if d in data]
    if not datasets:
        ax.text(0.5, 0.5, "No single-arm data", ha="center", va="center", transform=ax.transAxes)
        return

    # shared top features by absolute effect
    score = {}
    for ds_name in datasets:
        ds = data[ds_name]
        eff = ds.get("delta")
        feats = ds.get("features", [])
        if eff is None or eff.empty:
            continue
        cfg = ds["cfg"]
        treated = cfg.get("arm_treated")
        sub = eff[eff["arm"] == treated] if treated and treated in eff["arm"].values else eff
        mu = sub[feats].mean(axis=0)
        for f in mu.index:
            score[f] = score.get(f, 0.0) + abs(float(mu[f]))

    if not score:
        ax.text(0.5, 0.5, "No within-arm profile data", ha="center", va="center", transform=ax.transAxes)
        return

    top_feats = [k for k, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:12]]
    x = np.arange(len(top_feats))

    for ds_name in datasets:
        ds = data[ds_name]
        eff = ds.get("delta")
        if eff is None or eff.empty:
            continue
        cfg = ds["cfg"]
        treated = cfg.get("arm_treated")
        sub = eff[eff["arm"] == treated] if treated and treated in eff["arm"].values else eff
        avail = [f for f in top_feats if f in sub.columns]
        if not avail:
            continue
        mu = sub[avail].mean(axis=0).reindex(top_feats).values
        se = sub[avail].std(axis=0, ddof=1).reindex(top_feats).values / np.sqrt(max(sub.shape[0], 1))
        ax.errorbar(x, mu, yerr=1.96 * se, marker="o", lw=1.6, ms=4,
                    color=_DS_COLORS.get(ds_name, COLORS["neutral"]),
                    label=_ds_label(ds_name))

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean participant effect")
    ax.set_title("Single-arm within-arm change profiles", fontweight="bold")
    ax.legend(fontsize=5 if composite else 7, frameon=True)
    despine(ax)


def _panel_treated_fc_concordance(ax, data: dict[str, dict]):
    treated_vectors = {}
    for name, ds in data.items():
        eff = ds.get("delta")
        feats = ds.get("features", [])
        if eff is None or eff.empty:
            continue
        cfg = ds["cfg"]
        t = cfg.get("arm_treated")
        sub = eff[eff["arm"] == t] if t and t in eff["arm"].values else eff
        vec = sub[feats].mean(axis=0)
        treated_vectors[name] = vec

    if len(treated_vectors) < 2:
        ax.text(0.5, 0.5, "Need >=2 datasets", ha="center", va="center", transform=ax.transAxes)
        return

    names = list(treated_vectors)
    labels = [_ds_label(n) for n in names]
    corr = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a, la in zip(names, labels):
        for b, lb in zip(names, labels):
            if a == b:
                corr.loc[la, lb] = 1.0
                continue
            s1 = treated_vectors[a]
            s2 = treated_vectors[b]
            common = s1.index.intersection(s2.index)
            if len(common) < 3:
                corr.loc[la, lb] = np.nan
            else:
                corr.loc[la, lb] = s1[common].corr(s2[common])

    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Treated-arm fold-change concordance", fontweight="bold")


def generate():
    print("Supplementary Figure 6: Participant Heterogeneity and Temporal Dynamics")
    data = _load_all()
    if not data:
        print("  No data loaded; skipping.")
        return

    # ── Melanoma (Sade-Feldman) — flagship two-arm reference ──────────
    mel_name = "Melanoma"
    if mel_name not in data or data[mel_name].get("delta") is None:
        mel_name = next(
            (n for n, ds in data.items() if ds.get("delta") is not None),
            None,
        )
    if mel_name is None:
        print("  No datasets with paired data; skipping.")
        return

    mel_ds = data[mel_name]
    mel_eff = mel_ds.get("delta")
    mel_feats = mel_ds.get("features", FEATURES)

    # ── TNBC (Zhang) — second two-arm dataset ─────────────────────────
    tnbc_name = "TNBC"
    tnbc_ds = data.get(tnbc_name, {})
    tnbc_eff = tnbc_ds.get("delta")
    tnbc_feats = tnbc_ds.get("features", FEATURES)
    # ── Annotation strips ─────────────────────────────────────────────
    # Panel A: one strip — Responder / Non-responder (arm IS response for Melanoma)
    mel_annotations = []
    if mel_eff is not None:
        mel_annotations = [
            {
                "mapping": dict(zip(mel_eff["participant_id"], mel_eff["arm"])),
                "palette": {
                    "Responder": "#2ECC71",       # green
                    "Non-responder": "#E91E8C",   # magenta
                },
                "label": "Response",
                "group_order": ["Responder", "Non-responder"],
            }
        ]

    # Panel D: two strips — treatment arm + response (R/NR from adata)
    tnbc_annotations = []
    if tnbc_eff is not None:
        _tnbc_resp_map: dict = {}
        _tnbc_ad = tnbc_ds.get("adata")
        if _tnbc_ad is not None and "response" in _tnbc_ad.obs.columns:
            _tnbc_resp_map = (
                _tnbc_ad.obs.groupby("participant_id")["response"].first().to_dict()
            )
        tnbc_annotations = [
            {
                "mapping": dict(zip(tnbc_eff["participant_id"], tnbc_eff["arm"])),
                "palette": {
                    "anti-PDL1+Chemo": "#9B59B6",  # purple
                    "Chemo": "#F39C12",             # amber
                },
                "label": "Arm",
                "group_order": ["Chemo", "anti-PDL1+Chemo"],
            },
            {
                "mapping": _tnbc_resp_map,
                "palette": {
                    "R": "#2ECC71",    # green
                    "NR": "#E91E8C",   # magenta
                },
                "label": "Response",
            },
        ]

    panels = [
        # Row 1 — TNBC
        ("panel_A", lambda ax: _panel_heatmap(ax, tnbc_eff, tnbc_feats, f"{tnbc_name} participant × feature map", annotations=tnbc_annotations), (9.5, 6.8)),
        ("panel_B", lambda ax: _panel_variance_decomp(ax, tnbc_eff, tnbc_feats, f"{tnbc_name} variance decomposition"), (8.2, 6.8)),
        ("panel_C", lambda ax: _panel_direction_diversity(ax, tnbc_eff, tnbc_feats, f"{tnbc_name} effect direction diversity", legend_loc="upper right"), (8.2, 6.8)),
        # Row 2 — Melanoma
        ("panel_D", lambda ax: _panel_heatmap(ax, mel_eff, mel_feats, f"{mel_name} participant × feature map", annotations=mel_annotations), (9.5, 6.8)),
        ("panel_E", lambda ax: _panel_variance_decomp(ax, mel_eff, mel_feats, f"{mel_name} variance decomposition"), (8.2, 6.8)),
        ("panel_F", lambda ax: _panel_direction_diversity(ax, mel_eff, mel_feats, f"{mel_name} effect direction diversity"), (8.2, 6.8)),
        # Row 3 — Cross-dataset
        ("panel_G", lambda ax: _panel_sd_bars(ax, data), (8.8, 6.8)),
        ("panel_H", lambda ax: _panel_sd_scatter(ax, data), (7.6, 6.8)),
        # Row 4 — Temporal
        ("panel_I", lambda ax: _panel_within_arm_profile(ax, tnbc_eff, tnbc_feats, f"{tnbc_name} within-arm change profile"), (10.2, 6.0)),
        ("panel_J", lambda ax: _panel_aml_cart_profile(ax, data), (10.2, 6.0)),
        # Row 5
        ("panel_K", lambda ax: _panel_treated_fc_concordance(ax, data), (7.0, 6.2)),
    ]

    for panel_name, fn, size in panels:
        try:
            fig, ax = plt.subplots(figsize=size)
            fn(ax)
            fig.tight_layout()
            save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)
        except Exception as exc:
            print(f"    {panel_name}: failed ({exc})")
            plt.close("all")

    # ==================================================================
    # Composite artboard  (180 mm × ≤ 270 mm)
    # ==================================================================
    #   Row 0: A | B | C   (Melanoma: heatmap | variance | direction)
    #   Row 2: D | E | F   (TNBC:     heatmap | variance | direction)
    #   Row 4: G | H | I   (SD bars | SD scatter | within-arm profile)
    #   Row 6: J | K       (single-arm profile | FC concordance)
    # ==================================================================
    print("  Building composite figure ...")

    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": 5,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "legend.fontsize": 4,
        "legend.title_fontsize": 4,
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
    fig_c = plt.figure(figsize=(180 * _mm, 200 * _mm))

    # 7 rows: 4 content rows interleaved with 3 spacer rows
    outer = fig_c.add_gridspec(
        7, 1,
        height_ratios=[
            0.50,   # row 0: A | B | C  (Melanoma)
            0.18,   # spacer
            0.43,   # row 2: D | E | F  (TNBC)
            0.18,   # spacer
            0.50,   # row 4: G | H | I  (cross-dataset + within-arm)
            0.18,   # spacer
            0.50,   # row 6: J | K      (single-arm + FC concordance)
        ],
        hspace=0.0,
        left=0.06, right=0.98, top=0.97, bottom=0.04,
    )

    _HEAT_SHRINK_A = 0.80   # TNBC heatmap (two annotation strips need more space above)
    _HEAT_SHRINK_D = 0.90   # Melanoma heatmap (one annotation strip)

    # ── Row 0: A | B | C  (TNBC) ─────────────────────────────────────
    gs0 = outer[0].subgridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.50)
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])
    ax_c = fig_c.add_subplot(gs0[2])

    _panel_heatmap(ax_a, tnbc_eff, tnbc_feats, f"{tnbc_name} participant × feature map",
                   composite=True, annotations=tnbc_annotations)
    # Shrink ax_a from the top only, freeing space for the legend above the heatmap
    _pos = ax_a.get_position()
    ax_a.set_position([_pos.x0, _pos.y0, _pos.width, _pos.height * _HEAT_SHRINK_A])
    # Re-issue set_title with explicit y so it lands at the original row top
    ax_a.set_title(ax_a.get_title(), fontweight="bold", y=1.30, pad=0)
    # Align colorbar to shrunken heatmap bounds
    try:
        _cbar_ax = ax_a.collections[0].colorbar.ax
        _cp = _cbar_ax.get_position()
        _np = ax_a.get_position()
        _cbar_ax.set_position([_cp.x0, _np.y0, _cp.width, _np.height])
    except (IndexError, AttributeError):
        pass
    _panel_variance_decomp(ax_b, tnbc_eff, tnbc_feats,
                           f"{tnbc_name} variance decomposition", composite=True)
    _panel_direction_diversity(ax_c, tnbc_eff, tnbc_feats,
                               f"{tnbc_name} effect direction diversity", composite=True,
                               legend_loc="upper right")

    # ── Row 2: D | E | F  (Melanoma) ─────────────────────────────────
    gs1 = outer[2].subgridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.50)
    ax_d = fig_c.add_subplot(gs1[0])
    ax_e = fig_c.add_subplot(gs1[1])
    ax_f = fig_c.add_subplot(gs1[2])

    _panel_heatmap(ax_d, mel_eff, mel_feats, f"{mel_name} participant × feature map",
                   composite=True, annotations=mel_annotations)
    # Shrink ax_d from the top only, freeing space for the legend above the heatmap
    _pos = ax_d.get_position()
    ax_d.set_position([_pos.x0, _pos.y0, _pos.width, _pos.height * _HEAT_SHRINK_D])
    ax_d.set_title(ax_d.get_title(), fontweight="bold", y=1.16, pad=0)
    # Align colorbar to shrunken heatmap bounds
    try:
        _cbar_ax = ax_d.collections[0].colorbar.ax
        _cp = _cbar_ax.get_position()
        _np = ax_d.get_position()
        _cbar_ax.set_position([_cp.x0, _np.y0, _cp.width, _np.height])
    except (IndexError, AttributeError):
        pass
    _panel_variance_decomp(ax_e, mel_eff, mel_feats,
                           f"{mel_name} variance decomposition", composite=True)
    _panel_direction_diversity(ax_f, mel_eff, mel_feats,
                               f"{mel_name} effect direction diversity", composite=True)

    # ── Row 4: G | H | I  (cross-dataset + within-arm) ───────────────
    gs2 = outer[4].subgridspec(1, 3, width_ratios=[1.1, 1.0, 1.1], wspace=0.50)
    ax_g = fig_c.add_subplot(gs2[0])
    ax_h = fig_c.add_subplot(gs2[1])
    ax_i = fig_c.add_subplot(gs2[2])

    _panel_sd_bars(ax_g, data, composite=True)
    _panel_sd_scatter(ax_h, data, composite=True)
    _panel_within_arm_profile(ax_i, tnbc_eff, tnbc_feats,
                              f"{tnbc_name} within-arm change profile", composite=True)

    # ── Row 6: J | K  (single-arm profile + FC concordance) ──────────
    gs3 = outer[6].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.45)
    ax_j = fig_c.add_subplot(gs3[0])
    ax_k = fig_c.add_subplot(gs3[1])

    _panel_aml_cart_profile(ax_j, data, composite=True)
    _panel_treated_fc_concordance(ax_k, data)

    # ── Post-processing ───────────────────────────────────────────────
    for ax_pp in fig_c.get_axes():
        leg = ax_pp.get_legend()
        if leg:
            leg.get_frame().set_alpha(0.85)
            leg.get_frame().set_edgecolor("#CCCCCC")

    _cap_fontsize(fig_c, _MAX_FONT)

    # Bold panel labels — consistent offset
    _lbl_fs = 9
    _lbl_x = -0.10
    _lbl_y = 1.12
    _lbl_y_heat_a = _lbl_y / _HEAT_SHRINK_A
    _lbl_y_heat_d = _lbl_y / _HEAT_SHRINK_D

    for ax_lbl, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C"),
        (ax_d, "D"), (ax_e, "E"), (ax_f, "F"),
        (ax_g, "G"), (ax_h, "H"), (ax_i, "I"),
        (ax_j, "J"), (ax_k, "K"),
    ]:
        if ax_lbl is ax_a:
            _y = _lbl_y_heat_a
        elif ax_lbl is ax_d:
            _y = _lbl_y_heat_d
        else:
            _y = _lbl_y
        ax_lbl.text(
            _lbl_x, _y, lbl,
            transform=ax_lbl.transAxes,
            fontsize=_lbl_fs, fontweight="bold", va="top", ha="left",
        )

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, SUPP_OUTPUT, close=False)
    pdf_path = SUPP_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print("    Saved combined artboard (PNG + PDF)")

    # ── Cleanup ───────────────────────────────────────────────────────
    for ds in data.values():
        if "adata" in ds:
            del ds["adata"]
    data.clear()
    clear_cache()
    gc.collect()
    print("  SuppFig7 complete: 11 individual panels + combined (A–K)\n")


if __name__ == "__main__":
    apply_style()
    generate()
