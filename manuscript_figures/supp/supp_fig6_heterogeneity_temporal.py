"""
Supplementary Figure 6 - Participant heterogeneity and temporal dynamics.

Panels:
  A  Sade-Feldman individual-effect strip.
  B  Sade-Feldman participant x feature heatmap.
  C  Sade-Feldman response-stratified boxplots.
  D  Sade-Feldman variance decomposition.
  E  Sade-Feldman direction-diversity profile.
  F  Cross-dataset SD bars.
  G  Cross-dataset heterogeneity scatter.
  H  Sade-Feldman within-arm change profile.
  I  AML + CAR-T within-arm profile.
  J  Cross-dataset treated-arm fold-change concordance.
"""

from __future__ import annotations

import gc

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
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    load_clinical_trial_dataset,
    save_panel,
)

FIGURE_NAME = "SuppFig6_heterogeneity_temporal"

FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19", "CD14", "LYZ", "NKG7", "IL7R",
]

_DATASET_CFG = {
    "Sade-Feldman": {
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
        "loader": lambda: load_clinical_trial_dataset("aml"),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_treated": "Treatment",
        "arm_control": "Control",
        "visits": ("Pre", "Post"),
    },
    "CAR-T": {
        "loader": lambda: load_clinical_trial_dataset("cart"),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_treated": "CAR-T",
        "arm_control": "CAR-T",
        "visits": ("Pre", "Post"),
    },
    "Stephenson": {
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
        "loader": get_vaccine,
        "harmonize": False,
        "layer": "log1p_cpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": None,
        "arm_treated": None,
        "arm_control": None,
        "visits": ("Pre", "Post"),
    },
}

_DS_COLORS = dict(zip(_DATASET_CFG.keys(),
    ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]))
_DS_MARKERS = {
    "Sade-Feldman": "o",
    "AML": "s",
    "CAR-T": "D",
    "Stephenson": "P",
    "Vaccine": "X",
}


def _to_array(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _participant_delta(adata, cfg: dict, features: list[str]) -> pd.DataFrame | None:
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg["arm_col"]
    pre_v, post_v = cfg["visits"]

    required = [pid_col, visit_col]
    if arm_col:
        required.append(arm_col)
    if not all(c in adata.obs.columns for c in required):
        return None
    if not features:
        return None

    X = _to_array(adata[:, features].layers[cfg["layer"]] if cfg["layer"] in adata.layers else adata[:, features].X)
    df = pd.DataFrame(X, columns=features, index=adata.obs_names)
    df[pid_col] = adata.obs[pid_col].values
    df[visit_col] = adata.obs[visit_col].values
    if arm_col and arm_col in adata.obs.columns:
        df[arm_col] = adata.obs[arm_col].values
    else:
        df["arm"] = "All"

    group_cols = [pid_col, visit_col]
    arm_key = arm_col if arm_col and arm_col in df.columns else "arm"
    group_cols.append(arm_key)
    pv = (
        df.groupby(group_cols, observed=True)[features]
        .mean()
        .reset_index()
    )
    pv = pv[pv[visit_col].isin([pre_v, post_v])].copy()

    pre = pv[pv[visit_col] == pre_v].set_index(pid_col)
    post = pv[pv[visit_col] == post_v].set_index(pid_col)
    common = pre.index.intersection(post.index)
    if len(common) < 3:
        return None

    delta = post.loc[common, features] - pre.loc[common, features]
    delta["arm"] = pre.loc[common, arm_key]
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


def _panel_strip(ax, effects: pd.DataFrame, features: list[str], treated: str, control: str, title: str):
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    feat_order = effects[features].mean().sort_values().index.tolist()
    long = effects.melt(id_vars=["participant_id", "arm"], value_vars=feat_order,
                        var_name="feature", value_name="effect")

    arm_colors = {
        treated: "#d62728",   # Strong red
        control: "#2ca02c",   # Strong green
    }
    if treated == control:
        arm_colors = {treated: "#d62728"}

    x_map = {f: i for i, f in enumerate(feat_order)}
    rng = np.random.default_rng(42)
    for arm, color in arm_colors.items():
        sub = long[long["arm"] == arm]
        if sub.empty:
            continue
        x = sub["feature"].map(x_map).values + rng.uniform(-0.2, 0.2, size=len(sub))
        ax.scatter(x, sub["effect"].values, s=25, alpha=0.65, c=color, edgecolors="none", label=arm)

    for feat, i in x_map.items():
        mu = long[long["feature"] == feat]["effect"].mean()
        ax.hlines(mu, i - 0.35, i + 0.35, color="black", lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Participant effect (Post - Pre)")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_heatmap(ax, effects: pd.DataFrame, features: list[str], title: str):
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    mat = effects.set_index("participant_id")[features].fillna(0)
    vmax = np.nanpercentile(np.abs(mat.values), 95)
    vmax = 1.0 if not np.isfinite(vmax) or vmax <= 0 else float(vmax)

    sns.heatmap(
        mat,
        ax=ax,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        cbar_kws={"label": "Effect (Post - Pre)", "shrink": 0.8},
        xticklabels=True,
        yticklabels=False,
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Participants")


def _panel_response_box(ax, effects: pd.DataFrame, features: list[str], treated: str, control: str, title: str):
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    avail = [f for f in features if f in effects.columns]
    top = effects[avail].std().sort_values(ascending=False).head(8).index.tolist()
    long = effects.melt(id_vars=["participant_id", "arm"], value_vars=top,
                        var_name="feature", value_name="effect")
    palette = {treated: COLORS["treated"], control: COLORS["control"]}
    if treated == control:
        palette = {treated: COLORS["treated"]}

    sns.boxplot(data=long, x="feature", y="effect", hue="arm", palette=palette,
                linewidth=0.8, fliersize=1.5, ax=ax)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_fontsize(8)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, frameon=True, title="Arm")
    despine(ax)


def _panel_variance_decomp(ax, effects: pd.DataFrame, features: list[str], title: str):
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
        ss_between = 0.0
        ss_within = 0.0
        for arm in arms:
            g = vals[vals["arm"] == arm][feat]
            if g.empty:
                continue
            gm = g.mean()
            ss_between += len(g) * (gm - grand) ** 2
            ss_within += float(((g - gm) ** 2).sum())
        tot = ss_between + ss_within
        if tot <= 0:
            continue
        rows.append({"feature": feat, "between": ss_between / tot, "within": ss_within / tot})

    if not rows:
        ax.text(0.5, 0.5, "No variance decomposition", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    df = pd.DataFrame(rows).sort_values("between", ascending=False)
    y = np.arange(len(df))
    ax.barh(y, df["between"], color=COLORS["highlight"], alpha=0.85, label="Between-arm")
    ax.barh(y, df["within"], left=df["between"], color=COLORS["gray"], alpha=0.8, label="Within-arm")
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of total variance")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    ax.invert_yaxis()
    despine(ax)


def _panel_direction_diversity(ax, effects: pd.DataFrame, features: list[str], title: str):
    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    rows = []
    for feat in features:
        vals = effects[feat].dropna()
        if len(vals) < 3:
            continue
        pos = (vals > 0).sum()
        neg = (vals < 0).sum()
        if pos + neg == 0:
            continue
        rows.append({"feature": feat, "pos": pos / (pos + neg), "neg": neg / (pos + neg)})

    if not rows:
        ax.text(0.5, 0.5, "No direction data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    df = pd.DataFrame(rows).sort_values("pos")
    y = np.arange(len(df))
    ax.barh(y, -df["neg"], color=COLORS["control"], alpha=0.85, label="Negative")
    ax.barh(y, df["pos"], color=COLORS["treated"], alpha=0.85, label="Positive")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Fraction of participants")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_sd_bars(ax, data: dict[str, dict]):
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
                color=_DS_COLORS.get(name, COLORS["gray"]), alpha=0.85, label=name)

    ax.set_yticks(y)
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_xlabel("SD of participant effects")
    ax.set_title("Cross-dataset heterogeneity magnitude", fontweight="bold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


def _panel_sd_scatter(ax, data: dict[str, dict]):
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

    ref = "Sade-Feldman" if "Sade-Feldman" in sd_map else list(sd_map.keys())[0]
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
        ax.scatter(x, y, s=45, alpha=0.85, marker=_DS_MARKERS.get(name, "o"),
                   color=_DS_COLORS.get(name, COLORS["neutral"]), edgecolors="white", linewidth=0.4, label=name)

    if max_lim <= 0:
        max_lim = 1.0
    lim = max_lim * 1.1
    ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"SD ({ref})")
    ax.set_ylabel("SD (other dataset)")
    ax.set_title("Cross-dataset SD concordance", fontweight="bold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


def _panel_within_arm_profile(ax, effects: pd.DataFrame, features: list[str], title: str):
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
        color = COLORS["treated"] if ("Resp" in str(arm) or "Treat" in str(arm) or "CAR" in str(arm)) else COLORS["control"]
        ax.errorbar(x, mu, yerr=1.96 * se, marker="o", lw=1.6, ms=4, color=color, label=str(arm))

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(feat_order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean participant effect")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


def _panel_aml_cart_profile(ax, data: dict[str, dict]):
    # Include all single-arm datasets (AML, CAR-T, Vaccine, etc.)
    datasets = [d for d in data if d != "Sade-Feldman" and d != "Stephenson"]
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
        treated = cfg["arm_treated"]
        sub = eff[eff["arm"] == treated] if treated in eff["arm"].values else eff
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
        treated = cfg["arm_treated"]
        sub = eff[eff["arm"] == treated] if treated in eff["arm"].values else eff
        mu = sub[top_feats].mean(axis=0).values
        se = sub[top_feats].std(axis=0, ddof=1).values / np.sqrt(max(sub.shape[0], 1))
        ax.errorbar(x, mu, yerr=1.96 * se, marker="o", lw=1.6, ms=4,
                    color=_DS_COLORS.get(ds_name, COLORS["neutral"]), label=ds_name)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(top_feats, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean participant effect")
    ax.set_title("Single-arm within-arm change profiles", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_treated_fc_concordance(ax, data: dict[str, dict]):
    treated_vectors = {}
    for name, ds in data.items():
        eff = ds.get("delta")
        feats = ds.get("features", [])
        if eff is None or eff.empty:
            continue
        cfg = ds["cfg"]
        t = cfg["arm_treated"]
        sub = eff[eff["arm"] == t] if t in eff["arm"].values else eff
        vec = sub[feats].mean(axis=0)
        treated_vectors[name] = vec

    if len(treated_vectors) < 2:
        ax.text(0.5, 0.5, "Need >=2 datasets", ha="center", va="center", transform=ax.transAxes)
        return

    names = list(treated_vectors)
    corr = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                corr.loc[a, b] = 1.0
                continue
            s1 = treated_vectors[a]
            s2 = treated_vectors[b]
            common = s1.index.intersection(s2.index)
            if len(common) < 3:
                corr.loc[a, b] = np.nan
            else:
                corr.loc[a, b] = s1[common].corr(s2[common])

    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Treated-arm fold-change concordance", fontweight="bold")


def generate():
    print("Supplementary Figure 6: Participant Heterogeneity and Temporal Dynamics")
    data = _load_all()
    if not data:
        print("  No data loaded; skipping.")
        return

    # Explicit dataset selection for reproducibility (CODEX #1).
    # Sade-Feldman is the flagship two-arm dataset (melanoma immunotherapy).
    best_name = "Sade-Feldman"
    if best_name not in data or data[best_name].get("delta") is None:
        # Fallback: pick first dataset with paired data
        best_name = next(
            (n for n, ds in data.items() if ds.get("delta") is not None),
            None,
        )
    if best_name is None:
        print("  No datasets with paired data; skipping.")
        return

    feat_ds = data[best_name]
    feat_eff = feat_ds.get("delta")
    feat_feats = feat_ds.get("features", FEATURES)
    feat_cfg = feat_ds.get("cfg", {})
    feat_treated = feat_cfg.get("arm_treated", "Treated")
    feat_control = feat_cfg.get("arm_control", "Control")

    # Two-arm dataset for panels C/D: Sade-Feldman (Responder vs Non-responder)
    twoarm_name = best_name  # same dataset — it's the two-arm reference
    ta_ds = data[twoarm_name]
    ta_eff = ta_ds.get("delta")
    ta_feats = ta_ds.get("features", FEATURES)
    ta_cfg = ta_ds.get("cfg", {})
    ta_treated = ta_cfg.get("arm_treated", "Treated")
    ta_control = ta_cfg.get("arm_control", "Control")

    panels = [
        ("panel_A", lambda ax: _panel_strip(ax, feat_eff, feat_feats, feat_treated, feat_control, f"{best_name} individual effects"), (11.5, 6.0)),
        ("panel_B", lambda ax: _panel_heatmap(ax, feat_eff, feat_feats, f"{best_name} participant x feature map"), (9.5, 6.8)),
        ("panel_C", lambda ax: _panel_response_box(ax, ta_eff, ta_feats, ta_treated, ta_control, f"{twoarm_name} response-stratified effects"), (10.5, 6.0)),
        ("panel_D", lambda ax: _panel_variance_decomp(ax, ta_eff, ta_feats, f"{twoarm_name} variance decomposition"), (8.2, 6.8)),
        ("panel_E", lambda ax: _panel_direction_diversity(ax, feat_eff, feat_feats, f"{best_name} effect direction diversity"), (8.2, 6.8)),
        ("panel_F", lambda ax: _panel_sd_bars(ax, data), (8.8, 6.8)),
        ("panel_G", lambda ax: _panel_sd_scatter(ax, data), (7.6, 6.8)),
        ("panel_H", lambda ax: _panel_within_arm_profile(ax, feat_eff, feat_feats, f"{best_name} within-arm change profile"), (10.2, 6.0)),
        ("panel_I", lambda ax: _panel_aml_cart_profile(ax, data), (10.2, 6.0)),
        ("panel_J", lambda ax: _panel_treated_fc_concordance(ax, data), (7.0, 6.2)),
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

    for ds in data.values():
        if "adata" in ds:
            del ds["adata"]
    data.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
