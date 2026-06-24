"""
Figure 5 — Validation, Heterogeneity & Temporal Dynamics
=========================================================

Ten-panel figure combining permutation validation on Sade-Feldman,
participant-level heterogeneity analysis (Sade-Feldman and TNBC),
and temporal severity dynamics from the Stephenson COVID-19 cohort.

Panels
------
A  Permutation null distributions (top 3 most significant signatures).
B  Observed effects vs 95 % null range for all signatures.
C  Individual-effect strip plot (Sade-Feldman).
D  Response-stratified heterogeneity boxplots (Sade-Feldman).
E  Individual-effect strip plot (TNBC), colored by treatment arm; responders
   marked with black outline, non-responders with grey outline.
F  Response- and arm-stratified heterogeneity boxplots (TNBC): 4 groups per
   gene (Chemo-Responder, Chemo-Non-responder, anti-PDL1+Chemo-Responder,
   anti-PDL1+Chemo-Non-responder).
G  Signature trajectories by severity (Stephenson, DFO bins).
H  Severity divergence over DFO bins (4 representative signatures).
I  Temporal divergence heatmap (all signatures × DFO bins).
J  Time-specific Hedges' g effect sizes (all signatures × DFO bins).
"""

from __future__ import annotations

import gc
import hashlib
import pickle  # noqa: S403 — local dev cache of our own DataFrames
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    TrialDesign,
    add_log1p_cpm_layer,
    apply_style,
    despine,
    did_table,
    get_sade_feldman,
    get_stephenson,
    get_tnbc_zhang,
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure5_validation_dynamics"
VISITS: tuple[str, str] = ("Pre", "Post")
N_PERM = 999

_CACHE_DIR = Path(__file__).resolve().parent.parent / "_cache"
_CACHE_DIR.mkdir(exist_ok=True)

DESIGN = TrialDesign(
    participant_col="participant_id",
    visit_col="visit",
    arm_col="response_harmonized",
    arm_treated="Responder",
    arm_control="Non-responder",
)

# Temporal (Stephenson) constants
COL_SEVERE = COLORS["control"]   # orange
COL_MILD = COLORS["treated"]     # blue
DFO_BINS = ["DFO_0-7", "DFO_8-14", "DFO_15+"]
DFO_LABELS = ["0–7 d", "8–14 d", "15+ d"]

# Gene features for heterogeneity panels
HETERO_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7", "IL7R",
]

# TNBC arm labels and visual encoding
TNBC_ARM_CHEMO = "Chemo"
TNBC_ARM_COMBO = "anti-PDL1+Chemo"
TNBC_ARM_COLORS = {
    TNBC_ARM_CHEMO: COLORS["control"],   # orange
    TNBC_ARM_COMBO: COLORS["treated"],   # blue
}
# Outline color encodes response status
TNBC_RESP_EDGE = {"Responder": "black", "Non-responder": "none"}
# Four-group colors for boxplots (full and lighter tints per arm)
TNBC_GROUP_PALETTE = {
    f"{TNBC_ARM_CHEMO} – Responder":     COLORS["control"],   # orange
    f"{TNBC_ARM_CHEMO} – Non-responder": "#F0B97A",           # light orange
    f"{TNBC_ARM_COMBO} – Responder":     COLORS["treated"],   # blue
    f"{TNBC_ARM_COMBO} – Non-responder": "#9AB3D6",           # light blue
}


# ── data preparation ─────────────────────────────────────────────────────

def _to_array(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _prepare_sf_data(*, use_cache: bool = True) -> dict:
    """Load Sade-Feldman, run permutation test and compute participant deltas.

    The permutation results and participant DiD are cached to disk because the
    999-permutation loop takes several minutes.  Pass ``use_cache=False`` to
    force recomputation (e.g. after changing N_PERM or DESIGN).
    """
    _code_hash = hashlib.md5(  # noqa: S324 — cache tag, not security
        f"{N_PERM}|{DESIGN}|{VISITS}|{HETERO_FEATURES}".encode()
    ).hexdigest()[:8]
    cache_key = f"figure6_sf_perm_v1_{_code_hash}"
    cache_path = _CACHE_DIR / f"{cache_key}.pkl"

    # ── load adata fresh every time (too large to pickle) ────────────────
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        if "tpm" in adata.layers:
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        else:
            raise RuntimeError("No tpm layer for log1p_tpm creation.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")
    hetero_feats = [f for f in HETERO_FEATURES if f in adata.var_names]

    # ── try to restore expensive results from cache ───────────────────────
    if use_cache and cache_path.exists():
        print(f"  Loading cached SF permutation results from {cache_path.name}")
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)  # noqa: S301 — trusted local cache
        cached["adata"] = adata
        cached["sig_cols"] = sig_cols
        cached["hetero_feats"] = hetero_feats
        return cached

    # ── compute from scratch ──────────────────────────────────────────────
    common_kw = dict(
        features=sig_cols,
        design=DESIGN,
        visits=VISITS,
        layer="log1p_tpm",
        standardize=True,
    )

    # Participant-level DiD
    print("  Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)

    # Permutation test
    print(f"  Running {N_PERM} permutations ...")
    np.random.seed(42)
    arm_col = DESIGN.arm_col
    original_arm = adata.obs[arm_col].copy()
    perm_results: list[pd.DataFrame] = []

    for i in range(N_PERM):
        pid_col = DESIGN.participant_col
        pid_arm = adata.obs.groupby(pid_col, observed=True)[arm_col].first()
        shuffled = pid_arm.sample(frac=1, replace=False)
        shuffled.index = pid_arm.index
        adata.obs[arm_col] = adata.obs[pid_col].map(shuffled)
        try:
            df_perm = did_table(
                adata, aggregate="participant_visit", **common_kw,
            )
            df_perm["permutation"] = i
            perm_results.append(df_perm)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"    permutation {i + 1}/{N_PERM}")

    adata.obs[arm_col] = original_arm
    df_perm_all = pd.concat(perm_results, ignore_index=True)
    print(f"  Completed {df_perm_all['permutation'].nunique()} permutations")

    # Participant-level deltas for heterogeneity panels
    delta_df = _compute_participant_delta(adata, hetero_feats)

    result = {
        "df_part": df_part,
        "df_perm_all": df_perm_all,
        "delta_df": delta_df,
    }

    # ── persist to disk (adata excluded — too large) ──────────────────────
    if use_cache:
        with open(cache_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Cached SF permutation results to {cache_path.name}")

    result["adata"] = adata
    result["sig_cols"] = sig_cols
    result["hetero_feats"] = hetero_feats
    return result


def _compute_participant_delta(
    adata, features: list[str],
) -> pd.DataFrame | None:
    """Compute per-participant pre→post change for gene-level features."""
    pid_col = DESIGN.participant_col
    visit_col = DESIGN.visit_col
    arm_col = DESIGN.arm_col
    pre_v, post_v = VISITS

    if not features:
        return None

    layer = "log1p_tpm"
    X = _to_array(
        adata[:, features].layers[layer]
        if layer in adata.layers
        else adata[:, features].X
    )
    df = pd.DataFrame(X, columns=features, index=adata.obs_names)
    df[pid_col] = adata.obs[pid_col].values
    df[visit_col] = adata.obs[visit_col].values
    df[arm_col] = adata.obs[arm_col].values

    pv = (
        df.groupby([pid_col, visit_col, arm_col], observed=True)[features]
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
    delta["arm"] = pre.loc[common, arm_col]
    delta = delta.reset_index().rename(columns={pid_col: "participant_id"})
    return delta


def _compute_tnbc_participant_delta(
    adata, features: list[str], layer: str | None,
) -> pd.DataFrame | None:
    """Per-participant pre→post delta for TNBC, retaining arm and response."""
    pid_col = "participant_id"
    visit_col = "visit"
    arm_col = "arm"
    response_col = "response_harmonized"
    pre_v, post_v = VISITS

    if not features:
        return None

    X = _to_array(
        adata[:, features].layers[layer]
        if layer and layer in adata.layers
        else adata[:, features].X
    )
    df = pd.DataFrame(X, columns=features, index=adata.obs_names)
    df[pid_col] = adata.obs[pid_col].values
    df[visit_col] = adata.obs[visit_col].values
    df[arm_col] = adata.obs[arm_col].values
    df[response_col] = adata.obs[response_col].values

    pv = (
        df.groupby([pid_col, visit_col, arm_col, response_col], observed=True)[features]
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
    delta["arm"] = pre.loc[common, arm_col]
    delta["response"] = pre.loc[common, response_col]
    delta = delta.reset_index().rename(columns={pid_col: "participant_id"})
    return delta


def _prepare_tnbc_data() -> dict:
    """Load TNBC (Zhang), harmonize response, compute participant deltas."""
    adata = get_tnbc_zhang()
    adata = harmonize_response(adata)
    layer = "log1p_norm" if "log1p_norm" in adata.layers else None

    hetero_feats = [f for f in HETERO_FEATURES if f in adata.var_names]
    delta_df = _compute_tnbc_participant_delta(adata, hetero_feats, layer)

    return {
        "adata": adata,
        "delta_df": delta_df,
        "hetero_feats": hetero_feats,
    }


def _prepare_stephenson_data() -> dict:
    """Load Stephenson COVID-19 cohort and score signatures for temporal panels."""
    adata = get_stephenson()

    if "log1p_cpm" not in adata.layers:
        if "counts" in adata.layers:
            adata = add_log1p_cpm_layer(
                adata, counts_layer="counts", out_layer="log1p_cpm",
            )
        else:
            raise RuntimeError("No counts layer for log1p_cpm creation.")

    adata, sig_cols = score_signatures(adata, layer="log1p_cpm")

    # Keep only cells in DFO bins with severity info
    mask = (
        adata.obs["dfo_bin"].isin(DFO_BINS)
        & adata.obs["severity"].isin(["Mild", "Severe"])
    )
    adata_sub = adata[mask].copy()
    print(f"  Stephenson cells with DFO bin + severity: {adata_sub.n_obs:,}")

    # Per-participant-bin pseudobulk means
    grp_cols = ["participant_id", "dfo_bin", "severity"]
    pb = (
        adata_sub.obs[grp_cols + sig_cols]
        .groupby(grp_cols, observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    return {
        "adata": adata,
        "sig_cols": sig_cols,
        "pb": pb,
    }


# ── Panel A: Permutation null distributions ──────────────────────────────

def _panel_a(ax, data: dict) -> None:
    """Permutation null distributions overlaid for top 3 signatures."""
    df_part = data["df_part"]
    df_perm = data["df_perm_all"]
    sig_cols = data["sig_cols"]

    perm_pvals = {}
    for feat in sig_cols:
        null_betas = df_perm.loc[df_perm["feature"] == feat, "beta_DiD"].dropna()
        if len(null_betas) == 0:
            continue
        obs_row = df_part.loc[df_part["feature"] == feat]
        if obs_row.empty:
            continue
        obs_beta = obs_row["beta_DiD"].values[0]
        perm_p = (
            (np.sum(np.abs(null_betas) >= np.abs(obs_beta)) + 1)
            / (len(null_betas) + 1)
        )
        perm_pvals[feat] = perm_p

    if not perm_pvals:
        ax.text(0.5, 0.5, "No permutation results",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    top_feats = sorted(perm_pvals, key=perm_pvals.get)[:3]
    hist_colors = [COLORS["treated"], COLORS["control"], COLORS["neutral"]]

    for idx, feat in enumerate(top_feats):
        obs_beta = df_part.loc[df_part["feature"] == feat, "beta_DiD"].values[0]
        null_betas = df_perm.loc[df_perm["feature"] == feat, "beta_DiD"].dropna()
        color = hist_colors[idx % len(hist_colors)]

        ax.hist(
            null_betas, bins=25, color=color, alpha=0.35,
            edgecolor="none", density=True, zorder=2,
            label=f"{sig_display(feat)} null",
        )
        ax.axvline(obs_beta, color=color, lw=2, ls="-", zorder=4)

    ylim = ax.get_ylim()
    y_top = ylim[1]
    for idx, feat in enumerate(top_feats):
        perm_p = perm_pvals[feat]
        obs_beta = df_part.loc[df_part["feature"] == feat, "beta_DiD"].values[0]
        color = hist_colors[idx % len(hist_colors)]
        y_label = y_top * (0.93 - idx * 0.15)
        ax.text(
            obs_beta, y_label, f"    p = {perm_p:.3f}",
            fontsize=7, color=color, ha="left", fontweight="bold",
        )

    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (null distribution)")
    ax.set_ylabel("Density")
    ax.set_title("Permutation Null Distributions (Top 3)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel B: Observed effects vs null range ──────────────────────────────

def _panel_b(ax, data: dict) -> None:
    """Observed β_DiD with 95 % null interval for every signature."""
    df_part = data["df_part"]
    df_perm = data["df_perm_all"]
    sig_cols = data["sig_cols"]

    records = []
    for feat in sig_cols:
        obs_row = df_part.loc[df_part["feature"] == feat]
        if obs_row.empty:
            continue
        obs_beta = obs_row["beta_DiD"].values[0]
        null_betas = df_perm.loc[df_perm["feature"] == feat, "beta_DiD"].dropna()
        if len(null_betas) < 10:
            continue
        lo, hi = np.percentile(null_betas, [2.5, 97.5])
        perm_p = (
            (np.sum(np.abs(null_betas) >= np.abs(obs_beta)) + 1)
            / (len(null_betas) + 1)
        )
        records.append({
            "feature": feat,
            "display": sig_display(feat),
            "obs_beta": obs_beta,
            "null_lo": lo,
            "null_hi": hi,
            "significant": (obs_beta < lo) or (obs_beta > hi),
            "perm_p": perm_p,
        })

    rec_df = pd.DataFrame(records).sort_values("obs_beta", ascending=True)
    y_pos = np.arange(len(rec_df))

    for i, (_, row) in enumerate(rec_df.iterrows()):
        ax.barh(
            i, row["null_hi"] - row["null_lo"],
            left=row["null_lo"], height=0.6,
            color=COLORS["gray"], alpha=0.25, edgecolor="none", zorder=1,
        )

    colors = [
        COLORS["highlight"] if sig else COLORS["treated"]
        for sig in rec_df["significant"]
    ]
    ax.scatter(
        rec_df["obs_beta"], y_pos, c=colors, s=30,
        edgecolor="white", linewidth=0.5, zorder=3,
    )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rec_df["display"].values, fontsize=8)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title("Observed Effects vs Null Range", fontsize=10, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["highlight"], markersize=4,
               label="Outside 95% null"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["treated"], markersize=4,
               label="Within 95% null"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["gray"], alpha=0.25,
                       label="95% null range"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: Individual-effect strip plot ────────────────────────────────

def _panel_c(ax, data: dict) -> None:
    """Strip plot of per-participant effects for Sade-Feldman."""
    effects = data["delta_df"]
    features = data["hetero_feats"]

    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    avail = [f for f in features if f in effects.columns]
    feat_order = effects[avail].mean().sort_values().index.tolist()
    long = effects.melt(
        id_vars=["participant_id", "arm"], value_vars=feat_order,
        var_name="feature", value_name="effect",
    )

    arm_colors = {
        DESIGN.arm_treated: COLORS["treated"],
        DESIGN.arm_control: COLORS["control"],
    }

    x_map = {f: i for i, f in enumerate(feat_order)}
    rng = np.random.default_rng(42)
    for arm, color in arm_colors.items():
        sub = long[long["arm"] == arm]
        if sub.empty:
            continue
        x = sub["feature"].map(x_map).values + rng.uniform(-0.2, 0.2, len(sub))
        ax.scatter(
            x, sub["effect"].values, s=10, alpha=0.55,
            c=color, edgecolors="none", label=arm,
        )

    for feat, i in x_map.items():
        mu = long[long["feature"] == feat]["effect"].mean()
        ax.hlines(mu, i - 0.35, i + 0.35, color="black", lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Participant Effect (Post − Pre)")
    ax.set_title("Individual Treatment Effects - Melanoma", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: Response-stratified heterogeneity boxplots ──────────────────

def _panel_d(ax, data: dict) -> None:
    """Boxplots of participant effects stratified by response arm."""
    effects = data["delta_df"]
    features = data["hetero_feats"]

    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No paired effects",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    avail = [f for f in features if f in effects.columns]
    top = effects[avail].std().sort_values(ascending=False).head(8).index.tolist()
    long = effects.melt(
        id_vars=["participant_id", "arm"], value_vars=top,
        var_name="feature", value_name="effect",
    )
    palette = {
        DESIGN.arm_treated: COLORS["treated"],
        DESIGN.arm_control: COLORS["control"],
    }

    sns.boxplot(
        data=long, x="feature", y="effect", hue="arm", palette=palette,
        linewidth=0.8, fliersize=1.5, ax=ax,
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_fontsize(8)
    for i, feat in enumerate(top):
        t_vals = effects.loc[effects["arm"] == DESIGN.arm_treated, feat].dropna().values
        c_vals = effects.loc[effects["arm"] == DESIGN.arm_control, feat].dropna().values
        g = _hedges_g(t_vals, c_vals)
        if np.isfinite(g):
            ymax = long.loc[long["feature"] == feat, "effect"].max()
            ax.text(i, ymax + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"g={g:.2f}", ha="center", va="bottom", fontsize=6.5,
                    fontstyle="italic", color="grey")
    ax.set_xlabel("")
    ax.set_ylabel("Participant Effect (Post − Pre)")
    ax.set_title(
        "Response-Stratified Heterogeneity - Melanoma\n(g: Hedges' g, Responder vs Non-responder)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7, frameon=True, title="Arm")
    despine(ax)


# ── Panel E: TNBC individual-effect strip plot ───────────────────────────

def _panel_e(ax, tnbc_data: dict) -> None:
    """Strip plot of per-participant pre→post effects for TNBC.

    Markers are colored by treatment arm. Responders carry a black outline;
    non-responders have no outline.
    """
    effects = tnbc_data["delta_df"]
    features = tnbc_data["hetero_feats"]

    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No TNBC paired effects",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    avail = [f for f in features if f in effects.columns]
    # Use SF-derived order when available (matches panel C), else fall back to TNBC mean
    feat_order = effects[avail].mean().sort_values().index.tolist()
    long = effects.melt(
        id_vars=["participant_id", "arm", "response"], value_vars=feat_order,
        var_name="feature", value_name="effect",
    )

    x_map = {f: i for i, f in enumerate(feat_order)}
    rng = np.random.default_rng(42)

    for arm, arm_color in TNBC_ARM_COLORS.items():
        for resp, edge_color in TNBC_RESP_EDGE.items():
            sub = long[(long["arm"] == arm) & (long["response"] == resp)]
            if sub.empty:
                continue
            x = sub["feature"].map(x_map).values + rng.uniform(-0.2, 0.2, len(sub))
            lw = 0.8 if edge_color != "none" else 0.0
            ax.scatter(
                x, sub["effect"].values, s=9, alpha=0.75,
                c=arm_color, edgecolors=edge_color, linewidths=lw,
                zorder=3,
            )

    for feat, i in x_map.items():
        mu = long[long["feature"] == feat]["effect"].mean()
        ax.hlines(mu, i - 0.35, i + 0.35, color="black", lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Participant Effect (Post − Pre)")
    ax.set_title("Individual Treatment Effects - TNBC", fontsize=10, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=TNBC_ARM_COLORS[TNBC_ARM_CHEMO],
               markeredgecolor="none", markersize=4, label=TNBC_ARM_CHEMO),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=TNBC_ARM_COLORS[TNBC_ARM_COMBO],
               markeredgecolor="none", markersize=4, label=TNBC_ARM_COMBO),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#BBBBBB",
               markeredgecolor="black", markeredgewidth=0.8, markersize=4,
               label="Responder"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#BBBBBB",
               markeredgecolor="none", markersize=4,
               label="Non-responder"),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=True, framealpha=0.9,
              loc="upper left", ncol=2)
    despine(ax)


# ── Panel F: TNBC response- and arm-stratified boxplots ──────────────────

def _panel_f(ax, tnbc_data: dict) -> None:
    """4-group per gene: boxplot when n≥2, diamond marker when n=1.

    Groups: Chemo-R, Chemo-NR (n=1 → marker), anti-PDL1+Chemo-R, anti-PDL1+Chemo-NR.
    """
    effects = tnbc_data["delta_df"]
    features = tnbc_data["hetero_feats"]

    if effects is None or effects.empty:
        ax.text(0.5, 0.5, "No TNBC paired effects",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    avail = [f for f in features if f in effects.columns]
    # Use SF-derived top-gene order when available (matches panel D)
    top = effects[avail].std().sort_values(ascending=False).head(8).index.tolist()

    effects = effects.copy()
    effects["group"] = (
        effects["arm"].astype(str) + " – "
        + effects["response"].astype(str)
    )

    long = effects.melt(
        id_vars=["participant_id", "arm", "response", "group"], value_vars=top,
        var_name="feature", value_name="effect",
    )

    group_order = [g for g in TNBC_GROUP_PALETTE if g in long["group"].unique()]
    palette_map = {g: TNBC_GROUP_PALETTE[g] for g in group_order}

    n_groups = len(group_order)
    n_feats = len(top)
    box_w = 0.16
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * box_w

    # Separate data into box-eligible (n≥2) and single-sample (n=1)
    box_positions, box_data, box_colors = [], [], []
    single_xs, single_ys, single_colors = [], [], []

    for feat_idx, feat in enumerate(top):
        feat_data = long[long["feature"] == feat]
        for grp_idx, grp in enumerate(group_order):
            vals = feat_data[feat_data["group"] == grp]["effect"].dropna().values
            color = palette_map[grp]
            x_pos = feat_idx + offsets[grp_idx]
            if len(vals) == 0:
                continue
            elif len(vals) == 1:
                single_xs.append(x_pos)
                single_ys.append(vals[0])
                single_colors.append(color)
            else:
                box_positions.append(x_pos)
                box_data.append(vals)
                box_colors.append(color)

    # Draw boxes
    if box_data:
        bp = ax.boxplot(
            box_data, positions=box_positions, widths=box_w * 0.85,
            patch_artist=True, notch=False, showfliers=False,
            medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(linewidth=0.6),
            capprops=dict(linewidth=0.6),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_linewidth(0.6)

    # Draw diamond markers for single-sample groups
    if single_xs:
        ax.scatter(
            single_xs, single_ys, marker="D", s=12,
            c=single_colors, edgecolors="white", linewidths=0.3, zorder=4,
        )

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(n_feats))
    ax.set_xticklabels(top, rotation=35, ha="right", fontsize=8)
    ax.set_xlim(-0.5, n_feats - 0.5)
    ylo, yhi = ax.get_ylim()
    # Hedges' g annotations: Chemo vs anti-PDL1+Chemo (pooled across response)
    for feat_idx, feat in enumerate(top):
        chemo_vals = effects.loc[effects["arm"] == TNBC_ARM_CHEMO, feat].dropna().values
        combo_vals = effects.loc[effects["arm"] == TNBC_ARM_COMBO, feat].dropna().values
        g = _hedges_g(chemo_vals, combo_vals)
        if np.isfinite(g):
            feat_data = long.loc[long["feature"] == feat, "effect"].dropna()
            ymax_feat = float(np.nanpercentile(feat_data, 90)) if not feat_data.empty else yhi
            ax.text(feat_idx, ymax_feat + 0.12 * (yhi - ylo),
                    f"g={g:.2f}", ha="center", va="bottom", fontsize=6.5,
                    fontstyle="italic", color="grey")
    ax.set_ylim(ylo - 0.25 * (yhi - ylo), yhi + 0.25 * (yhi - ylo))
    ax.set_ylabel("Participant Effect (Post − Pre)")
    ax.set_title(
        "Response-Stratified Heterogeneity - TNBC\n(Hedges' g: Chemo vs anti-PDL1+Chemo)",
        fontsize=10, fontweight="bold",
    )

    # Legend: box patch for multi-sample groups, diamond for single-sample
    grp_ns = {g: effects[effects["group"] == g].shape[0] for g in group_order}
    legend_handles = []
    for grp in group_order:
        color = palette_map[grp]
        if grp_ns[grp] <= 1:
            legend_handles.append(
                Line2D([0], [0], marker="D", color="w",
                       markerfacecolor=color, markeredgecolor="none",
                       markersize=4, label=grp)
            )
        else:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.85, label=grp)
            )

    ax.legend(handles=legend_handles, fontsize=7, frameon=True,
              loc="lower right", ncol=2)
    despine(ax)


# ── Panel G: Temporal trajectories by severity ───────────────────────────

def _panel_g(ax, steph_data: dict) -> None:
    """Line plot of mean signature score over DFO bins for Mild vs Severe."""
    pb = steph_data["pb"]
    sig_cols = steph_data["sig_cols"]

    targets = []
    for keyword in ["Cytotoxic", "Interferon", "Memory", "Antigen"]:
        for col in sig_cols:
            if keyword in col:
                targets.append(col)
                break
    if len(targets) < 4:
        targets = sig_cols[:4]

    markers = ["o", "s", "^", "D"]
    for idx, col in enumerate(targets):
        for sev, color, ls in [("Severe", COL_SEVERE, "-"), ("Mild", COL_MILD, "--")]:
            means = []
            for b in DFO_BINS:
                vals = pb.loc[
                    (pb["severity"] == sev) & (pb["dfo_bin"] == b), col
                ]
                means.append(vals.mean() if len(vals) > 0 else np.nan)
            ax.plot(
                range(len(DFO_BINS)), means,
                color=color, ls=ls, lw=1.8, marker=markers[idx],
                markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                alpha=0.85,
            )
            if not np.isnan(means[-1]) and sev == "Mild":
                ax.annotate(
                    sig_display(col),
                    (len(DFO_BINS) - 1, means[-1]),
                    fontsize=6, ha="left", va="center",
                    xytext=(6, -2), textcoords="offset points",
                    color=color,
                )

    ax.set_xticks(range(len(DFO_BINS)))
    ax.set_xticklabels(DFO_LABELS)
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("Mean Signature Score")
    ax.set_title("Signature Trajectories by Severity - COVID-19", fontsize=10, fontweight="bold")

    handles = [
        Line2D([0], [0], color=COL_SEVERE, lw=2, ls="-", label="Severe"),
        Line2D([0], [0], color=COL_MILD, lw=2, ls="--", label="Mild"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel H: Severity divergence ─────────────────────────────────────────

def _panel_h(ax, steph_data: dict) -> None:
    """Line plot of (Severe − Mild) mean per DFO bin for 4 signatures."""
    pb = steph_data["pb"]
    sig_cols = steph_data["sig_cols"]

    targets = []
    for keyword in ["Cytotoxic", "Interferon", "Memory", "Antigen"]:
        for col in sig_cols:
            if keyword in col:
                targets.append(col)
                break
    if len(targets) < 4:
        targets = sig_cols[:4]

    palette = [COLORS["highlight"], COLORS["neutral"],
               COLORS["success"], COLORS["treated"]]
    markers = ["o", "s", "^", "D"]

    for idx, col in enumerate(targets):
        divs = []
        for b in DFO_BINS:
            sev_vals = pb.loc[
                (pb["severity"] == "Severe") & (pb["dfo_bin"] == b), col
            ]
            mild_vals = pb.loc[
                (pb["severity"] == "Mild") & (pb["dfo_bin"] == b), col
            ]
            divs.append(sev_vals.mean() - mild_vals.mean())

        ax.plot(
            range(len(DFO_BINS)), divs,
            color=palette[idx], lw=2, marker=markers[idx],
            markersize=6, markeredgecolor="white", markeredgewidth=0.5,
            label=sig_display(col),
        )

    ax.axhline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xticks(range(len(DFO_BINS)))
    ax.set_xticklabels(DFO_LABELS)
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("Divergence (Severe − Mild)")
    ax.set_title("Severity Divergence Over Time - COVID-19", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel I: Temporal divergence heatmap ─────────────────────────────────

def _panel_i(ax, steph_data: dict) -> None:
    """Heatmap of severity divergence (Severe − Mild) × DFO bin."""
    pb = steph_data["pb"]
    sig_cols = steph_data["sig_cols"]

    records = []
    for col in sig_cols:
        for b in DFO_BINS:
            sev_mean = pb.loc[
                (pb["severity"] == "Severe") & (pb["dfo_bin"] == b), col
            ].mean()
            mild_mean = pb.loc[
                (pb["severity"] == "Mild") & (pb["dfo_bin"] == b), col
            ].mean()
            records.append({
                "display": sig_display(col),
                "dfo_bin": b,
                "divergence": sev_mean - mild_mean,
            })

    df = pd.DataFrame(records)
    pivot = df.pivot(index="display", columns="dfo_bin", values="divergence")
    pivot = pivot.reindex(columns=DFO_BINS)
    pivot.columns = DFO_LABELS

    # Cluster rows by similarity
    vals = pivot.fillna(0).values
    if vals.shape[0] > 2:
        link = linkage(vals, method="ward")
        order = leaves_list(link)
        pivot = pivot.iloc[order]

    vmax = np.nanmax(np.abs(pivot.values)) * 0.9
    sns.heatmap(
        pivot, ax=ax, cmap="RdBu_r", center=0,
        vmin=-vmax, vmax=vmax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Divergence (Severe − Mild)", "shrink": 0.8},
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
    )
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("")
    ax.set_title("Temporal Divergence Heatmap - COVID-19", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)


# ── Panel J: Time-specific Hedges' g ────────────────────────────────────

def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g  (x − y)  with small-sample correction."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled == 0:
        return np.nan
    g = (np.mean(x) - np.mean(y)) / pooled
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return g * correction


def _panel_j(ax, steph_data: dict) -> None:
    """Grouped horizontal bar chart of Hedges' g per signature × DFO bin."""
    pb = steph_data["pb"]
    sig_cols = steph_data["sig_cols"]

    records = []
    for col in sig_cols:
        for b in DFO_BINS:
            sev = pb.loc[
                (pb["severity"] == "Severe") & (pb["dfo_bin"] == b), col
            ].dropna().values
            mild = pb.loc[
                (pb["severity"] == "Mild") & (pb["dfo_bin"] == b), col
            ].dropna().values
            g = _hedges_g(sev, mild)
            records.append({
                "feature": col,
                "display": sig_display(col),
                "dfo_bin": b,
                "g": g,
            })

    df = pd.DataFrame(records)
    pivot = df.pivot(index="display", columns="dfo_bin", values="g")
    pivot = pivot.reindex(columns=DFO_BINS)

    # Sort by absolute mean effect
    pivot["abs_mean"] = pivot.abs().mean(axis=1)
    pivot = pivot.sort_values("abs_mean", ascending=True).drop(columns="abs_mean")

    y_pos = np.arange(len(pivot))
    n_bins = len(DFO_BINS)
    bar_h = 0.8 / n_bins
    bin_colors = [COLORS["treated"], COLORS["neutral"], COLORS["highlight"]]

    for i, (b, color) in enumerate(zip(DFO_BINS, bin_colors)):
        offset = (i - (n_bins - 1) / 2) * bar_h
        vals = pivot[b].values
        ax.barh(
            y_pos + offset, vals,
            height=bar_h, color=color, alpha=0.8,
            label=DFO_LABELS[i], edgecolor="none",
        )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Hedges' g (Severe − Mild)")
    ax.set_title("Time-Specific Effect Sizes - COVID-19", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Composite generation ────────────────────────────────────────────────

def generate() -> None:
    """Create and save all Figure 6 panels."""
    apply_style()
    print("Figure 5: Validation, Heterogeneity & Temporal Dynamics")

    # Sade-Feldman data (panels A–D) — permutation results cached to disk
    sf_data = _prepare_sf_data()

    # TNBC data (panels E–F)
    tnbc_data = _prepare_tnbc_data()

    # Stephenson data (panels G–J)
    steph_data = _prepare_stephenson_data()

    # Panels A–D: Sade-Feldman
    sf_panels = [
        ("panel_A_permutation_null", _panel_a, (6.5, 5)),
        ("panel_B_observed_vs_null", _panel_b, (6.5, 5)),
        ("panel_C_individual_effects", _panel_c, (11.5, 6.0)),
        ("panel_D_response_stratified", _panel_d, (10.5, 6.0)),
    ]
    for panel_name, func, size in sf_panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax, sf_data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # Panels E–F: TNBC
    tnbc_panels = [
        ("panel_E_tnbc_individual_effects", _panel_e, (11.5, 6.0)),
        ("panel_F_tnbc_response_stratified", _panel_f, (10.5, 6.0)),
    ]
    for panel_name, func, size in tnbc_panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax, tnbc_data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # Panels G–J: Stephenson
    steph_panels = [
        ("panel_G_temporal_trajectories", _panel_g, (7, 5.5)),
        ("panel_H_severity_divergence", _panel_h, (7, 5.5)),
        ("panel_I_temporal_heatmap", _panel_i, (7, 5.5)),
        ("panel_J_time_specific_effect_sizes", _panel_j, (7, 5.5)),
    ]
    for panel_name, func, size in steph_panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax, steph_data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # ── Combined artboard (180 × 260 mm) ─────────────────────────────────
    _SMALL_RC = {
        "font.size": 5,
        "axes.titlesize": 5.5,
        "axes.labelsize": 5,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "legend.fontsize": 4,
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
    fig_c = plt.figure(figsize=(180 * _mm, 215 * _mm))

    #   Row 0: A | B     (permutation)
    #   Row 1: C | D     (SF heterogeneity)
    #   Row 2: E | F     (TNBC heterogeneity)
    #   Row 3: G | H     (temporal lines)
    #   Row 4: I | J     (heatmap + bar)
    outer = fig_c.add_gridspec(
        5, 1,
        height_ratios=[1, 1, 1, 1, 1],
        hspace=0.70,
        left=0.10, right=0.95, top=0.97, bottom=0.05,
    )

    gs0 = outer[0].subgridspec(1, 2, wspace=0.45)
    ax_a = fig_c.add_subplot(gs0[0])
    ax_b = fig_c.add_subplot(gs0[1])

    gs1 = outer[1].subgridspec(1, 2, wspace=0.45, width_ratios=[1, 1.2])
    ax_cc = fig_c.add_subplot(gs1[0])
    ax_d = fig_c.add_subplot(gs1[1])

    gs2 = outer[2].subgridspec(1, 2, wspace=0.45, width_ratios=[1, 1.2])
    ax_e = fig_c.add_subplot(gs2[0])
    ax_f = fig_c.add_subplot(gs2[1])

    gs3 = outer[3].subgridspec(1, 2, wspace=0.45)
    ax_g = fig_c.add_subplot(gs3[0])
    ax_h = fig_c.add_subplot(gs3[1])

    gs4 = outer[4].subgridspec(1, 2, wspace=0.45)
    ax_i = fig_c.add_subplot(gs4[0])
    ax_j = fig_c.add_subplot(gs4[1])

    # Panels A–D: Sade-Feldman data
    _panel_a(ax_a, sf_data)
    _panel_b(ax_b, sf_data)
    _panel_c(ax_cc, sf_data)
    _panel_d(ax_d, sf_data)

    # Panels E–F: TNBC data
    _panel_e(ax_e, tnbc_data)
    _panel_f(ax_f, tnbc_data)

    # Panels G–J: Stephenson data
    _panel_g(ax_g, steph_data)
    _panel_h(ax_h, steph_data)
    _panel_i(ax_i, steph_data)
    _panel_j(ax_j, steph_data)

    # Move legends inside plots for the composite
    _inside = {
        ax_a: "upper right", ax_b: "lower right",
        ax_cc: "upper right", ax_d: "upper right",
        ax_g: "upper right", ax_h: "upper right",
        ax_j: "lower right",
    }
    for ax_target, loc in _inside.items():
        leg = ax_target.get_legend()
        if leg:
            handles = leg.legend_handles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax_target.legend(
                handles=handles, labels=labels,
                fontsize=3.5, loc=loc,
                frameon=True, framealpha=0.85,
                edgecolor="#CCCCCC", borderpad=0.3,
                handlelength=1, handletextpad=0.3,
                labelspacing=0.2,
            )

    # Panels E and F: 2-column legends placed at the top
    leg_e = ax_e.get_legend()
    if leg_e:
        handles = leg_e.legend_handles
        labels = [t.get_text() for t in leg_e.get_texts()]
        leg_e.remove()
        ax_e.legend(
            handles=handles, labels=labels,
            fontsize=3.5, loc="upper left", ncol=2,
            frameon=True, framealpha=0.85,
            edgecolor="#CCCCCC", borderpad=0.3,
            handlelength=1, handletextpad=0.3,
            labelspacing=0.2,
        )

    leg_f = ax_f.get_legend()
    if leg_f:
        handles = leg_f.legend_handles
        labels = [t.get_text() for t in leg_f.get_texts()]
        leg_f.remove()
        ax_f.legend(
            handles=handles, labels=labels,
            fontsize=3.5, loc="lower right", ncol=2,
            frameon=True, framealpha=0.85,
            edgecolor="#CCCCCC", borderpad=0.3,
            handlelength=1, handletextpad=0.3,
            labelspacing=0.2,
        )

    # Panel H: reduce legend marker size
    leg_h = ax_h.get_legend()
    if leg_h:
        for handle in leg_h.legend_handles:
            handle.set_markersize(3)

    # Panel G: extend x-axis right to make room for signature text
    xl = ax_g.get_xlim()
    ax_g.set_xlim(xl[0], xl[1] + 0.8)
    for txt in ax_g.texts:
        txt.set_fontsize(4)

    # Panel I: increase heatmap annotation font and xtick/xlabel
    for txt in ax_i.texts:
        txt.set_fontsize(5.5)
    ax_i.tick_params(axis="x", labelsize=5.5)
    ax_i.set_xlabel("Days from Onset", fontsize=6)

    # Match xlabel, ylabel, legend font size across all panels
    _label_fs = 6
    for ax in [ax_a, ax_b, ax_cc, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j]:
        ax.xaxis.label.set_fontsize(_label_fs)
        ax.yaxis.label.set_fontsize(_label_fs)
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                txt.set_fontsize(4.5)

    _cap_fontsize(fig_c, _MAX_FONT_COMPOSITE)

    # Bold panel labels (after cap so they stay prominent)
    _lbl_fs = 9
    for ax, lbl in [
        (ax_a, "A"), (ax_b, "B"), (ax_cc, "C"), (ax_d, "D"),
        (ax_e, "E"), (ax_f, "F"),
        (ax_g, "G"), (ax_h, "H"), (ax_i, "I"), (ax_j, "J"),
    ]:
        ax.text(-0.25, 1.12, lbl, transform=ax.transAxes,
                fontsize=_lbl_fs, fontweight="bold", va="top", ha="left")

    plt.rcParams.update(_prev_rc)

    save_panel(fig_c, FIGURE_NAME, FIGURE_NAME, MAIN_OUTPUT, close=False)
    pdf_path = MAIN_OUTPUT / f"{FIGURE_NAME}_panels" / f"{FIGURE_NAME}.pdf"
    fig_c.savefig(str(pdf_path), format="pdf", bbox_inches="tight",
                  facecolor="white")
    plt.close(fig_c)
    print(f"    Saved combined artboard (PNG + PDF)")

    # Cleanup
    for d in [sf_data, tnbc_data, steph_data]:
        adata = d.get("adata")
        if adata is not None:
            del adata
    del sf_data, tnbc_data, steph_data
    gc.collect()

    print(f"  Figure 6 complete: 10 individual panels + combined (A–J)\n")


if __name__ == "__main__":
    apply_style()
    generate()
