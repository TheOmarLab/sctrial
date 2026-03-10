"""
Figure 6 — Validation, Heterogeneity & Temporal Dynamics
=========================================================

Eight-panel figure combining permutation validation on Sade-Feldman,
participant-level heterogeneity analysis, and temporal severity
dynamics from the Stephenson COVID-19 cohort.

Panels
------
A  Permutation null distributions (top 3 most significant signatures).
B  Observed effects vs 95 % null range for all signatures.
C  Individual-effect strip plot (Sade-Feldman).
D  Response-stratified heterogeneity boxplots (Sade-Feldman).
E  Signature trajectories by severity (Stephenson, DFO bins).
F  Severity divergence over DFO bins (4 representative signatures).
G  Temporal divergence heatmap (all signatures × DFO bins).
H  Time-specific Hedges' g effect sizes (all signatures × DFO bins).
"""

from __future__ import annotations

import gc
import warnings

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
    harmonize_response,
    save_panel,
    score_signatures,
    sig_display,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure6_validation_dynamics"
VISITS: tuple[str, str] = ("Pre", "Post")
N_PERM = 999

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


# ── data preparation ─────────────────────────────────────────────────────

def _to_array(mat) -> np.ndarray:
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def _prepare_sf_data() -> dict:
    """Load Sade-Feldman, run permutation test and compute participant deltas."""
    adata = get_sade_feldman()
    if "log1p_tpm" not in adata.layers:
        if "tpm" in adata.layers:
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        else:
            raise RuntimeError("No tpm layer for log1p_tpm creation.")
    adata = harmonize_response(adata)
    adata, sig_cols = score_signatures(adata, layer="log1p_tpm")

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
    print("  Running permutation test ...")
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
    hetero_feats = [f for f in HETERO_FEATURES if f in adata.var_names]
    delta_df = _compute_participant_delta(adata, hetero_feats)

    return {
        "df_part": df_part,
        "df_perm_all": df_perm_all,
        "sig_cols": sig_cols,
        "adata": adata,
        "delta_df": delta_df,
        "hetero_feats": hetero_feats,
    }


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
            obs_beta, y_label, f"  p = {perm_p:.3f}",
            fontsize=7, color=color, ha="left", fontweight="bold",
        )

    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (null distribution)")
    ax.set_ylabel("Density")
    ax.set_title("Permutation Null Distributions (Top 3)", fontsize=10)
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
        rec_df["obs_beta"], y_pos, c=colors, s=55,
        edgecolor="white", linewidth=0.5, zorder=3,
    )

    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rec_df["display"].values, fontsize=8)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title("Observed Effects vs Null Range", fontsize=10)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["highlight"], markersize=7,
               label="Outside 95% null"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["treated"], markersize=7,
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
            x, sub["effect"].values, s=18, alpha=0.55,
            c=color, edgecolors="none", label=arm,
        )

    for feat, i in x_map.items():
        mu = long[long["feature"] == feat]["effect"].mean()
        ax.hlines(mu, i - 0.35, i + 0.35, color="black", lw=2)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Participant Effect (Post − Pre)")
    ax.set_title("Individual Treatment Effects", fontsize=10)
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
    ax.set_ylabel("Participant Effect (Post − Pre)")
    ax.set_title("Response-Stratified Heterogeneity", fontsize=10)
    ax.legend(fontsize=7, frameon=True, title="Arm")
    despine(ax)


# ── Panel E: Temporal trajectories by severity ───────────────────────────

def _panel_e(ax, steph_data: dict) -> None:
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
            if not np.isnan(means[-1]):
                ax.annotate(
                    sig_display(col) if sev == "Severe" else "",
                    (len(DFO_BINS) - 1, means[-1]),
                    fontsize=6, ha="left", va="center",
                    xytext=(6, 0), textcoords="offset points",
                    color=color,
                )

    ax.set_xticks(range(len(DFO_BINS)))
    ax.set_xticklabels(DFO_LABELS)
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("Mean Signature Score")
    ax.set_title("Signature Trajectories by Severity", fontsize=10)

    handles = [
        Line2D([0], [0], color=COL_SEVERE, lw=2, ls="-", label="Severe"),
        Line2D([0], [0], color=COL_MILD, lw=2, ls="--", label="Mild"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel F: Severity divergence ─────────────────────────────────────────

def _panel_f(ax, steph_data: dict) -> None:
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
            markersize=7, markeredgecolor="white", markeredgewidth=0.5,
            label=sig_display(col),
        )

    ax.axhline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)
    ax.set_xticks(range(len(DFO_BINS)))
    ax.set_xticklabels(DFO_LABELS)
    ax.set_xlabel("Days from Onset")
    ax.set_ylabel("Divergence (Severe − Mild)")
    ax.set_title("Severity Divergence Over Time", fontsize=10)
    ax.legend(fontsize=7, loc="best", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel G: Temporal divergence heatmap ─────────────────────────────────

def _panel_g(ax, steph_data: dict) -> None:
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
    ax.set_title("Temporal Divergence Heatmap", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)


# ── Panel H: Time-specific Hedges' g ────────────────────────────────────

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


def _panel_h(ax, steph_data: dict) -> None:
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
    ax.set_title("Time-Specific Effect Sizes", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Composite generation ────────────────────────────────────────────────

def generate() -> None:
    """Create and save all Figure 6 panels."""
    apply_style()
    print("Figure 6: Validation, Heterogeneity & Temporal Dynamics")

    # Sade-Feldman data (panels A-D)
    sf_data = _prepare_sf_data()

    # Stephenson data (panels E-G)
    steph_data = _prepare_stephenson_data()

    # Panels A-D use Sade-Feldman
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

    # Panels E-H use Stephenson
    steph_panels = [
        ("panel_E_temporal_trajectories", _panel_e, (7, 5.5)),
        ("panel_F_severity_divergence", _panel_f, (7, 5.5)),
        ("panel_G_temporal_heatmap", _panel_g, (7, 5.5)),
        ("panel_H_time_specific_effect_sizes", _panel_h, (7, 5.5)),
    ]
    for panel_name, func, size in steph_panels:
        fig, ax = plt.subplots(figsize=size)
        func(ax, steph_data)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # Cleanup
    for d in [sf_data, steph_data]:
        adata = d.get("adata")
        if adata is not None:
            del adata
    del sf_data, steph_data
    gc.collect()

    print(f"  Figure 6 complete: {FIGURE_NAME}")


if __name__ == "__main__":
    apply_style()
    generate()
