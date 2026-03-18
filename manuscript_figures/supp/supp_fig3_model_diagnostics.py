"""
Supplementary Figure 3 — Model Diagnostics and Assumption Checks.
=================================================================

Validate that the OLS interaction (DiD) model assumptions hold across
all datasets with paired two-arm pre/post data.

Panels:
  A  Q-Q plots of model residuals (faceted, one per dataset).
  B  Residual vs fitted values (multi-dataset overlay).
  C  Influence diagnostics: Cook's distance per dataset.
  D  Baseline diagnostics: arm means (two-arm) / pre-post means (single-arm).
  E  Signal enrichment: observed |effect| vs permutation null quantiles.
  F  Full assumption diagnostics: normality + heteroscedasticity (merged 1×2).
  G  Funnel plot: model effect vs standard error.
  H  Observed rejection rate vs nominal alpha (signal excess over null).
  I  Pseudoreplication diagnostics: cell-level vs participant-level
     inference comparison across all 5 datasets (β scatter, −log10(p),
     SE bars).
  J  Runtime scaling across datasets (Cleveland dot plot; moved from
     Figure 3 panel C).

Non-overlap guardrail: no sensitivity analysis (→ SF4), no cross-dataset
biological concordance (→ SF5), no heterogeneity (→ SF6).
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy import stats

from sctrial import TrialDesign, did_table

from .._shared import (
    SUPP_OUTPUT,
    add_log1p_cpm_layer,
    apply_style,
    clear_cache,
    despine,
    get_aml,
    get_cart,
    get_sade_feldman,
    get_stephenson,
    get_vaccine,
    harmonize_response,
    save_panel,
)

FIGURE_NAME = "SuppFig3_model_diagnostics"

_TEST_FEATURES = [
    "CD8A", "CD4", "PDCD1", "HAVCR2", "LAG3", "CTLA4",
    "GZMB", "PRF1", "IFNG", "TNF", "IL2", "CD19",
    "CD14", "LYZ", "NKG7", "CD3D", "FOXP3", "IL7R",
]

_DATASET_CFG = {
    "Sade-Feldman": {
        # Two-arm DiD: Responder vs Non-responder, Pre vs Post (paired).
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
        # Single-arm paired: Treatment arm only, Pre vs Post (11 paired).
        # Control arm has Pre only → no two-arm contrast possible.
        "design": "single_arm_paired",
        "loader": lambda: get_aml(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "Treatment",  # keep only this arm
        "visits": ("Pre", "Post"),
    },
    "CAR-T": {
        # Single-arm paired: all patients receive CAR-T, Pre vs Post (31 paired).
        "design": "single_arm_paired",
        "loader": lambda: get_cart(),
        "harmonize": False,
        "layer": "log1p_norm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_filter": "CAR-T",
        "visits": ("Pre", "Post"),
    },
    "Stephenson": {
        # Two-arm DiD: Mild vs Severe COVID patients, D0 vs D28 (5 paired).
        "design": "two_arm",
        "loader": get_stephenson,
        "harmonize": False,
        "layer": "log1p_cpm",  # created on-the-fly from counts
        "participant_col": "participant_id",
        "visit_col": "Collection_Day",
        "arm_col": "severity",
        "arm_treated": "Severe",
        "arm_control": "Mild",
        "visits": ("D0", "D28"),
    },
    "Vaccine": {
        # Single-arm paired: all subjects vaccinated, Pre vs Post (6 paired).
        "design": "single_arm_paired",
        "loader": get_vaccine,
        "harmonize": False,
        "layer": "log1p_cpm",  # created on-the-fly from counts
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": None,  # no arm column
        "visits": ("Pre", "Post"),
    },
}

_DS_PALETTE = dict(
    zip(_DATASET_CFG.keys(), ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"])
)


# ── Data pipeline ─────────────────────────────────────────────────


def _matrix_from_layer(adata, features: list[str], layer: str) -> np.ndarray:
    X = (
        adata[:, features].layers[layer]
        if layer in adata.layers
        else adata[:, features].X
    )
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _participant_visit_means(
    adata, cfg: dict, features: list[str]
) -> pd.DataFrame:
    """Compute per-participant-visit means, respecting the study design."""
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg.get("arm_col")  # None for single-arm without arm column
    design = cfg.get("design", "two_arm")

    mat = _matrix_from_layer(adata, features, cfg["layer"])
    expr = pd.DataFrame(mat, columns=features, index=adata.obs_names)
    expr[pid_col] = adata.obs[pid_col].values
    expr[visit_col] = adata.obs[visit_col].values
    if arm_col and arm_col in adata.obs.columns:
        expr[arm_col] = adata.obs[arm_col].values

    # Filter to specific arm if single-arm design.
    arm_filter = cfg.get("arm_filter")
    if arm_filter and arm_col and arm_col in expr.columns:
        expr = expr[expr[arm_col] == arm_filter].copy()

    group_cols = [pid_col, visit_col]
    if arm_col and arm_col in expr.columns:
        group_cols.append(arm_col)

    pv = (
        expr.groupby(group_cols, observed=True)[features]
        .mean()
        .reset_index()
    )

    if design == "unpaired":
        # Cross-sectional: no pairing requirement.
        return pv

    # Paired designs: keep only participants with both visits.
    counts = pv.groupby(pid_col)[visit_col].nunique()
    paired = counts[counts >= 2].index
    pv = pv[pv[pid_col].isin(paired)].copy()
    return pv


def _ols_interaction(y: np.ndarray, post: np.ndarray, treated: np.ndarray | None):
    """Fit OLS and return diagnostics.

    Two-arm DiD:   Y ~ 1 + post + treated + post:treated  (beta[3] = DiD)
    Single-arm:    Y ~ 1 + post                           (beta[1] = effect)
    """
    if treated is not None:
        X = np.column_stack(
            [np.ones_like(post), post, treated, post * treated]
        )
        effect_idx = 3
    else:
        X = np.column_stack([np.ones_like(post), post])
        effect_idx = 1

    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        return None

    rss = float(np.sum(resid**2))
    sigma2 = rss / dof if dof > 0 else np.nan

    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))

    beta_eff = float(beta[effect_idx])
    se_eff = float(se[effect_idx])
    if not np.isfinite(se_eff) or se_eff <= 0:
        t_eff = np.nan
        p_eff = np.nan
    else:
        t_eff = beta_eff / se_eff
        p_eff = 2 * stats.t.sf(np.abs(t_eff), dof)

    # Influence diagnostics.
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (resid**2 / (p * sigma2)) * (hat / (1.0 - hat) ** 2)
    cooks[~np.isfinite(cooks)] = np.nan

    return {
        "beta": beta_eff,
        "se": se_eff,
        "t": t_eff,
        "p": p_eff,
        "resid": resid,
        "fitted": fitted,
        "hat": hat,
        "cooks": cooks,
        "sigma": float(np.sqrt(sigma2)) if sigma2 >= 0 else np.nan,
        "n_obs": int(n),
    }


def _fit_dataset(name: str, adata, cfg: dict) -> dict | None:
    """Run OLS for each feature and return effect + residual tables.

    Supports two designs via cfg["design"]:
      "two_arm"           — Y ~ 1 + post + treated + post:treated  (DiD)
      "single_arm_paired" — Y ~ 1 + post  (within-arm pre/post)
    """
    design = cfg.get("design", "two_arm")
    features = [f for f in _TEST_FEATURES if f in adata.var_names]
    if len(features) < 4:
        return None

    pv = _participant_visit_means(adata, cfg, features)
    if pv.empty:
        return None

    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg.get("arm_col")  # None for single-arm without arm column
    pre_v, post_v = cfg["visits"]

    # Restrict to requested visits.
    pv = pv[pv[visit_col].isin([pre_v, post_v])].copy()
    if pv.empty:
        return None

    if design == "two_arm":
        # Two-arm DiD: filter to the two arms.
        pv = pv[
            pv[arm_col].isin([cfg["arm_treated"], cfg["arm_control"]])
        ].copy()
        if pv.empty:
            return None
        arm_counts = pv.groupby(arm_col)[pid_col].nunique()
        if (
            cfg["arm_treated"] not in arm_counts
            or cfg["arm_control"] not in arm_counts
        ):
            return None
        n_per_group = int(
            min(
                arm_counts[cfg["arm_treated"]],
                arm_counts[cfg["arm_control"]],
            )
        )
        if n_per_group < 2:
            return None
        treated = (pv[arm_col].values == cfg["arm_treated"]).astype(float)
    else:
        # Single-arm: no arm covariate.
        n_per_group = int(pv[pid_col].nunique())
        if n_per_group < 2:
            return None
        treated = None

    post = (pv[visit_col].values == post_v).astype(float)

    effect_rows: list[dict] = []
    resid_rows: list[pd.DataFrame] = []
    meta_cols = [pid_col, visit_col]
    if arm_col and arm_col in pv.columns:
        meta_cols.append(arm_col)

    for feat in features:
        fit = _ols_interaction(pv[feat].values.astype(float), post, treated)
        if fit is None:
            continue
        effect_rows.append({
            "feature": feat,
            "beta": fit["beta"],
            "se": fit["se"],
            "t": fit["t"],
            "p": fit["p"],
            "sigma": fit["sigma"],
            "n_obs": fit["n_obs"],
        })

        tmp = pv[meta_cols].copy()
        tmp["feature"] = feat
        tmp["residual"] = fit["resid"]
        tmp["fitted"] = fit["fitted"]
        tmp["leverage"] = fit["hat"]
        tmp["cooks_d"] = fit["cooks"]
        resid_rows.append(tmp)

    if not effect_rows:
        return None

    effect_df = pd.DataFrame(effect_rows)
    resid_df = pd.concat(resid_rows, axis=0, ignore_index=True)

    # Permutation null.
    null_abs = _permutation_null(pv, features, cfg, n_perm=200)

    design_label = "DiD" if design == "two_arm" else "pre–post"

    return {
        "name": name,
        "effects": effect_df,
        "residuals": resid_df,
        "features": features,
        "n_per_group": n_per_group,
        "null_abs": null_abs,
        "pv": pv,
        "cfg": cfg,
        "design_label": design_label,
    }


def _permutation_null(
    pv: pd.DataFrame,
    features: list[str],
    cfg: dict,
    n_perm: int = 200,
) -> np.ndarray:
    """Generate null distribution of |beta| by permuting labels.

    Two-arm: permute arm assignment across participants.
    Single-arm: permute visit assignment across participants.
    """
    design = cfg.get("design", "two_arm")
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    _pre_v, post_v = cfg["visits"]

    post = (pv[visit_col].values == post_v).astype(float)
    rng = np.random.default_rng(42)
    out: list[float] = []

    if design == "two_arm":
        arm_col = cfg["arm_col"]
        pid_df = (
            pv[[pid_col, arm_col]].drop_duplicates(subset=[pid_col]).copy()
        )
        orig = (pid_df[arm_col].values == cfg["arm_treated"]).astype(float)
        pids = pid_df[pid_col].values

        for _ in range(n_perm):
            perm = rng.permutation(orig)
            map_df = pd.DataFrame({pid_col: pids, "treated_perm": perm})
            treated = (
                pv[[pid_col]]
                .merge(map_df, on=pid_col, how="left")["treated_perm"]
                .values.astype(float)
            )
            for feat in features:
                fit = _ols_interaction(
                    pv[feat].values.astype(float), post, treated
                )
                if fit is not None and np.isfinite(fit["beta"]):
                    out.append(abs(fit["beta"]))
    else:
        # Single-arm: permute visit labels WITHIN each participant.
        pids = pv[pid_col].values
        unique_pids = np.unique(pids)
        for _ in range(n_perm):
            post_perm = post.copy()
            for pid in unique_pids:
                mask = pids == pid
                post_perm[mask] = rng.permutation(post_perm[mask])
            for feat in features:
                fit = _ols_interaction(
                    pv[feat].values.astype(float), post_perm, None
                )
                if fit is not None and np.isfinite(fit["beta"]):
                    out.append(abs(fit["beta"]))

    return np.asarray(out, dtype=float)


def _load_results() -> dict[str, dict]:
    """Load all datasets and fit OLS DiD for each."""
    out: dict[str, dict] = {}
    for name, cfg in _DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)
            # Create log1p_cpm layer if needed but missing.
            layer = cfg["layer"]
            if layer == "log1p_cpm" and "log1p_cpm" not in adata.layers:
                if "counts" in adata.layers:
                    adata = add_log1p_cpm_layer(
                        adata, counts_layer="counts", out_layer="log1p_cpm",
                    )
                else:
                    print(f"  {name}: skipped (no counts layer for log1p_cpm)")
                    continue
            res = _fit_dataset(name, adata, cfg)
            if res is not None:
                out[name] = res
                print(
                    f"  {name}: {len(res['effects'])} features, "
                    f"n/group={res['n_per_group']}"
                )
            else:
                print(
                    f"  {name}: skipped (insufficient paired data)"
                )
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return out


# ── Panel functions ───────────────────────────────────────────────


def _panel_qq(fig, axes, results: dict[str, dict]):
    """A: Q-Q of residuals (one subplot per dataset)."""
    names = list(results)
    for i, ax in enumerate(np.ravel(axes)):
        if i >= len(names):
            ax.axis("off")
            continue
        name = names[i]
        vals = results[name]["residuals"]["residual"].dropna().values
        if len(vals) < 4:
            ax.text(
                0.5, 0.5, "Too few residuals",
                ha="center", va="center", transform=ax.transAxes,
            )
            continue
        stats.probplot(vals, dist="norm", plot=ax)
        ax.set_title(name, fontsize=10, fontweight="bold")
        lines = ax.get_lines()
        if len(lines) >= 2:
            lines[0].set_markersize(4)
            lines[0].set_alpha(0.7)
            lines[0].set_markerfacecolor(_DS_PALETTE.get(name, "grey"))
            lines[1].set_color("#333333")
            lines[1].set_linewidth(1.5)
        despine(ax)


def _panel_resid_fitted(fig_faceted, results: dict[str, dict]):
    """B: Residual vs fitted value scatter, faceted per dataset."""
    names = list(results.keys())
    n_ds = len(names)
    ncols = min(n_ds, 3)
    nrows = max(1, (n_ds + ncols - 1) // ncols)
    axes = fig_faceted.subplots(nrows, ncols, squeeze=False)
    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        df = results[name]["residuals"]
        ax.scatter(
            df["fitted"], df["residual"],
            s=10, alpha=0.35,
            color=_DS_PALETTE.get(name, "grey"),
            rasterized=True,
        )
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Fitted value", fontsize=8)
        ax.set_ylabel("Residual", fontsize=8)
        ax.set_title(name, fontsize=10, fontweight="bold")
        despine(ax)
    # Hide unused axes
    for idx in range(n_ds, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)


def _panel_influence(ax, results: dict[str, dict]):
    """C: Cook's distance boxplot per dataset."""
    rows = []
    for name, res in results.items():
        d = res["residuals"]["cooks_d"].dropna()
        for v in d.values:
            rows.append({"Dataset": name, "Cook's D": float(v)})
    if not rows:
        ax.text(
            0.5, 0.5, "No influence data",
            ha="center", va="center", transform=ax.transAxes,
        )
        return
    df = pd.DataFrame(rows)
    sns.boxplot(
        data=df, x="Dataset", y="Cook's D",
        ax=ax, palette=_DS_PALETTE, fliersize=1.5,
    )
    sns.stripplot(
        data=df, x="Dataset", y="Cook's D",
        ax=ax, color="black", size=2.0, alpha=0.25, jitter=0.2,
    )
    # 4/n threshold per dataset + participant-level flagged rate annotation
    for i, name in enumerate(df["Dataset"].unique()):
        n = results[name]["effects"]["n_obs"].median()
        n_features = len(results[name]["features"])
        if np.isfinite(n) and n > 0:
            thr = 4.0 / n
            ax.plot(
                [i - 0.35, i + 0.35], [thr, thr],
                color="red", lw=1.0, ls="--",
            )
            # Count participants flagged in ≥1 feature (not pooled obs)
            resid_df = results[name]["residuals"]
            pid_col = results[name]["cfg"]["participant_col"]
            if pid_col in resid_df.columns:
                flagged_pids = resid_df.loc[
                    resid_df["cooks_d"] > thr, pid_col
                ].nunique()
                total_pids = resid_df[pid_col].nunique()
                rate = flagged_pids / total_pids if total_pids > 0 else 0
                ax.text(
                    i, ax.get_ylim()[1] * 0.95,
                    f"{flagged_pids}/{total_pids} participants\nflagged"
                    f" ({rate:.0%})",
                    ha="center", va="top", fontsize=5.5, color="red",
                    fontstyle="italic",
                )
            else:
                # Fallback: pooled observation count
                ds_vals = df.loc[df["Dataset"] == name, "Cook's D"].values
                n_flagged = int(np.sum(ds_vals > thr))
                n_total = len(ds_vals)
                rate = n_flagged / n_total if n_total > 0 else 0
                ax.text(
                    i, ax.get_ylim()[1] * 0.95,
                    f"{n_flagged}/{n_total} obs. flagged\n({rate:.0%},"
                    f" {n_features} features)",
                    ha="center", va="top", fontsize=5.5, color="red",
                    fontstyle="italic",
                )
    ax.set_title("Influence Diagnostics (Cook's D)", fontweight="bold")
    despine(ax)


def _panel_baseline_comparability(fig, axes, results: dict[str, dict]):
    """D: Baseline mean comparability.

    Two-arm datasets: scatter of (control pre-mean, treated pre-mean) per
    feature.  Points near diagonal → arms have similar baseline expression.

    Single-arm datasets: scatter of (Pre mean, Post mean) per feature to
    show the magnitude/direction of change.
    """
    names = list(results)
    for i, ax in enumerate(np.ravel(axes)):
        if i >= len(names):
            ax.axis("off")
            continue
        name = names[i]
        res = results[name]
        pv = res["pv"]
        cfg = res["cfg"]
        features = res["features"]
        design = cfg.get("design", "two_arm")

        visit_col = cfg["visit_col"]
        pre_v, post_v = cfg["visits"]

        if design == "two_arm":
            arm_col = cfg["arm_col"]
            pre = pv[pv[visit_col] == pre_v]
            if pre.empty:
                ax.text(
                    0.5, 0.5, "No pre-treatment data",
                    ha="center", va="center", transform=ax.transAxes,
                )
                continue
            x_vals = (
                pre[pre[arm_col] == cfg["arm_control"]][features].mean()
            )
            y_vals = (
                pre[pre[arm_col] == cfg["arm_treated"]][features].mean()
            )
            x_label = f"{cfg['arm_control']} mean"
            y_label = f"{cfg['arm_treated']} mean"
            subtitle = f"{name} — baseline comparability"
        else:
            # Single-arm: compare Pre vs Post means.
            pre_data = pv[pv[visit_col] == pre_v]
            post_data = pv[pv[visit_col] == post_v]
            if pre_data.empty or post_data.empty:
                ax.text(
                    0.5, 0.5, "No pre/post data",
                    ha="center", va="center", transform=ax.transAxes,
                )
                continue
            x_vals = pre_data[features].mean()
            y_vals = post_data[features].mean()
            x_label = "Pre-treatment mean"
            y_label = "Post-treatment mean"
            subtitle = f"{name} — pre vs post"

        ax.scatter(
            x_vals.values, y_vals.values,
            s=40, alpha=0.8,
            color=_DS_PALETTE.get(name, "grey"),
            edgecolors="white", linewidth=0.5, zorder=3,
        )
        texts = [
            ax.text(cx, ty, feat, fontsize=7, fontweight="bold")
            for feat, cx, ty in zip(
                features, x_vals.values, y_vals.values
            )
        ]
        adjust_text(
            texts, ax=ax,
            force_text=(2.0, 2.0), force_points=(2.0, 2.0),
            expand=(1.5, 1.5),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

        lo = min(x_vals.min(), y_vals.min()) * 0.9
        hi = max(x_vals.max(), y_vals.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], ls="--", color="black", lw=0.9)

        r, _ = stats.pearsonr(x_vals.values, y_vals.values)
        ax.text(
            0.05, 0.92, f"r = {r:.3f}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_aspect("equal", adjustable="datalim")
        despine(ax)


def _panel_signal_enrichment(ax, results: dict[str, dict]):
    """E: Observed |effect| quantiles vs permutation null quantiles.

    Points above the diagonal indicate features with effects larger than
    expected under the null (signal enrichment), NOT type-I error calibration.
    """
    for name, res in results.items():
        label = f"{name} ({res['design_label']})"
        obs = np.sort(np.abs(res["effects"]["beta"].dropna().values))
        null = np.sort(res.get("null_abs", np.array([])))
        if len(obs) < 3 or len(null) < 3:
            continue
        q = np.linspace(0.05, 0.95, min(len(obs), 200))
        obs_q = np.quantile(obs, q)
        null_q = np.quantile(null, q)
        ax.plot(
            null_q, obs_q, marker="o", ms=3.0, lw=1.2,
            color=_DS_PALETTE.get(name, "grey"), label=label,
        )
    lim = ax.get_xlim()
    hi = max(ax.get_ylim()[1], lim[1])
    ax.plot([0, hi], [0, hi], ls="--", color="black", lw=0.9,
            label="No enrichment")
    ax.set_xlabel("Permutation null |effect| quantiles")
    ax.set_ylabel("Observed |effect| quantiles")
    ax.set_title("Signal Enrichment vs Permutation Null", fontweight="bold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)



def _panel_normality_tests(ax, results: dict[str, dict]):
    """F-left: Per-feature Shapiro-Wilk, summarised as median W per dataset.

    Tests are run on each feature's residuals separately (not pooled across
    features), since pooling residuals from different models violates the
    independence assumption of the Shapiro-Wilk test.
    """
    rows = []
    for name, res in results.items():
        rdf = res["residuals"]
        w_vals, sk_vals = [], []
        for feat in res["features"]:
            fvals = rdf.loc[rdf["feature"] == feat, "residual"].dropna().values
            if len(fvals) < 8:
                continue
            sw_stat, _ = stats.shapiro(fvals)
            w_vals.append(sw_stat)
            sk_vals.append(stats.skew(fvals))
        if not w_vals:
            continue
        rows.append({
            "Dataset": name,
            "Median W": float(np.median(w_vals)),
            "Frac W>0.95": float(np.mean(np.array(w_vals) > 0.95)),
            "Median skew": float(np.median(sk_vals)),
            "n_features": len(w_vals),
        })

    if not rows:
        ax.text(
            0.5, 0.5, "No test results",
            ha="center", va="center", transform=ax.transAxes,
        )
        return

    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    colors = [_DS_PALETTE.get(n, "grey") for n in df["Dataset"]]

    # Main: median Shapiro W statistic bars
    ax.barh(y, df["Median W"], color=colors, edgecolor="white",
            height=0.6, alpha=0.85)
    for i, (_, row) in enumerate(df.iterrows()):
        label = (f"med W={row['Median W']:.3f}, "
                 f"{row['Frac W>0.95']:.0%} pass "
                 f"({int(row['n_features'])} features)")
        ax.text(row["Median W"] + 0.002, i, label, va="center",
                fontsize=7, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["Dataset"], fontsize=9)
    ax.set_xlabel("Median Shapiro-Wilk W (per-feature)")
    ax.set_title("Residual Normality Diagnostics", fontweight="bold")
    ax.axvline(0.95, color="red", ls="--", lw=0.8, alpha=0.7,
               label="W=0.95 reference")
    ax.legend(fontsize=6, frameon=True)
    despine(ax)


def _panel_funnel(ax, results: dict[str, dict]):
    """G: Funnel plot — model effect vs standard error."""
    for name, res in results.items():
        label = f"{name} ({res['design_label']})"
        df = res["effects"]
        ax.scatter(
            df["beta"], df["se"],
            s=45, alpha=0.8,
            color=_DS_PALETTE.get(name, "grey"),
            edgecolors="white", linewidth=0.5, label=label,
        )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.invert_yaxis()  # convention: SE=0 at top
    ax.set_xlabel("Model effect (β)")
    ax.set_ylabel("Standard error")
    ax.set_title("Effect Size vs Standard Error (Funnel)", fontweight="bold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


def _panel_assumptions_merged(fig_merged, results: dict[str, dict]):
    """F: Combined normality + heteroscedasticity in 1×2 subplot."""
    ax1, ax2 = fig_merged.subplots(1, 2)
    _panel_normality_tests(ax1, results)
    _panel_heteroscedasticity(ax2, results)


def _panel_rejection_vs_alpha(ax, results: dict[str, dict]):
    """H: Observed rejection rate vs nominal alpha.

    For alphas in [0.01..0.20], compute the fraction of features where
    |observed effect| > quantile(|null effects|, 1-alpha).  Above the
    diagonal indicates more signal than expected under the null (not
    type-I error calibration, since observed effects may contain true signal).
    """
    alphas = np.arange(0.01, 0.205, 0.01)
    for name, res in results.items():
        label = f"{name} ({res['design_label']})"
        obs_abs = np.abs(res["effects"]["beta"].dropna().values)
        null = res.get("null_abs", np.array([]))
        if len(obs_abs) < 3 or len(null) < 10:
            continue
        rates = []
        for alpha in alphas:
            thr = np.quantile(null, 1 - alpha)
            rate = np.mean(obs_abs > thr)
            rates.append(float(rate))
        ax.plot(alphas, rates, marker="o", ms=3, lw=1.3,
                color=_DS_PALETTE.get(name, "grey"), label=label)
    ax.plot([0, 0.25], [0, 0.25], ls="--", color="black", lw=0.9,
            label="No signal (null)")
    ax.set_xlabel("Nominal α (permutation threshold)")
    ax.set_ylabel("Fraction of features exceeding threshold")
    ax.set_title("Observed Rejection Rate vs Nominal α", fontweight="bold")
    ax.legend(fontsize=6, frameon=True)
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 1.05)
    despine(ax)


# ── Heteroscedasticity helper (used by merged panel F) ─────────────


def _panel_heteroscedasticity(ax, results: dict[str, dict]):
    """F-right: Per-feature Breusch-Pagan, summarised per dataset.

    Tests are run on each feature's residuals separately (not pooled),
    since pooling residuals from different models violates i.i.d. assumptions.
    Reports median BP statistic and fraction with p < 0.05.
    """
    rows = []
    for name, res in results.items():
        rdf = res["residuals"]
        bp_stats, bp_ps = [], []
        for feat in res["features"]:
            fmask = rdf["feature"] == feat
            vals = rdf.loc[fmask, "residual"].dropna().values
            fitted = rdf.loc[fmask, "fitted"].dropna().values
            if len(vals) < 8 or len(fitted) < 8:
                continue
            n = min(len(vals), len(fitted))
            vals, fitted = vals[:n], fitted[:n]
            sq_resid = vals ** 2
            X_bp = np.column_stack([np.ones(n), fitted])
            try:
                beta_bp = np.linalg.lstsq(X_bp, sq_resid, rcond=None)[0]
                sq_resid_hat = X_bp @ beta_bp
                ss_reg = np.sum((sq_resid_hat - sq_resid.mean()) ** 2)
                ss_tot = np.sum((sq_resid - sq_resid.mean()) ** 2)
                if ss_tot == 0:
                    continue
                r2_bp = ss_reg / ss_tot
                bp_stat = n * r2_bp
                bp_p = 1.0 - stats.chi2.cdf(bp_stat, df=1)
                bp_stats.append(bp_stat)
                bp_ps.append(bp_p)
            except Exception:
                continue

        if not bp_stats:
            continue
        rows.append({
            "Dataset": name,
            "Median BP": float(np.median(bp_stats)),
            "Frac p<0.05": float(np.mean(np.array(bp_ps) < 0.05)),
            "n_features": len(bp_stats),
        })

    if not rows:
        ax.text(0.5, 0.5, "No test results", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    colors = [_DS_PALETTE.get(n, "grey") for n in df["Dataset"]]

    ax.barh(y, df["Median BP"], color=colors, edgecolor="white",
            height=0.6, alpha=0.85)
    for i, (_, row) in enumerate(df.iterrows()):
        label = (f"med BP={row['Median BP']:.2f}, "
                 f"{row['Frac p<0.05']:.0%} sig "
                 f"({int(row['n_features'])} features)")
        ax.text(row["Median BP"] + 0.05, i, label, va="center",
                fontsize=7, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["Dataset"], fontsize=9)
    ax.set_xlabel("Median Breusch-Pagan statistic (per-feature)")
    ax.set_title("Heteroscedasticity Diagnostics", fontweight="bold")
    ax.axvline(3.84, color="red", ls="--", lw=0.8, alpha=0.7,
               label=r"$\chi^2$(1) = 3.84 reference")
    ax.legend(fontsize=6, frameon=True)
    despine(ax)


# ── Generate ──────────────────────────────────────────────────────


# ── Pseudoreplication panels (I) ──────────────────────────────────


def _naive_cell_prepost(expr_df, features, visit_col, visits):
    """Naive cell-level pre-vs-post: two-sample t-test ignoring participant.

    This intentionally ignores participant structure — every cell is treated
    as independent, which is exactly the pseudoreplication fallacy.
    Returns DataFrame with columns (feature, beta, pval, se).
    """
    visit_pre, visit_post = visits
    pre = expr_df[expr_df[visit_col] == visit_pre]
    post = expr_df[expr_df[visit_col] == visit_post]
    rows = []
    for feat in features:
        x_pre = pre[feat].dropna().values
        x_post = post[feat].dropna().values
        if len(x_pre) < 2 or len(x_post) < 2:
            rows.append({"feature": feat, "beta": np.nan,
                         "pval": np.nan, "se": np.nan})
            continue
        beta = x_post.mean() - x_pre.mean()
        # Welch's t-test (unequal variance) for consistency:
        # SE uses separate variances, df uses Welch-Satterthwaite.
        t_res = stats.ttest_ind(x_post, x_pre, equal_var=False)
        se = abs(beta / t_res.statistic) if t_res.statistic != 0 else 0.0
        pval = float(t_res.pvalue)
        rows.append({"feature": feat, "beta": beta, "pval": pval, "se": se})
    return pd.DataFrame(rows)


def _participant_paired_delta(expr_df, features, pid_col, visit_col, visits):
    """Participant-level paired Δ: per-participant pre-post change.

    Proper inference: Δ_i = post_i − pre_i, then one-sample t-test on Δ_i.
    Returns DataFrame with columns (feature, beta, pval, se).
    """
    visit_pre, visit_post = visits
    pre = (expr_df[expr_df[visit_col] == visit_pre]
           .set_index(pid_col)[features])
    post = (expr_df[expr_df[visit_col] == visit_post]
            .set_index(pid_col)[features])
    shared = pre.index.intersection(post.index)
    if len(shared) < 2:
        return pd.DataFrame(
            [{"feature": f, "beta": np.nan, "pval": np.nan, "se": np.nan}
             for f in features])
    delta = post.loc[shared] - pre.loc[shared]  # (n_participants, n_features)
    rows = []
    for feat in features:
        d = delta[feat].dropna().values
        if len(d) < 2:
            rows.append({"feature": feat, "beta": np.nan,
                         "pval": np.nan, "se": np.nan})
            continue
        mean_d = d.mean()
        se_d = d.std(ddof=1) / np.sqrt(len(d))
        if se_d > 0:
            t_stat = mean_d / se_d
            pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(d) - 1))
        elif mean_d == 0:
            # No variance, no effect — undefined test
            pval = np.nan
        else:
            # No variance but nonzero mean — infinitely significant
            pval = 0.0
        rows.append({"feature": feat, "beta": mean_d, "pval": pval,
                     "se": se_d})
    return pd.DataFrame(rows)


def _pseudorep_data_for_dataset(
    name: str, cfg: dict,
) -> dict | None:
    """Run cell-level and participant-level inference for one dataset.

    Two-arm designs use DiD (beta_DiD).
    Single-arm designs use pre-vs-post OLS: y ~ time + participant_FE.
    Both return normalised columns (feature, beta, pval, se) so that
    cell vs participant comparisons are on equal footing.
    """
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
            else:
                return None
        if layer == "log1p_tpm" and "log1p_tpm" not in adata.layers:
            if "tpm" in adata.layers:
                adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
            else:
                return None

        features = [f for f in _TEST_FEATURES if f in adata.var_names]
        if len(features) < 4:
            return None

        design_type = cfg.get("design", "two_arm")

        if design_type == "two_arm":
            # ── Two-arm: use did_table for DiD interaction ──
            arm_col_val = cfg.get("arm_col", "response")
            design = TrialDesign(
                participant_col=cfg["participant_col"],
                visit_col=cfg["visit_col"],
                arm_col=arm_col_val,
                arm_treated=cfg["arm_treated"],
                arm_control=cfg["arm_control"],
            )
            common_kw = dict(
                features=features,
                design=design,
                visits=cfg["visits"],
                layer=layer,
                standardize=True,
            )
            df_cell = did_table(adata, aggregate="cell", **common_kw)
            df_part = did_table(
                adata, aggregate="participant_visit", **common_kw,
            )
            # Rename DiD columns to normalised names
            for df in [df_cell, df_part]:
                df.rename(columns={
                    "beta_DiD": "beta", "p_DiD": "pval", "se_DiD": "se",
                }, inplace=True)
        else:
            # ── Single-arm: naive cell pre/post vs participant paired Δ ──
            arm_filter = cfg.get("arm_filter")
            arm_col_val = cfg.get("arm_col")
            if arm_filter and arm_col_val:
                adata = adata[adata.obs[arm_col_val] == arm_filter].copy()

            pid_col = cfg["participant_col"]
            visit_col = cfg["visit_col"]
            visits = cfg["visits"]

            # Build cell-level expression DataFrame
            mat = _matrix_from_layer(adata, features, layer)
            expr_cell = pd.DataFrame(
                mat, columns=features, index=adata.obs_names,
            )
            expr_cell[pid_col] = adata.obs[pid_col].values
            expr_cell[visit_col] = adata.obs[visit_col].values

            # Cell-level: naive two-sample pre-vs-post (pseudoreplication)
            df_cell = _naive_cell_prepost(
                expr_cell, features, visit_col, visits,
            )

            # Participant-level: pseudobulk means → paired Δ with t-test
            expr_part = (
                expr_cell.groupby([pid_col, visit_col], observed=True)[features]
                .mean()
                .reset_index()
            )
            df_part = _participant_paired_delta(
                expr_part, features, pid_col, visit_col, visits,
            )

        del adata
        gc.collect()
        return {"df_cell": df_cell, "df_part": df_part, "name": name}

    except Exception as exc:
        import traceback
        print(f"  Pseudorep {name}: failed ({exc})")
        traceback.print_exc()
        return None


def _panel_pseudoreplication_single(ds_name, res, design_type="two_arm"):
    """One-dataset pseudoreplication panel: 1×3 (β scatter, −log10(p), SE).

    Returns figure or None if data is invalid.
    design_type controls axis labels: "two_arm" → DiD, else → Δ (pre-post).
    """
    df_cell = res["df_cell"]
    df_part = res["df_part"]

    # Drop features where either level has NaN
    merged = df_cell[["feature", "beta", "pval", "se"]].merge(
        df_part[["feature", "beta", "pval", "se"]],
        on="feature", suffixes=("_cell", "_part"),
    )
    merged = merged.dropna(
        subset=["beta_cell", "beta_part", "pval_cell", "pval_part",
                "se_cell", "se_part"],
    )
    if len(merged) < 3:
        return None

    # Axis labels depend on design
    if design_type == "two_arm":
        beta_label = r"$\beta_{\mathrm{DiD}}$"
        method_tag = "DiD"
    else:
        beta_label = r"$\Delta$ (pre$-$post)"
        method_tag = r"$\Delta$ (within-arm)"

    color = _DS_PALETTE.get(ds_name, "#555555")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    # Col 0: β scatter (cell vs participant) with adjustText
    ax = axes[0]
    x = merged["beta_cell"].values
    y = merged["beta_part"].values
    span = max(np.ptp(x), np.ptp(y)) if len(x) > 1 else 1.0
    margin = span * 0.15 if span > 0 else 0.5
    lo = min(x.min(), y.min()) - margin
    hi = max(x.max(), y.max()) + margin
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999999", lw=1, zorder=1)
    ax.scatter(x, y, s=50, color=color, edgecolor="white",
               linewidth=0.5, zorder=3)
    texts = []
    for _, row in merged.iterrows():
        texts.append(
            ax.text(
                row["beta_cell"], row["beta_part"], row["feature"],
                fontsize=5.5, ha="left", va="bottom",
            )
        )
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.4),
        expand=(1.3, 1.5),
        force_text=(0.8, 0.8),
    )
    if len(x) >= 3:
        r, p = stats.pearsonr(x, y)
        ax.text(
            0.05, 0.95, f"r = {r:.3f}\np = {p:.2e}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="none", alpha=0.8),
        )
    ax.set_xlabel(f"Cell-level {beta_label}", fontsize=9)
    ax.set_ylabel(f"Participant-level {beta_label}", fontsize=9)
    ax.set_title("Effect Size Correlation", fontsize=10, fontweight="bold")
    despine(ax)

    # Col 1: −log10(p) bars — use log scale so both levels are visible
    ax = axes[1]
    merged["nlog10_cell"] = -np.log10(
        merged["pval_cell"].clip(lower=1e-300))
    merged["nlog10_part"] = -np.log10(
        merged["pval_part"].clip(lower=1e-300))
    m_sorted = merged.sort_values("nlog10_part", ascending=True)
    y_pos = np.arange(len(m_sorted))
    bar_h = 0.35
    # Use log1p transform: log10(1 + nlog10p) — compresses large values,
    # keeps small values visible
    nlog_cell_vals = m_sorted["nlog10_cell"].values
    nlog_part_vals = m_sorted["nlog10_part"].values
    # Check if scale ratio is extreme (>20x) — use log x-axis
    max_cell = np.nanmax(nlog_cell_vals) if len(nlog_cell_vals) else 1
    max_part = np.nanmax(nlog_part_vals) if len(nlog_part_vals) else 1
    use_log_x = (max_cell / max(max_part, 0.01) > 20) or \
                (max_part / max(max_cell, 0.01) > 20)

    ax.barh(y_pos - bar_h / 2, nlog_cell_vals,
            height=bar_h, color=color, alpha=0.5,
            label="Cell-level", edgecolor="none")
    ax.barh(y_pos + bar_h / 2, nlog_part_vals,
            height=bar_h, color=color, alpha=0.9,
            label="Participant-level", edgecolor="none")
    # α = 0.05 threshold
    thresh = -np.log10(0.05)
    ax.axvline(thresh, ls="--", color="#999999", lw=0.8)
    if use_log_x:
        ax.set_xscale("symlog", linthresh=1.0)
        # Nice tick placement for symlog
        ax.set_xlim(left=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(m_sorted["feature"].values, fontsize=6.5)
    ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=9)
    ax.set_title("Cell vs Participant Significance", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)

    # Col 2: SE bars — use log x-axis when scale ratio is extreme
    ax = axes[2]
    m_sorted2 = merged.sort_values("se_part", ascending=True)
    y_pos2 = np.arange(len(m_sorted2))
    se_cell_vals = m_sorted2["se_cell"].values
    se_part_vals = m_sorted2["se_part"].values
    max_se_cell = np.nanmax(se_cell_vals) if len(se_cell_vals) else 1
    max_se_part = np.nanmax(se_part_vals) if len(se_part_vals) else 1
    use_log_se = (max_se_part / max(max_se_cell, 1e-10) > 20) or \
                 (max_se_cell / max(max_se_part, 1e-10) > 20)

    ax.barh(y_pos2 - bar_h / 2, se_cell_vals,
            height=bar_h, color=color, alpha=0.5,
            label="Cell-level SE", edgecolor="none")
    ax.barh(y_pos2 + bar_h / 2, se_part_vals,
            height=bar_h, color=color, alpha=0.9,
            label="Participant-level SE", edgecolor="none")
    if use_log_se:
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlim(left=0)
    ax.set_yticks(y_pos2)
    ax.set_yticklabels(m_sorted2["feature"].values, fontsize=6.5)
    ax.set_xlabel("Standard Error", fontsize=9)
    ax.set_title("Precision Comparison", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)

    fig.suptitle(
        f"Cell-Level vs Participant-Level Inference: {ds_name} ({method_tag})",
        fontweight="bold", fontsize=12, y=1.02,
    )
    fig.tight_layout()
    return fig


def generate():
    """Create and save all Supplementary Figure 3 panels."""
    print("Supplementary Figure 3: Model Diagnostics and Assumption Checks")
    results = _load_results()
    if not results:
        print("  No valid datasets found; skipping.")
        return

    names = list(results)
    n_ds = len(names)

    # A: Residual Q-Q (faceted)
    ncols_qq = min(n_ds, 3)
    nrows_qq = max(1, (n_ds + ncols_qq - 1) // ncols_qq)
    fig, axes = plt.subplots(
        nrows_qq, ncols_qq, figsize=(4.6 * ncols_qq, 4.2 * nrows_qq),
    )
    if n_ds == 1:
        axes = np.array([axes])
    _panel_qq(fig, axes, results)
    fig.suptitle("Residual Q-Q Plots", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # B: Residual vs fitted (faceted per dataset)
    ncols_b = min(n_ds, 3)
    nrows_b = max(1, (n_ds + ncols_b - 1) // ncols_b)
    fig = plt.figure(figsize=(5.2 * ncols_b, 4.8 * nrows_b))
    _panel_resid_fitted(fig, results)
    fig.suptitle("Residual vs Fitted", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_panel(fig, "panel_B", FIGURE_NAME, SUPP_OUTPUT)

    # C: Influence diagnostics
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    _panel_influence(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_C", FIGURE_NAME, SUPP_OUTPUT)

    # D: Baseline diagnostics (faceted)
    ncols_pt = min(n_ds, 3)
    nrows_pt = max(1, (n_ds + ncols_pt - 1) // ncols_pt)
    fig, axes = plt.subplots(
        nrows_pt, ncols_pt, figsize=(5.0 * ncols_pt, 4.8 * nrows_pt),
    )
    if n_ds == 1:
        axes = np.array([axes])
    _panel_baseline_comparability(fig, axes, results)
    fig.suptitle(
        "Baseline Mean Comparability",
        fontweight="bold", y=1.02, fontsize=11,
    )
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # E: Signal enrichment vs permutation null
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _panel_signal_enrichment(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # F: Merged normality + heteroscedasticity (1×2)
    fig = plt.figure(figsize=(16.0, 5.2))
    _panel_assumptions_merged(fig, results)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # G: Funnel plot
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _panel_funnel(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # H: Observed rejection rate vs nominal alpha
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _panel_rejection_vs_alpha(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # I: Pseudoreplication — cell-level vs participant-level (separate per dataset)
    print("  Computing pseudoreplication comparisons ...")
    pseudo_idx = 0
    for ds_name, ds_cfg in _DATASET_CFG.items():
        print(f"    {ds_name} ...")
        pr = _pseudorep_data_for_dataset(ds_name, ds_cfg)
        if pr is None:
            print(f"    {ds_name}: skipped (data unavailable)")
            continue
        # Skip datasets where cell-level inference is degenerate (all NaN)
        if pr["df_cell"]["beta"].isna().all():
            print(f"    {ds_name}: skipped (cell-level stats degenerate)")
            continue
        ds_design = ds_cfg.get("design", "two_arm")
        fig_pr = _panel_pseudoreplication_single(ds_name, pr, ds_design)
        if fig_pr is not None:
            pseudo_idx += 1
            panel_label = f"panel_I{pseudo_idx}_{ds_name.replace('-', '_')}"
            save_panel(fig_pr, panel_label, FIGURE_NAME, SUPP_OUTPUT)
        else:
            print(f"    {ds_name}: skipped (too few valid features)")
    if pseudo_idx == 0:
        print("  Pseudoreplication: no datasets had valid cell+participant stats.")

    # J: Runtime scaling (Cleveland dot plot, moved from Figure 3)
    try:
        from ..main.figure3_robustness_benchmarking import (
            _panel_c as _fig3_panel_c,
        )
        from ..main.figure3_robustness_benchmarking import (
            _prepare_scalability_data,
        )
        scale_data = _prepare_scalability_data()
        fig_rt, ax_rt = plt.subplots(figsize=(8, 5.5))
        _fig3_panel_c(ax_rt, {"scale_data": scale_data})
        fig_rt.tight_layout()
        save_panel(fig_rt, "panel_J_runtime_scaling", FIGURE_NAME, SUPP_OUTPUT)
    except Exception as exc:
        print(f"  Warning: Could not generate runtime panel J: {exc}")

    # Cleanup
    results.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
