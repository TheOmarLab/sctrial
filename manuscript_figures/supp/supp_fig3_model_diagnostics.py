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
  E  Calibration of observed effects against permutation null.
  F  Residual normality diagnostics (Shapiro-Wilk, skewness, kurtosis).
  G  Funnel plot: effect size vs standard error.
  H  Heteroscedasticity diagnostics (Breusch-Pagan test).

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

from .._shared import (
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
        "loader": lambda: load_clinical_trial_dataset("aml"),
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
        "loader": lambda: load_clinical_trial_dataset("cart"),
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
        "beta_did": beta_eff,
        "se_did": se_eff,
        "t_did": t_eff,
        "p_did": p_eff,
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
            "beta_DiD": fit["beta_did"],
            "se_DiD": fit["se_did"],
            "t_DiD": fit["t_did"],
            "p_DiD": fit["p_did"],
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

    # Store the participant-visit table for parallel-trends panel.
    return {
        "name": name,
        "effects": effect_df,
        "residuals": resid_df,
        "features": features,
        "n_per_group": n_per_group,
        "null_abs": null_abs,
        "pv": pv,
        "cfg": cfg,
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
                if fit is not None and np.isfinite(fit["beta_did"]):
                    out.append(abs(fit["beta_did"]))
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
                if fit is not None and np.isfinite(fit["beta_did"]):
                    out.append(abs(fit["beta_did"]))

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


def _panel_resid_fitted(ax, results: dict[str, dict]):
    """B: Residual vs fitted value scatter (all datasets overlaid)."""
    for name, res in results.items():
        df = res["residuals"]
        ax.scatter(
            df["fitted"], df["residual"],
            s=10, alpha=0.35,
            color=_DS_PALETTE.get(name, "grey"),
            label=name, rasterized=True,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Fitted", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


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
    # 4/n threshold per dataset
    for i, name in enumerate(df["Dataset"].unique()):
        n = results[name]["effects"]["n_obs"].median()
        if np.isfinite(n) and n > 0:
            thr = 4.0 / n
            ax.plot(
                [i - 0.35, i + 0.35], [thr, thr],
                color="red", lw=1.0, ls="--",
            )
    ax.set_title("Influence Diagnostics (Cook's D)", fontweight="bold")
    despine(ax)


def _panel_parallel_trends(fig, axes, results: dict[str, dict]):
    """D: Pre-treatment baseline check.

    Two-arm datasets: scatter of (control pre-mean, treated pre-mean) per
    feature.  Points near diagonal → parallel trends satisfied.

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
            subtitle = f"{name} — parallel trends"
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


def _panel_calibration(ax, results: dict[str, dict]):
    """E: Observed |beta| quantiles vs permutation null quantiles."""
    for name, res in results.items():
        obs = np.sort(np.abs(res["effects"]["beta_DiD"].dropna().values))
        null = np.sort(res.get("null_abs", np.array([])))
        if len(obs) < 3 or len(null) < 3:
            continue
        q = np.linspace(0.05, 0.95, min(len(obs), 200))
        obs_q = np.quantile(obs, q)
        null_q = np.quantile(null, q)
        ax.plot(
            null_q, obs_q, marker="o", ms=3.0, lw=1.2,
            color=_DS_PALETTE.get(name, "grey"), label=name,
        )
    lim = ax.get_xlim()
    hi = max(ax.get_ylim()[1], lim[1])
    ax.plot([0, hi], [0, hi], ls="--", color="black", lw=0.9)
    ax.set_xlabel("Permutation |beta| quantiles")
    ax.set_ylabel("Observed |beta| quantiles")
    ax.set_title("Calibration vs Permutation Null", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)



def _panel_normality_tests(ax, results: dict[str, dict]):
    """F: Visual summary of residual normality diagnostics per dataset."""
    rows = []
    for name, res in results.items():
        rdf = res["residuals"]
        vals = rdf["residual"].dropna().values
        if len(vals) < 8:
            continue

        sk = stats.skew(vals)
        ku = stats.kurtosis(vals)

        # Shapiro-Wilk on all residuals (no subsampling)
        sw_stat, sw_p = stats.shapiro(vals)

        rows.append({
            "Dataset": name,
            "Shapiro W": sw_stat,
            "Shapiro p": sw_p,
            "Skewness": sk,
            "Kurtosis": ku,
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

    # Main: Shapiro W statistic bars
    ax.barh(y, df["Shapiro W"], color=colors, edgecolor="white",
            height=0.6, alpha=0.85)
    for i, (_, row) in enumerate(df.iterrows()):
        label = f"W={row['Shapiro W']:.3f}, skew={row['Skewness']:.2f}"
        ax.text(row["Shapiro W"] + 0.002, i, label, va="center",
                fontsize=7, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["Dataset"], fontsize=9)
    ax.set_xlabel("Shapiro-Wilk W statistic")
    ax.set_title("Residual Normality Diagnostics", fontweight="bold")
    ax.axvline(0.95, color="red", ls="--", lw=0.8, alpha=0.7, label="W=0.95 threshold")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


def _panel_funnel(ax, results: dict[str, dict]):
    """G: Funnel plot — effect size vs standard error."""
    for name, res in results.items():
        df = res["effects"]
        ax.scatter(
            df["beta_DiD"], df["se_DiD"],
            s=45, alpha=0.8,
            color=_DS_PALETTE.get(name, "grey"),
            edgecolors="white", linewidth=0.5, label=name,
        )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.invert_yaxis()  # convention: SE=0 at top
    ax.set_xlabel("DiD effect (beta)")
    ax.set_ylabel("Standard error")
    ax.set_title("Effect Size vs Standard Error (Funnel)", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


# ── Panel H: Heteroscedasticity diagnostics ────────────────────────


def _panel_heteroscedasticity(ax, results: dict[str, dict]):
    """H: Breusch-Pagan test for residual homoscedasticity per dataset."""
    rows = []
    for name, res in results.items():
        rdf = res["residuals"]
        vals = rdf["residual"].dropna().values
        fitted = rdf["fitted"].dropna().values
        if len(vals) < 8 or len(fitted) < 8:
            continue
        # Align lengths
        n = min(len(vals), len(fitted))
        vals, fitted = vals[:n], fitted[:n]

        # Manual Breusch-Pagan: regress squared residuals on fitted values
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
            bp_stat = n * r2_bp  # chi-squared(1) under H0
            bp_p = 1.0 - stats.chi2.cdf(bp_stat, df=1)
        except Exception:
            continue

        rows.append({
            "Dataset": name,
            "BP stat": bp_stat,
            "BP p": bp_p,
        })

    if not rows:
        ax.text(0.5, 0.5, "No test results", ha="center", va="center",
                transform=ax.transAxes)
        return

    df = pd.DataFrame(rows)
    y = np.arange(len(df))
    colors = [_DS_PALETTE.get(n, "grey") for n in df["Dataset"]]

    ax.barh(y, df["BP stat"], color=colors, edgecolor="white",
            height=0.6, alpha=0.85)
    for i, (_, row) in enumerate(df.iterrows()):
        pval_str = f"p={row['BP p']:.2e}" if row["BP p"] < 0.001 else f"p={row['BP p']:.3f}"
        label = f"BP={row['BP stat']:.2f}, {pval_str}"
        ax.text(row["BP stat"] + 0.05, i, label, va="center",
                fontsize=7, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["Dataset"], fontsize=9)
    ax.set_xlabel("Breusch-Pagan test statistic")
    ax.set_title("Heteroscedasticity Diagnostics", fontweight="bold")
    ax.axvline(3.84, color="red", ls="--", lw=0.8, alpha=0.7,
               label="χ²(1) = 3.84 (α=0.05)")
    ax.legend(fontsize=7, frameon=True)
    despine(ax)


# ── Generate ──────────────────────────────────────────────────────


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

    # B: Residual vs fitted
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _panel_resid_fitted(ax, results)
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
    _panel_parallel_trends(fig, axes, results)
    fig.suptitle(
        "Baseline Diagnostics: Arm Means (two-arm) / Pre-Post Means (single-arm)",
        fontweight="bold", y=1.02, fontsize=11,
    )
    fig.tight_layout()
    save_panel(fig, "panel_D", FIGURE_NAME, SUPP_OUTPUT)

    # E: Calibration vs permutation null
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _panel_calibration(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_E", FIGURE_NAME, SUPP_OUTPUT)

    # F: Residual normality diagnostics
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    _panel_normality_tests(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_F", FIGURE_NAME, SUPP_OUTPUT)

    # G: Funnel plot
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    _panel_funnel(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_G", FIGURE_NAME, SUPP_OUTPUT)

    # H: Heteroscedasticity diagnostics (Breusch-Pagan)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    _panel_heteroscedasticity(ax, results)
    fig.tight_layout()
    save_panel(fig, "panel_H", FIGURE_NAME, SUPP_OUTPUT)

    # Cleanup
    results.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
