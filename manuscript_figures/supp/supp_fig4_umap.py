"""
Supplementary Figure 3 - Model diagnostics and assumption checks.

Panels:
  A  Q-Q plots of true model residuals (per dataset).
  B  Effect size vs standard error (funnel plot).
  C  Residual vs fitted plot.
  D  Influence diagnostics (Cook's distance).
  E  Calibration against permutation null.
  F  Residual summary table.
  G  Minimum detectable effect vs sample size.
  H  Effect-size distributions by dataset.
"""

from __future__ import annotations

import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .._shared import (
    SUPP_OUTPUT,
    apply_style,
    clear_cache,
    despine,
    get_sade_feldman,
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
    "Melanoma": {
        "loader": lambda: load_clinical_trial_dataset("melanoma"),
        "harmonize": False,
        "layer": "log1p_tpm",
        "participant_col": "participant_id",
        "visit_col": "visit",
        "arm_col": "response",
        "arm_treated": "Post_Treatment",
        "arm_control": "Treatment_Naive",
        "visits": ("Pre", "Post"),
    },
    # CAR-T intentionally excluded: no valid two-arm contrast.
}

_DS_PALETTE = dict(zip(_DATASET_CFG.keys(), sns.color_palette("Set2", len(_DATASET_CFG))))


def _matrix_from_layer(adata, features: list[str], layer: str) -> np.ndarray:
    X = adata[:, features].layers[layer] if layer in adata.layers else adata[:, features].X
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _participant_visit_means(adata, cfg: dict, features: list[str]) -> pd.DataFrame:
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg["arm_col"]

    mat = _matrix_from_layer(adata, features, cfg["layer"])
    expr = pd.DataFrame(mat, columns=features, index=adata.obs_names)
    expr[pid_col] = adata.obs[pid_col].values
    expr[visit_col] = adata.obs[visit_col].values
    expr[arm_col] = adata.obs[arm_col].values

    pv = (
        expr.groupby([pid_col, visit_col, arm_col], observed=True)[features]
        .mean()
        .reset_index()
    )

    # Keep only paired participants.
    counts = pv.groupby(pid_col)[visit_col].nunique()
    paired = counts[counts >= 2].index
    pv = pv[pv[pid_col].isin(paired)].copy()
    return pv


def _ols_interaction(y: np.ndarray, post: np.ndarray, treated: np.ndarray):
    X = np.column_stack([np.ones_like(post), post, treated, post * treated])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    dof = n - p
    if dof <= 0:
        return None

    rss = float(np.sum(resid ** 2))
    sigma2 = rss / dof if dof > 0 else np.nan

    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0))

    beta_did = float(beta[3])
    se_did = float(se[3])
    if not np.isfinite(se_did) or se_did <= 0:
        t_did = np.nan
        p_did = np.nan
    else:
        t_did = beta_did / se_did
        p_did = 2 * stats.t.sf(np.abs(t_did), dof)

    # Influence diagnostics.
    hat = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        cooks = (resid ** 2 / (p * sigma2)) * (hat / (1.0 - hat) ** 2)
    cooks[~np.isfinite(cooks)] = np.nan

    return {
        "beta_did": beta_did,
        "se_did": se_did,
        "t_did": t_did,
        "p_did": p_did,
        "resid": resid,
        "fitted": fitted,
        "hat": hat,
        "cooks": cooks,
        "sigma": float(np.sqrt(sigma2)) if sigma2 >= 0 else np.nan,
        "n_obs": int(n),
    }


def _fit_dataset(name: str, adata, cfg: dict) -> dict | None:
    features = [f for f in _TEST_FEATURES if f in adata.var_names]
    if len(features) < 4:
        return None

    pv = _participant_visit_means(adata, cfg, features)
    if pv.empty:
        return None

    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg["arm_col"]
    pre_v, post_v = cfg["visits"]

    # Restrict to requested visits and two valid arms.
    pv = pv[pv[visit_col].isin([pre_v, post_v])].copy()
    pv = pv[pv[arm_col].isin([cfg["arm_treated"], cfg["arm_control"]])].copy()
    if pv.empty:
        return None

    arm_counts = pv.groupby(arm_col)[pid_col].nunique()
    if cfg["arm_treated"] not in arm_counts or cfg["arm_control"] not in arm_counts:
        return None
    n_per_group = int(min(arm_counts[cfg["arm_treated"]], arm_counts[cfg["arm_control"]]))
    if n_per_group < 2:
        return None

    post = (pv[visit_col].values == post_v).astype(float)
    treated = (pv[arm_col].values == cfg["arm_treated"]).astype(float)

    effect_rows = []
    resid_rows = []
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

        tmp = pv[[pid_col, visit_col, arm_col]].copy()
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

    # Permutation null on interaction effects.
    null_abs = _permutation_null(pv, features, cfg, n_perm=200)

    return {
        "name": name,
        "effects": effect_df,
        "residuals": resid_df,
        "features": features,
        "n_per_group": n_per_group,
        "null_abs": null_abs,
    }


def _permutation_null(pv: pd.DataFrame, features: list[str], cfg: dict, n_perm: int = 200) -> np.ndarray:
    pid_col = cfg["participant_col"]
    visit_col = cfg["visit_col"]
    arm_col = cfg["arm_col"]
    pre_v, post_v = cfg["visits"]

    post = (pv[visit_col].values == post_v).astype(float)

    pid_df = pv[[pid_col, arm_col]].drop_duplicates(subset=[pid_col]).copy()
    orig = (pid_df[arm_col].values == cfg["arm_treated"]).astype(float)
    pids = pid_df[pid_col].values

    rng = np.random.default_rng(42)
    out = []
    for _ in range(n_perm):
        perm = rng.permutation(orig)
        map_df = pd.DataFrame({pid_col: pids, "treated_perm": perm})
        treated = pv[[pid_col]].merge(map_df, on=pid_col, how="left")["treated_perm"].values.astype(float)

        vals = []
        for feat in features:
            fit = _ols_interaction(pv[feat].values.astype(float), post, treated)
            if fit is not None and np.isfinite(fit["beta_did"]):
                vals.append(abs(fit["beta_did"]))
        if vals:
            out.extend(vals)

    return np.asarray(out, dtype=float)


def _load_results() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, cfg in _DATASET_CFG.items():
        try:
            adata = cfg["loader"]()
            if cfg.get("harmonize", False):
                adata = harmonize_response(adata)
            res = _fit_dataset(name, adata, cfg)
            if res is not None:
                out[name] = res
                print(f"  {name}: {len(res['effects'])} features, n/group={res['n_per_group']}")
            else:
                print(f"  {name}: skipped (insufficient paired two-arm data)")
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
    return out


def _panel_qq(fig, axes, results: dict[str, dict]):
    names = list(results)
    for i, ax in enumerate(np.ravel(axes)):
        if i >= len(names):
            ax.axis("off")
            continue
        name = names[i]
        vals = results[name]["residuals"]["residual"].dropna().values
        if len(vals) < 4:
            ax.text(0.5, 0.5, "No residuals", ha="center", va="center", transform=ax.transAxes)
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


def _panel_funnel(ax, results: dict[str, dict]):
    for name, res in results.items():
        df = res["effects"]
        ax.scatter(
            df["beta_DiD"], df["se_DiD"], s=45, alpha=0.8,
            color=_DS_PALETTE.get(name, "grey"), edgecolors="white", linewidth=0.5, label=name
        )
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("DiD effect (beta)")
    ax.set_ylabel("Standard error")
    ax.set_title("Effect Size vs Standard Error", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_resid_vs_fitted(ax, results: dict[str, dict]):
    for name, res in results.items():
        df = res["residuals"]
        ax.scatter(
            df["fitted"], df["residual"], s=10, alpha=0.35,
            color=_DS_PALETTE.get(name, "grey"), label=name
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Fitted", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_influence(ax, results: dict[str, dict]):
    rows = []
    for name, res in results.items():
        d = res["residuals"]["cooks_d"].dropna()
        for v in d.values:
            rows.append({"Dataset": name, "Cook's D": float(v)})
    if not rows:
        ax.text(0.5, 0.5, "No influence data", ha="center", va="center", transform=ax.transAxes)
        return
    df = pd.DataFrame(rows)
    sns.boxplot(data=df, x="Dataset", y="Cook's D", ax=ax, palette=_DS_PALETTE, fliersize=1.5)
    sns.stripplot(data=df, x="Dataset", y="Cook's D", ax=ax, color="black", size=2.0, alpha=0.25, jitter=0.2)
    # conservative generic threshold around 4/n
    for i, name in enumerate(df["Dataset"].unique()):
        n = results[name]["effects"]["n_obs"].median()
        if np.isfinite(n) and n > 0:
            thr = 4.0 / n
            ax.plot([i - 0.35, i + 0.35], [thr, thr], color="red", lw=1.0, ls="--")
    ax.set_title("Influence Diagnostics (Cook's D)", fontweight="bold")
    despine(ax)


def _panel_calibration(ax, results: dict[str, dict]):
    for name, res in results.items():
        obs = np.sort(np.abs(res["effects"]["beta_DiD"].dropna().values))
        null = np.sort(res.get("null_abs", np.array([])))
        if len(obs) < 3 or len(null) < 3:
            continue
        q = np.linspace(0.05, 0.95, min(len(obs), 200))
        obs_q = np.quantile(obs, q)
        null_q = np.quantile(null, q)
        ax.plot(null_q, obs_q, marker="o", ms=3.0, lw=1.2,
                color=_DS_PALETTE.get(name, "grey"), label=name)
    lim = ax.get_xlim()
    hi = max(ax.get_ylim()[1], lim[1])
    ax.plot([0, hi], [0, hi], ls="--", color="black", lw=0.9)
    ax.set_xlabel("Permutation |beta| quantiles")
    ax.set_ylabel("Observed |beta| quantiles")
    ax.set_title("Calibration vs Permutation Null", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_resid_summary(ax, results: dict[str, dict]):
    rows = []
    cols = ["Dataset", "N residuals", "Mean", "Std", "Skew", "Kurtosis"]
    for name, res in results.items():
        vals = res["residuals"]["residual"].dropna().values
        if len(vals) < 3:
            continue
        rows.append([
            name,
            f"{len(vals)}",
            f"{np.mean(vals):.4f}",
            f"{np.std(vals):.4f}",
            f"{stats.skew(vals):.2f}",
            f"{stats.kurtosis(vals):.2f}",
        ])
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "No residual summary", ha="center", va="center", transform=ax.transAxes)
        return
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.8)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Residual Summary (True Model Residuals)", fontweight="bold", pad=12)


def _mde(n: np.ndarray, sigma: float, alpha: float = 0.05, power: float = 0.8) -> np.ndarray:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * sigma * np.sqrt(2.0 / n)


def _panel_mde(ax, results: dict[str, dict]):
    n_grid = np.arange(4, 61, 2)
    for name, res in results.items():
        sigma = float(np.nanmedian(res["effects"]["sigma"].values))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        y = _mde(n_grid, sigma)
        ax.plot(n_grid, y, lw=1.8, color=_DS_PALETTE.get(name, "grey"), label=name)
        n_actual = res["n_per_group"]
        if n_actual > 0:
            ax.scatter([n_actual], [_mde(np.array([n_actual]), sigma)[0]],
                       color=_DS_PALETTE.get(name, "grey"), edgecolors="black", zorder=5, s=45)
    ax.set_xlabel("Participants per arm")
    ax.set_ylabel("Minimum detectable effect")
    ax.set_title("Power Curve with Observed Cohort Sizes", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def _panel_effect_hist(ax, results: dict[str, dict]):
    for name, res in results.items():
        vals = res["effects"]["beta_DiD"].dropna().values
        ax.hist(vals, bins=15, alpha=0.5, label=name,
                color=_DS_PALETTE.get(name, "grey"), edgecolor="white")
    ax.axvline(0, color="black", lw=0.9, ls="--")
    ax.set_xlabel("DiD effect (beta)")
    ax.set_ylabel("Count")
    ax.set_title("Effect-Size Distribution", fontweight="bold")
    ax.legend(fontsize=8, frameon=True)
    despine(ax)


def generate():
    print("Supplementary Figure 3: Model Diagnostics and Assumption Checks")
    results = _load_results()
    if not results:
        print("  No valid datasets found; skipping.")
        return

    names = list(results)

    # A: residual QQ per dataset
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2))
    if len(names) == 1:
        axes = np.array([axes])
    _panel_qq(fig, axes, results)
    fig.tight_layout()
    save_panel(fig, "panel_A", FIGURE_NAME, SUPP_OUTPUT)

    # B-H single panels
    panel_specs = [
        ("panel_B", _panel_funnel, (7.0, 5.8)),
        ("panel_C", _panel_resid_vs_fitted, (7.0, 5.8)),
        ("panel_D", _panel_influence, (7.4, 5.8)),
        ("panel_E", _panel_calibration, (7.0, 5.8)),
        ("panel_F", _panel_resid_summary, (8.8, 4.8)),
        ("panel_G", _panel_mde, (7.0, 5.8)),
        ("panel_H", _panel_effect_hist, (7.0, 5.8)),
    ]

    for panel_name, fn, size in panel_specs:
        fig, ax = plt.subplots(figsize=size)
        fn(ax, results)
        fig.tight_layout()
        save_panel(fig, panel_name, FIGURE_NAME, SUPP_OUTPUT)

    results.clear()
    clear_cache()
    gc.collect()
    print("  Done.\n")


if __name__ == "__main__":
    apply_style()
    generate()
