"""
Figure 4 — Statistical Robustness
==================================

Four-panel (2x2) figure demonstrating statistical robustness of the
Difference-in-Differences framework on the Sade-Feldman immunotherapy dataset.

Panels
------
A : Bootstrap vs analytical standard-error comparison
B : FDR calibration — cell-level vs participant-level p-values
C : Permutation null distribution (most significant signature)
D : Observed effects vs 95 % null range for all signatures
"""

from __future__ import annotations

import gc
import warnings

from .._shared import (
    COLORS,
    MAIN_OUTPUT,
    TrialDesign,
    apply_style,
    despine,
    did_table,
    get_sade_feldman,
    harmonize_response,
    np,
    pd,
    plt,
    save_panel,
    score_signatures,
    sig_display,
    stats,
)

warnings.filterwarnings("ignore")

FIGURE_NAME = "Figure4_statistical_robustness"
VISITS: tuple[str, str] = ("Pre", "Post")
N_PERM = 100
N_BOOT = 200


# ── data preparation ─────────────────────────────────────────────────────

def _prepare_data() -> dict:
    """Load data and run all analyses required by the four panels."""
    # 1. Load and score -------------------------------------------------
    adata = get_sade_feldman()
    adata = harmonize_response(adata)

    # Ensure log1p_tpm layer exists
    if "log1p_tpm" not in adata.layers:
        import scanpy as sc

        if "tpm" in adata.layers:
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        elif "counts" in adata.layers:
            sc.pp.normalize_total(adata, target_sum=1e6, layer="counts",
                                  key_added="tpm")
            adata.layers["log1p_tpm"] = np.log1p(adata.layers["tpm"])
        else:
            adata.layers["log1p_tpm"] = np.log1p(
                adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            )

    adata, sig_cols = score_signatures(adata)
    print(f"  Scored {len(sig_cols)} signatures")

    # 2. Trial design ---------------------------------------------------
    design = TrialDesign(
        participant_col="participant_id",
        visit_col="visit",
        arm_col="response_harmonized",
        arm_treated="Responder",
        arm_control="Non-responder",
    )

    common_kw = dict(
        features=sig_cols,
        design=design,
        visits=VISITS,
        layer="log1p_tpm",
        standardize=True,
    )

    # 3. Cell-level DiD -------------------------------------------------
    print("  Running cell-level DiD ...")
    df_cell = did_table(adata, aggregate="cell", **common_kw)

    # 4. Participant-level DiD (analytical) -----------------------------
    print("  Running participant-level DiD ...")
    df_part = did_table(adata, aggregate="participant_visit", **common_kw)

    # 5. Participant-level DiD (wild cluster bootstrap) -----------------
    print("  Running bootstrap DiD ...")
    df_boot = did_table(
        adata, aggregate="participant_visit", use_bootstrap=True,
        n_boot=N_BOOT, **common_kw,
    )

    # 6. Pairs cluster bootstrap for SE estimation ----------------------
    #    Resample participants with replacement, refit OLS each time,
    #    and take the SD of beta_DiD across resamples as bootstrap SE.
    print("  Running pairs cluster bootstrap for SE estimation ...")
    boot_betas = {feat: [] for feat in sig_cols}
    np.random.seed(42)
    participants = adata.obs["participant_id"].unique()

    # Build pseudobulk once for fast resampling
    _pb_data = (
        adata.obs[["participant_id", "visit", "response"] + sig_cols]
        .groupby(["participant_id", "visit", "response"], observed=True)[sig_cols]
        .mean()
        .reset_index()
    )

    n_success = 0
    for b in range(N_BOOT):
        boot_pids = np.random.choice(participants, size=len(participants),
                                     replace=True)
        # Resample pseudobulk rows (much faster than resampling cells)
        boot_rows = []
        pid_counts: dict[str, int] = {}
        for pid in boot_pids:
            pid_counts[pid] = pid_counts.get(pid, 0) + 1
            sub = _pb_data[_pb_data["participant_id"] == pid].copy()
            sub["participant_id"] = f"{pid}__{pid_counts[pid]}"
            boot_rows.append(sub)
        boot_pb = pd.concat(boot_rows, ignore_index=True)

        try:
            # Manual OLS DiD on pseudobulk
            boot_pb["visit_num"] = (boot_pb["visit"] == "Post").astype(int)
            boot_pb["arm_num"] = (boot_pb["response"] == "Responder").astype(int)
            boot_pb["interaction"] = boot_pb["visit_num"] * boot_pb["arm_num"]

            for feat in sig_cols:
                y = boot_pb[feat].values
                X = np.column_stack([
                    np.ones(len(boot_pb)),
                    boot_pb["visit_num"].values,
                    boot_pb["arm_num"].values,
                    boot_pb["interaction"].values,
                ])
                if np.any(np.isnan(y)):
                    continue
                try:
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    if np.isfinite(beta[3]):
                        boot_betas[feat].append(beta[3])
                except np.linalg.LinAlgError:
                    pass
            n_success += 1
        except Exception:
            pass

    print(f"  Bootstrap: {n_success}/{N_BOOT} successful replicates")
    boot_se = {
        feat: np.nanstd(vals) if len(vals) >= 5 else np.nan
        for feat, vals in boot_betas.items()
    }

    # 7. Permutation test -----------------------------------------------
    print("  Running permutation test ...")
    np.random.seed(42)
    original_response = adata.obs["response"].copy()
    perm_results: list[pd.DataFrame] = []

    for i in range(N_PERM):
        pid_response = adata.obs.groupby("participant_id")["response"].first()
        shuffled = pid_response.sample(frac=1, replace=False)
        shuffled.index = pid_response.index
        adata.obs["response"] = adata.obs["participant_id"].map(shuffled)
        try:
            df_perm = did_table(
                adata, aggregate="participant_visit", **common_kw,
            )
            df_perm["permutation"] = i
            perm_results.append(df_perm)
        except Exception:
            pass

    adata.obs["response"] = original_response  # restore

    df_perm_all = pd.concat(perm_results, ignore_index=True)
    print(f"  Completed {df_perm_all['permutation'].nunique()} permutations")

    return {
        "df_cell": df_cell,
        "df_part": df_part,
        "df_boot": df_boot,
        "boot_se": boot_se,
        "df_perm_all": df_perm_all,
        "sig_cols": sig_cols,
        "adata": adata,
    }


# ── Panel A: Bootstrap vs Analytical SE ──────────────────────────────────

def _panel_a(ax, data: dict) -> None:
    """Scatter of analytical SE vs pairs-cluster-bootstrap SE."""
    df_part = data["df_part"]
    boot_se = data["boot_se"]

    feats, analytical, bootstrap = [], [], []
    for _, row in df_part.iterrows():
        feat = row["feature"]
        se_an = row["se_DiD"]
        se_bt = boot_se.get(feat, np.nan)
        if np.isfinite(se_an) and np.isfinite(se_bt):
            feats.append(feat)
            analytical.append(se_an)
            bootstrap.append(se_bt)

    analytical = np.array(analytical)
    bootstrap = np.array(bootstrap)

    if len(analytical) == 0:
        ax.text(0.5, 0.5, "Insufficient bootstrap\nreplicates for comparison",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Bootstrap vs Analytical SE", fontweight="bold")
        despine(ax)
        return

    # Identity line
    lo = min(analytical.min(), bootstrap.min()) * 0.85
    hi = max(analytical.max(), bootstrap.max()) * 1.15
    ax.plot([lo, hi], [lo, hi], ls="--", color=COLORS["gray"], lw=1, zorder=1)

    # Scatter
    ax.scatter(analytical, bootstrap, s=50, color=COLORS["treated"],
               edgecolor="white", linewidth=0.5, zorder=3)

    # Labels
    for feat, x, y in zip(feats, analytical, bootstrap):
        ax.annotate(
            sig_display(feat), (x, y),
            fontsize=6.5, ha="left", va="bottom",
            xytext=(3, 3), textcoords="offset points",
        )

    # Correlation
    r, p = stats.pearsonr(analytical, bootstrap)
    ax.text(
        0.05, 0.95, f"r = {r:.2f}\np = {p:.1e}",
        transform=ax.transAxes, fontsize=8,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
    )

    ax.set_xlabel("Analytical SE")
    ax.set_ylabel("Bootstrap SE")
    ax.set_title("Bootstrap vs Analytical SE", fontsize=10)
    despine(ax)


# ── Panel B: FDR Calibration (cell vs participant) ───────────────────────

def _panel_b(ax, data: dict) -> None:
    """Horizontal bar chart comparing -log10(p) at cell vs participant level."""
    df_cell = data["df_cell"].copy()
    df_part = data["df_part"].copy()

    # Merge on feature
    merged = df_cell[["feature", "p_DiD"]].merge(
        df_part[["feature", "p_DiD"]], on="feature", suffixes=("_cell", "_part"),
    )
    merged["display"] = merged["feature"].apply(sig_display)
    merged["nlog10_cell"] = -np.log10(merged["p_DiD_cell"].clip(lower=1e-300))
    merged["nlog10_part"] = -np.log10(merged["p_DiD_part"].clip(lower=1e-300))
    merged = merged.sort_values("nlog10_part", ascending=True)

    y_pos = np.arange(len(merged))
    bar_h = 0.35

    ax.barh(y_pos - bar_h / 2, merged["nlog10_cell"].values,
            height=bar_h, color=COLORS["highlight"], alpha=0.8,
            label="Cell-level", edgecolor="none")
    ax.barh(y_pos + bar_h / 2, merged["nlog10_part"].values,
            height=bar_h, color=COLORS["treated"], alpha=0.8,
            label="Participant-level", edgecolor="none")

    # Significance thresholds
    ax.axvline(-np.log10(0.05), ls="--", color=COLORS["gray"], lw=0.8,
               label="p = 0.05")
    ax.axvline(-np.log10(0.01), ls=":", color=COLORS["gray"], lw=0.8,
               label="p = 0.01")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["display"].values, fontsize=8)
    ax.set_xlabel("$-\\log_{10}(p)$")
    ax.set_title("Cell vs Participant Inference", fontsize=10)
    ax.legend(fontsize=7, loc="lower right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel C: Permutation Null Distribution ───────────────────────────────

def _panel_c(ax, data: dict) -> None:
    """Permutation null distributions overlaid for top 3 signatures."""
    df_part = data["df_part"]
    df_perm = data["df_perm_all"]
    sig_cols = data["sig_cols"]

    # Compute permutation p-values for all signatures
    perm_pvals = {}
    for feat in sig_cols:
        null_betas = df_perm.loc[df_perm["feature"] == feat, "beta_DiD"].dropna()
        if len(null_betas) == 0:
            continue
        obs_row = df_part.loc[df_part["feature"] == feat]
        if obs_row.empty:
            continue
        obs_beta = obs_row["beta_DiD"].values[0]
        perm_p = (np.abs(null_betas) >= np.abs(obs_beta)).mean()
        perm_pvals[feat] = perm_p

    if not perm_pvals:
        ax.text(0.5, 0.5, "No permutation results",
                ha="center", va="center", transform=ax.transAxes)
        despine(ax)
        return

    # Show top 3 most significant signatures
    top_feats = sorted(perm_pvals, key=perm_pvals.get)[:3]
    hist_colors = [COLORS["treated"], COLORS["control"], COLORS["neutral"]]

    for idx, feat in enumerate(top_feats):
        perm_p = perm_pvals[feat]
        obs_beta = df_part.loc[df_part["feature"] == feat, "beta_DiD"].values[0]
        null_betas = df_perm.loc[df_perm["feature"] == feat, "beta_DiD"].dropna()
        color = hist_colors[idx % len(hist_colors)]

        ax.hist(null_betas, bins=20, color=color, alpha=0.35,
                edgecolor="none", density=True, zorder=2,
                label=f"{sig_display(feat)} null")

        # Observed effect as vertical line
        ax.axvline(obs_beta, color=color, lw=2, ls="-", zorder=4)
        # Small label near the line
        ax.text(obs_beta, ax.get_ylim()[1] * 0.9 - idx * 0.15 * ax.get_ylim()[1],
                f"p={perm_p:.3f}", fontsize=7, color=color, ha="left",
                fontweight="bold")

    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (null distribution)")
    ax.set_ylabel("Density")
    ax.set_title("Permutation Null Distributions (Top 3)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right", frameon=True, framealpha=0.9)
    despine(ax)


# ── Panel D: Observed Effects vs Null Range ──────────────────────────────

def _panel_d(ax, data: dict) -> None:
    """Observed beta_DiD with 95 % null interval for every signature."""
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
        records.append({
            "feature": feat,
            "display": sig_display(feat),
            "obs_beta": obs_beta,
            "null_lo": lo,
            "null_hi": hi,
            "significant": (obs_beta < lo) or (obs_beta > hi),
        })

    rec_df = pd.DataFrame(records).sort_values("obs_beta", ascending=True)
    y_pos = np.arange(len(rec_df))

    # 95 % null bands (gray)
    for i, (_, row) in enumerate(rec_df.iterrows()):
        ax.barh(i, row["null_hi"] - row["null_lo"],
                left=row["null_lo"], height=0.6,
                color=COLORS["gray"], alpha=0.25, edgecolor="none", zorder=1)

    # Observed dots
    colors = [
        COLORS["highlight"] if sig else COLORS["treated"]
        for sig in rec_df["significant"]
    ]
    ax.scatter(rec_df["obs_beta"], y_pos, c=colors, s=55,
               edgecolor="white", linewidth=0.5, zorder=3)

    # Zero line
    ax.axvline(0, ls=":", color=COLORS["gray"], lw=0.8, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rec_df["display"].values, fontsize=8)
    ax.set_xlabel(r"$\beta_{\mathrm{DiD}}$ (standardized)")
    ax.set_title("Observed Effects vs Null Range", fontsize=10)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["highlight"],
               markersize=7, label="Outside 95% null"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["treated"],
               markersize=7, label="Within 95% null"),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS["gray"], alpha=0.25,
                       label="95% null range"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right",
              frameon=True, framealpha=0.9)
    despine(ax)


# ── Composite generation ─────────────────────────────────────────────────

def generate() -> None:
    """Create and save Figure 4 individual panels."""
    apply_style()
    data = _prepare_data()

    # ── Save individual panels ------------------------------------------
    panel_funcs = [
        ("panel_A_bootstrap_vs_analytical_se", _panel_a),
        ("panel_B_fdr_calibration", _panel_b),
        ("panel_C_permutation_null", _panel_c),
        ("panel_D_observed_vs_null_range", _panel_d),
    ]
    for panel_name, func in panel_funcs:
        fig_p, ax_p = plt.subplots(figsize=(6.5, 5))
        func(ax_p, data)
        save_panel(fig_p, panel_name, FIGURE_NAME, MAIN_OUTPUT)

    # ── Cleanup ---------------------------------------------------------
    adata = data.get("adata")
    if adata is not None:
        del adata
    del data
    gc.collect()

    print(f"  Figure 4 complete: {FIGURE_NAME}")
