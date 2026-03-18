from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from anndata import AnnData
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ..design import TrialDesign
from ._utils import encode_visit

__all__ = [
    "module_score_pseudobulk",
    "module_score_did_by_pool",
    "module_score_within_arm_by_pool",
]


def _map_pool(celltype: str, pool_map: dict[str, Sequence[str]] | None) -> str | None:
    """Map celltype to pool using the pool_map."""
    if pool_map is None:
        return None
    for pool, labels in pool_map.items():
        if celltype in labels:
            return pool
    return None


def module_score_pseudobulk(
    adata: AnnData,
    module_cols: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    pool_col: str | None = None,
    pool_map: dict[str, Sequence[str]] | None = None,
    celltype_col: str | None = None,
    min_cells_per_group: int = 5,
    exclude_crossovers: bool = True,
) -> pd.DataFrame:
    """Build pseudobulk module scores (participant × visit × pool × module).

    Parameters
    ----------
    adata
        AnnData object.
    module_cols
        Columns in adata.obs with module scores.
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    pool_col
        Column in adata.obs to use as pool labels directly.
    pool_map
        Mapping of pool -> list of celltypes. Used if pool_col is None.
    celltype_col
        Column for fine cell types (used with pool_map).
    min_cells_per_group
        Minimum cells per participant×visit×pool to keep.
    exclude_crossovers
        Whether to exclude crossover cells.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: participant, visit, arm, pool,
        module, ``module_score``, and ``n_cells``.
    """
    df = adata.obs.copy()
    if exclude_crossovers and design.crossover_col and design.crossover_col in df.columns:
        df = df[df[design.crossover_col] == 0].copy()

    df = df[df[design.visit_col].isin(visits)].copy()

    missing = [c for c in module_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Module score columns missing: {missing[:5]}")

    if pool_col is None:
        if pool_map is None or celltype_col is None:
            raise ValueError("Provide pool_col or (pool_map + celltype_col).")
        df["pool"] = df[celltype_col].apply(lambda x: _map_pool(x, pool_map))
    else:
        if pool_col not in df.columns:
            raise KeyError(f"pool_col '{pool_col}' not found in adata.obs")
        df["pool"] = df[pool_col]

    df = df.dropna(subset=["pool"])

    # Build column lists conditionally — arm_col may be None for single-arm
    id_cols = [design.participant_col, design.visit_col]
    if design.arm_col is not None:
        df[design.arm_col] = df[design.arm_col].astype(str).str.strip()
        id_cols.append(design.arm_col)

    # Long format: participant × visit × [arm] × pool × module
    melt_id = id_cols + ["pool"]
    long_df = df[
        melt_id + list(module_cols)
    ].melt(
        id_vars=melt_id,
        value_vars=list(module_cols),
        var_name="module",
        value_name="module_score",
    )
    long_df = long_df.dropna(subset=["module_score"])

    # Pseudobulk aggregation
    pb = (
        long_df.groupby(
            melt_id + ["module"],
            observed=True,
        )["module_score"]
        .mean()
        .reset_index()
    )
    pb = pb[pb["module_score"].notna()].copy()

    counts = (
        df.groupby(
            id_cols + ["pool"], observed=True
        )
        .size()
        .reset_index(name="n_cells")
    )

    pb = pb.merge(
        counts,
        on=id_cols + ["pool"],
        how="left",
    )
    pb = pb[pb["n_cells"] >= min_cells_per_group].copy()
    return pb


def _perm_test_diff(
    delta: pd.Series,
    arms: pd.Series,
    n_perm: int,
    seed: int,
    treated_label: str | None = None,
) -> float:
    """Permutation test for difference between arm-level mean deltas.

    Parameters
    ----------
    delta
        Participant-level change scores (post − pre).
    arms
        Arm labels aligned to *delta*.
    n_perm
        Number of random permutations.
    seed
        Random seed for reproducibility.
    treated_label
        Label identifying the treated arm.  If ``None``, the first
        element of *arms* is used.

    Returns
    -------
    float
        Two-sided permutation p-value in ``[0, 1]``.
    """
    rng = np.random.default_rng(seed)
    values = delta.to_numpy()
    labels = arms.to_numpy()
    if treated_label is None:
        treated_label = labels[0]
    obs = values[labels == treated_label].mean() - values[labels != treated_label].mean()
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(np.asarray(labels))
        perm_diff = values[perm == treated_label].mean() - values[perm != treated_label].mean()
        if np.abs(perm_diff) >= np.abs(obs):
            count += 1
    return float((count + 1) / (n_perm + 1))


def module_score_did_by_pool(
    pb: pd.DataFrame,
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    min_paired: int = 2,
    n_perm: int = 1000,
    seed: int = 42,
    fdr_within: str | None = "module",
    fdr_global: bool = True,
    allow_unpaired: bool = False,
) -> pd.DataFrame:
    """Compute DiD on module scores by pool with permutation p-values.

    Parameters
    ----------
    pb
        Output of module_score_pseudobulk().
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    n_perm
        Number of permutations for DiD p-values.
    seed
        Random seed.
    fdr_within
        If "module", FDR is computed within each module across pools.
        If "pool", FDR is computed within each pool across modules.
        If None, global FDR.
    fdr_global
        Only used when *fdr_within* is not None.  If True (default),
        an additional ``FDR_DiD_global`` column is added that applies
        BH-FDR correction globally across **all** tests, mirroring
        the behaviour when ``fdr_within=None``.  This is useful because
        per-group FDR (``FDR_DiD``) controls the false discovery rate
        only within each group and does **not** control the overall
        false discovery rate across all tests.
    allow_unpaired
        If True, fit an unpaired OLS DiD (module_score ~ visit + arm + visit×arm)
        using all available participant-visit observations. This is a fallback
        when no paired participants exist and should be interpreted cautiously.

    Returns
    -------
    pd.DataFrame
        One row per (pool, module) with columns: ``pool``, ``module``,
        ``mean_delta_treated``, ``mean_delta_control``, ``beta_DiD``,
        ``p_DiD``, ``p_treated``, ``p_control``, ``n_units``, ``FDR_DiD``.
        When *fdr_within* is set and *fdr_global* is True, an additional
        ``FDR_DiD_global`` column contains the globally-corrected q-values.
    """
    rows: list[dict[str, Any]] = []
    arm_treated = str(design.arm_treated).strip()
    arm_control = str(design.arm_control).strip()

    for (pool, module), sub in pb.groupby(["pool", "module"], observed=True):
        sub = sub[sub[design.visit_col].isin(visits)].copy()
        sub[design.arm_col] = sub[design.arm_col].astype(str).str.strip()
        sub = sub[sub[design.arm_col].isin([arm_treated, arm_control])].copy()
        if sub[design.visit_col].nunique() < 2 or sub[design.arm_col].nunique() < 2:
            continue

        if allow_unpaired:
            df = encode_visit(sub.copy(), design.visit_col, visits).reset_index(drop=True)
            df["arm_bin"] = (df[design.arm_col] == arm_treated).astype(int)
            model = smf.ols("module_score ~ visit_num + arm_bin + visit_num:arm_bin", data=df)
            fit = model.fit(cov_type="HC1")

            # Compute mean deltas from unpaired group means
            treated = df[df["arm_bin"] == 1]
            control = df[df["arm_bin"] == 0]
            treated_means = treated.groupby(design.visit_col)["module_score"].mean()
            control_means = control.groupby(design.visit_col)["module_score"].mean()
            if visits[0] not in treated_means or visits[1] not in treated_means:
                continue
            if visits[0] not in control_means or visits[1] not in control_means:
                continue
            mean_delta_treated = float(treated_means.loc[visits[1]] - treated_means.loc[visits[0]])
            mean_delta_control = float(control_means.loc[visits[1]] - control_means.loc[visits[0]])

            # Within-arm visit effect (unpaired OLS) for reporting p-values
            p_treated = np.nan
            p_control = np.nan
            try:
                fit_t = smf.ols("module_score ~ visit_num", data=treated).fit(cov_type="HC1")
                p_treated = float(fit_t.pvalues.get("visit_num", np.nan))
            except (ValueError, TypeError):
                p_treated = np.nan
            try:
                fit_c = smf.ols("module_score ~ visit_num", data=control).fit(cov_type="HC1")
                p_control = float(fit_c.pvalues.get("visit_num", np.nan))
            except (ValueError, TypeError):
                p_control = np.nan

            # Report effective participant count after model row drops
            model_row_idx = fit.model.data.row_labels
            n_units_eff = int(df[design.participant_col].loc[model_row_idx].nunique())
            rows.append(
                {
                    "pool": pool,
                    "module": module,
                    "mean_delta_treated": mean_delta_treated,
                    "mean_delta_control": mean_delta_control,
                    "beta_DiD": float(fit.params.get("visit_num:arm_bin", np.nan)),
                    "p_DiD": float(fit.pvalues.get("visit_num:arm_bin", np.nan)),
                    "p_treated": p_treated,
                    "p_control": p_control,
                    "n_units": n_units_eff,
                }
            )
            continue

        wide = sub.pivot_table(
            index=design.participant_col,
            columns=design.visit_col,
            values="module_score",
            aggfunc="mean",
        )
        if visits[0] not in wide.columns or visits[1] not in wide.columns:
            continue
        wide = wide.dropna()
        if len(wide) < min_paired:
            continue

        # attach arm labels
        pid_arm = sub.groupby(design.participant_col)[design.arm_col].first().to_dict()
        wide["arm"] = wide.index.map(pid_arm)

        # compute deltas
        wide["delta"] = wide[visits[1]] - wide[visits[0]]
        deltas = wide.dropna(subset=["delta"])

        if deltas["arm"].nunique() < 2:
            continue

        # paired within-arm tests
        p_arm = {}
        for arm_label in deltas["arm"].unique():
            sub_arm = deltas[deltas["arm"] == arm_label]
            if len(sub_arm) >= 3:
                delta_vals = sub_arm[visits[1]].values - sub_arm[visits[0]].values
                if np.allclose(delta_vals, 0):
                    p = 1.0
                else:
                    try:
                        _, p = wilcoxon(delta_vals)
                    except (ValueError, TypeError):
                        p = np.nan
            else:
                p = np.nan
            p_arm[arm_label] = p

        arm_means = deltas.groupby("arm", observed=True)["delta"].mean()
        mean_delta_treated = float(arm_means.get(arm_treated, np.nan))
        mean_delta_control = float(arm_means.get(arm_control, np.nan))
        did = mean_delta_treated - mean_delta_control

        p_did = _perm_test_diff(
            deltas["delta"],
            deltas["arm"],
            n_perm=n_perm,
            seed=seed,
            treated_label=arm_treated,
        )

        rows.append(
            {
                "pool": pool,
                "module": module,
                "mean_delta_treated": mean_delta_treated,
                "mean_delta_control": mean_delta_control,
                "beta_DiD": float(did),
                "p_DiD": float(p_did),
                "p_treated": float(p_arm.get(arm_treated, np.nan)),
                "p_control": float(p_arm.get(arm_control, np.nan)),
                "n_units": int(deltas.index.nunique()),
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    if fdr_within is None:
        mask = res["p_DiD"].notna()
        res["FDR_DiD"] = np.nan
        if mask.sum() > 0:
            res.loc[mask, "FDR_DiD"] = multipletests(res.loc[mask, "p_DiD"], method="fdr_bh")[1]
    else:
        warnings.warn(
            f"FDR correction is applied within each '{fdr_within}' group "
            f"(column FDR_DiD). Per-group FDR does not control the overall "
            f"false discovery rate across all tests. Consult "
            f"FDR_DiD_global (when fdr_global=True) for a globally "
            f"corrected q-value.",
            stacklevel=2,
        )
        res["FDR_DiD"] = np.nan
        for key, sub in res.groupby(fdr_within, observed=True):
            mask = sub["p_DiD"].notna()
            if mask.sum() > 0:
                res.loc[sub.index[mask], "FDR_DiD"] = multipletests(
                    sub.loc[mask, "p_DiD"], method="fdr_bh"
                )[1]

        if fdr_global:
            mask = res["p_DiD"].notna()
            res["FDR_DiD_global"] = np.nan
            if mask.sum() > 0:
                res.loc[mask, "FDR_DiD_global"] = multipletests(
                    res.loc[mask, "p_DiD"], method="fdr_bh"
                )[1]

    return res


def module_score_within_arm_by_pool(
    pb: pd.DataFrame,
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    min_paired: int = 3,
    fdr_within: str | None = "module",
) -> pd.DataFrame:
    """Within-arm paired tests on module scores by pool.

    Parameters
    ----------
    pb
        Output of module_score_pseudobulk().
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    min_paired
        Minimum paired participants per pool/module.
    fdr_within
        If "module", FDR is computed within each module across pools.
        If "pool", FDR is computed within each pool across modules.
        If None, global FDR.

    Returns
    -------
    pd.DataFrame
        One row per (pool, module) with columns: ``pool``, ``module``,
        ``mean_delta``, ``p_time``, ``n_units``, ``FDR_time``.
    """
    rows: list[dict[str, Any]] = []

    for (pool, module), sub in pb.groupby(["pool", "module"], observed=True):
        wide = sub.pivot_table(
            index=design.participant_col,
            columns=design.visit_col,
            values="module_score",
            aggfunc="mean",
        )
        if visits[0] not in wide.columns or visits[1] not in wide.columns:
            continue
        wide = wide.dropna()
        if len(wide) < min_paired:
            continue

        pre = wide[visits[0]].to_numpy(dtype=float)
        post = wide[visits[1]].to_numpy(dtype=float)
        delta = post - pre
        try:
            _, p_val = wilcoxon(delta)
        except (ValueError, TypeError):
            p_val = np.nan

        rows.append(
            {
                "pool": pool,
                "module": module,
                "mean_delta": float(delta.mean()),
                "p_time": float(p_val),
                "n_units": int(len(wide)),
            }
        )

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    if fdr_within is None:
        mask = res["p_time"].notna()
        res["FDR_time"] = np.nan
        if mask.sum() > 0:
            res.loc[mask, "FDR_time"] = multipletests(res.loc[mask, "p_time"], method="fdr_bh")[1]
    else:
        res["FDR_time"] = np.nan
        for key, sub in res.groupby(fdr_within, observed=True):
            mask = sub["p_time"].notna()
            if mask.sum() > 0:
                res.loc[sub.index[mask], "FDR_time"] = multipletests(
                    sub.loc[mask, "p_time"], method="fdr_bh"
                )[1]

    return res
