from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from ..design import TrialDesign

__all__ = [
    "module_score_pseudobulk",
    "module_score_did_by_pool",
]


def _map_pool(celltype: str, pool_map: dict[str, Sequence[str]] | None) -> str | None:
    if pool_map is None:
        return None
    for pool, labels in pool_map.items():
        if celltype in labels:
            return pool
    return None


def module_score_pseudobulk(
    adata,
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

    # Long format: participant × visit × pool × module
    long_df = df[
        [design.participant_col, design.visit_col, design.arm_col, "pool"] + list(module_cols)
    ].melt(
        id_vars=[design.participant_col, design.visit_col, design.arm_col, "pool"],
        value_vars=list(module_cols),
        var_name="module",
        value_name="module_score",
    )

    # Pseudobulk aggregation
    pb = (
        long_df.groupby(
            [design.participant_col, design.visit_col, design.arm_col, "pool", "module"],
            observed=True,
        )["module_score"]
        .mean()
        .reset_index()
    )

    counts = (
        df.groupby([design.participant_col, design.visit_col, design.arm_col, "pool"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )

    pb = pb.merge(
        counts,
        on=[design.participant_col, design.visit_col, design.arm_col, "pool"],
        how="left",
    )
    pb = pb[pb["n_cells"] >= min_cells_per_group].copy()
    return pb


def _perm_test_diff(delta: pd.Series, arms: pd.Series, n_perm: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values = delta.values
    labels = arms.values
    treated_label = labels[0]
    obs = values[labels == treated_label].mean() - values[labels != treated_label].mean()
    perm_vals = []
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        perm_vals.append(values[perm == treated_label].mean() - values[perm != treated_label].mean())
    perm_vals = np.asarray(perm_vals)
    return float((np.abs(perm_vals) >= np.abs(obs)).mean())


def module_score_did_by_pool(
    pb: pd.DataFrame,
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    min_paired: int = 2,
    n_perm: int = 1000,
    seed: int = 42,
    fdr_within: str | None = "module",
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

        # attach arm labels
        pid_arm = sub.groupby(design.participant_col)[design.arm_col].first()
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
                try:
                    _, p = wilcoxon(sub_arm[visits[1]].values - sub_arm[visits[0]].values)
                except (ValueError, TypeError):
                    p = np.nan
            else:
                p = np.nan
            p_arm[arm_label] = p

        did = deltas[deltas["arm"] == design.arm_treated]["delta"].mean() - \
              deltas[deltas["arm"] == design.arm_control]["delta"].mean()

        p_did = _perm_test_diff(deltas["delta"], deltas["arm"], n_perm=n_perm, seed=seed)

        rows.append({
            "pool": pool,
            "module": module,
            "mean_delta_treated": float(deltas[deltas["arm"] == design.arm_treated]["delta"].mean()),
            "mean_delta_control": float(deltas[deltas["arm"] == design.arm_control]["delta"].mean()),
            "beta_DiD": float(did),
            "p_DiD": float(p_did),
            "p_treated": float(p_arm.get(design.arm_treated, np.nan)),
            "p_control": float(p_arm.get(design.arm_control, np.nan)),
            "n_units": int(deltas.index.nunique()),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    if fdr_within is None:
        mask = res["p_DiD"].notna()
        res["FDR_DiD"] = np.nan
        if mask.sum() > 0:
            res.loc[mask, "FDR_DiD"] = multipletests(res.loc[mask, "p_DiD"], method="fdr_bh")[1]
    else:
        res["FDR_DiD"] = np.nan
        for key, sub in res.groupby(fdr_within, observed=True):
            mask = sub["p_DiD"].notna()
            if mask.sum() > 0:
                res.loc[sub.index[mask], "FDR_DiD"] = multipletests(
                    sub.loc[mask, "p_DiD"], method="fdr_bh"
                )[1]

    return res
