from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from anndata import AnnData
from ..design import TrialDesign
from ..adata_tools import subset_primary
from ..utils import wild_cluster_bootstrap_t

def abundance_did(
    adata: AnnData,
    design: TrialDesign,
    visits: Tuple[str,str],
    exclude_crossovers: bool = True,
    transform: str = "arcsin_sqrt",
    min_units: int = 5,
    covariates: Optional[List[str]] = None,
    use_bootstrap: bool = False,
    n_boot: int = 999,
    seed: int = 42,
) -> pd.DataFrame:
    """Test treatment-induced cell-type abundance changes via DiD on proportions.

    This function calculates cell-type proportions per participant-visit and 
    fits a DiD model to test for treatment-induced compositional shifts.

    Parameters
    ----------
    adata
        AnnData object.
    design
        A `TrialDesign` object.
    visits
        Tuple of (baseline, followup) visit labels.
    exclude_crossovers
        Whether to exclude crossover cells.
    transform
        Mathematical transformation for proportions:
        - 'arcsin_sqrt': arcsin(sqrt(p)), standard for proportions.
        - 'logit': log(p / (1-p)).
        - 'none': use raw proportions.
    min_units
        Minimum number of paired participants required for a cell type to be 
        tested.
    covariates
        Additional columns in `adata.obs` to include as fixed effects.
    use_bootstrap
        If True, uses Wild Cluster Bootstrap for p-values.
    n_boot
        Number of bootstrap permutations.
    seed
        Random seed.

    Returns
    -------
    pd.DataFrame
        Table with one row per cell type containing beta_DiD and significance.

    Examples
    --------
    >>> ab_res = abundance_did(adata, design, visits=("V1", "V2"))
    >>> print(ab_res)
    """
    if design.celltype_col is None:
        raise ValueError("celltype_col is required for abundance_did")

    ad = subset_primary(adata, design, visits=visits, exclude_crossovers=exclude_crossovers)
    obs = ad.obs.copy()

    # counts per unit×visit×arm×celltype
    grp_cols = [design.participant_col, design.visit_col, design.arm_col, design.celltype_col]
    
    # We need to preserve covariates. Covariates are usually participant-level or participant-visit level.
    # If they are participant-level, they are constant for all cells of a participant.
    counts = (
        obs
        .groupby(grp_cols, observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    totals = counts.groupby([design.participant_col, design.visit_col, design.arm_col], observed=True)["n_cells"].sum().reset_index(name="total_cells")
    counts = counts.merge(totals, on=[design.participant_col, design.visit_col, design.arm_col], how="left")
    counts["prop"] = counts["n_cells"] / (counts["total_cells"] + 1e-12)

    if covariates:
        # Merge covariates back into counts. 
        # Assume covariates are constant per (participant, visit).
        cov_df = obs[[design.participant_col, design.visit_col] + covariates].drop_duplicates()
        counts = counts.merge(cov_df, on=[design.participant_col, design.visit_col], how="left")

    if transform == "arcsin_sqrt":
        y = np.arcsin(np.sqrt(counts["prop"].clip(0,1)))
        counts["y"] = y
    elif transform == "logit":
        p = counts["prop"].clip(1e-6, 1-1e-6)
        counts["y"] = np.log(p/(1-p))
    else:
        counts["y"] = counts["prop"]

    counts[design.visit_col] = pd.Categorical(counts[design.visit_col], categories=list(visits), ordered=True)
    counts["visit_num"] = counts[design.visit_col].map({visits[0]:0, visits[1]:1}).astype(float)
    counts["arm_bin"] = design.arm_bin(counts)

    rows=[]
    for ct in sorted(counts[design.celltype_col].unique()):
        tmp = counts[counts[design.celltype_col]==ct].copy()
        # keep paired units only
        wide = tmp.pivot_table(index=design.participant_col, columns=design.visit_col, aggfunc="size", fill_value=0, observed=True)
        keep = wide[(wide.get(visits[0],0)>0)&(wide.get(visits[1],0)>0)].index
        tmp = tmp[tmp[design.participant_col].isin(keep)].copy()

        n_units = tmp[design.participant_col].nunique()
        if n_units < min_units:
            continue
        # must have both arms among units
        arm_counts = tmp.groupby("arm_bin")[design.participant_col].nunique()
        if (arm_counts>0).sum() < 2:
            continue
        
        # Ensure there is at least some variation in the outcome
        if tmp["y"].nunique() < 2:
            continue

        formula = f"y ~ visit_num + visit_num:arm_bin + C({design.participant_col})"
        if covariates:
            formula += " + " + " + ".join(covariates)

        # residual df guard for FE
        model = smf.ols(formula, data=tmp)
        
        try:
            fit = model.fit()
            term = "visit_num:arm_bin"
            
            # Check if interaction term was estimable
            if term not in fit.params or np.isnan(fit.params[term]):
                continue

            p_val = float(fit.pvalues[term])
            if use_bootstrap:
                p_val = wild_cluster_bootstrap_t(
                    fit,
                    X=fit.model.exog,
                    clusters=tmp[design.participant_col].values,
                    term_name=term,
                    B=n_boot,
                    seed=seed
                )

            rows.append({
                "celltype": ct,
                "n_participants": int(n_units),
                "beta_DiD": float(fit.params[term]),
                "se_DiD": float(fit.bse[term]),
                "p_DiD": p_val,
                "beta_time": float(fit.params.get("visit_num", np.nan)),
                "p_time": float(fit.pvalues.get("visit_num", np.nan)),
            })
        except Exception:
            continue
            
    if not rows:
        return pd.DataFrame(columns=[
            "celltype", "n_participants", "beta_DiD", "se_DiD", 
            "p_DiD", "beta_time", "p_time", "FDR_DiD"
        ])

    res = pd.DataFrame(rows).sort_values("p_DiD")
    
    # FDR correction
    mask = res["p_DiD"].notna()
    res["FDR_DiD"] = np.nan
    if mask.sum() > 0:
        res.loc[mask, "FDR_DiD"] = multipletests(res.loc[mask, "p_DiD"], method="fdr_bh")[1]
        
    return res.reset_index(drop=True)
