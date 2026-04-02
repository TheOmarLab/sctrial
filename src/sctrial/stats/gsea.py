from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

from ..design import TrialDesign
from .comparisons import between_arm_comparison, within_arm_comparison
from .did import did_table
from .pseudobulk import pseudobulk_did

if TYPE_CHECKING:
    import gseapy as gp

try:
    import gseapy as gp
except ImportError:
    gp = None

__all__ = [
    "run_gsea_cross_sectional",
    "run_gsea_did",
    "run_gsea_did_by_celltype",
    "run_gsea_did_multi",
    "run_gsea_pseudobulk",
    "run_gsea_within_arm",
]


def _ensure_gseapy() -> None:
    """Ensure gseapy is installed."""
    if gp is None:
        raise ImportError(
            "gseapy is required for GSEA functions. Install with 'pip install gseapy'."
        )


def _rank_did_results(
    res: pd.DataFrame,
    rank_by: str,
    min_units: int,
) -> pd.DataFrame:
    """Rank the DiD results."""
    valid = res[res["n_units"] >= min_units].copy()
    if len(valid) == 0:
        raise ValueError(
            f"No genes have sufficient data (min_units={min_units}). "
            f"Try reducing min_units or checking your data."
        )

    if rank_by == "signed_confidence":
        valid["rank"] = np.sign(valid["beta_DiD"].fillna(0)) * -np.log10(
            valid["p_DiD"].fillna(1) + 1e-12
        )
    elif rank_by == "beta":
        valid["rank"] = valid["beta_DiD"].fillna(0)
    elif rank_by == "tstat":
        valid["rank"] = valid["beta_DiD"].fillna(0) / (valid["se_DiD"].fillna(1) + 1e-12)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")

    return valid[["feature", "rank"]].dropna().sort_values("rank", ascending=False)


def _rank_between_arm_results(
    res: pd.DataFrame,
    rank_by: str,
    min_units: int,
) -> pd.DataFrame:
    """Rank the between-arm comparison results."""
    valid = res[res["n_units"] >= min_units].copy()
    if len(valid) == 0:
        raise ValueError(
            f"No genes have sufficient data (min_units={min_units}). "
            f"Try reducing min_units or checking your data."
        )

    if rank_by == "signed_confidence":
        valid["rank"] = np.sign(valid["beta_arm"].fillna(0)) * -np.log10(
            valid["p_arm"].fillna(1) + 1e-12
        )
    elif rank_by == "beta":
        valid["rank"] = valid["beta_arm"].fillna(0)
    elif rank_by == "tstat":
        valid["rank"] = valid["beta_arm"].fillna(0) / (valid["se_arm"].fillna(1) + 1e-12)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")

    return valid[["feature", "rank"]].dropna().sort_values("rank", ascending=False)


def _rank_within_arm_results(
    res: pd.DataFrame,
    rank_by: str,
    min_units: int,
) -> pd.DataFrame:
    """Rank the within-arm comparison results."""
    valid = res[res["n_units"] >= min_units].copy()
    if len(valid) == 0:
        raise ValueError(
            f"No genes have sufficient data (min_units={min_units}). "
            f"Try reducing min_units or checking your data."
        )

    if rank_by == "signed_confidence":
        valid["rank"] = np.sign(valid["beta_time"].fillna(0)) * -np.log10(
            valid["p_time"].fillna(1) + 1e-12
        )
    elif rank_by == "beta":
        valid["rank"] = valid["beta_time"].fillna(0)
    elif rank_by == "tstat":
        valid["rank"] = valid["beta_time"].fillna(0) / (valid["se_time"].fillna(1) + 1e-12)
    else:
        raise ValueError(f"Unknown rank_by: {rank_by}")

    return valid[["feature", "rank"]].dropna().sort_values("rank", ascending=False)


def run_gsea_cross_sectional(
    adata: AnnData,
    gene_sets: str | dict[str, list[str]],
    design: TrialDesign,
    visit: str,
    layer: str | None = None,
    rank_by: str = "tstat",
    min_units: int = 4,
    return_obj: bool = False,
    **kwargs,
) -> pd.DataFrame | gp.Prerank:
    """Perform GSEA on cross-sectional data using between-arm comparison.

    Similar to run_gsea_did but for cross-sectional designs (single visit).
    Uses between_arm_comparison to rank genes by treatment vs control
    difference at a fixed visit, then runs gseapy.prerank.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    gene_sets
        A library name (e.g. 'KEGG_2021_Human') or a dictionary mapping
        pathway names to gene lists.
    design
        A `TrialDesign` object.
    visit
        The visit label to analyze (single timepoint).
    layer
        Layer to extract gene expression from.
    rank_by
        Metric for ranking genes: 'signed_confidence', 'beta', or 'tstat'.
    min_units
        Minimum number of participants required per gene.
    return_obj
        Whether to return the full gseapy object.
    **kwargs
        Additional parameters passed to gseapy.prerank.

    Returns
    -------
    pd.DataFrame or gseapy.Prerank
        Enrichment results or the gseapy result object.
    """
    _ensure_gseapy()

    genes = adata.var_names.tolist()
    res = between_arm_comparison(
        adata,
        visit=visit,
        features=genes,
        design=design,
        layer=layer,
        aggregate="participant_visit",
        standardize=True,
        method="ols",
    )

    ranking = _rank_between_arm_results(res, rank_by=rank_by, min_units=min_units)
    pre_res = gp.prerank(rnk=ranking, gene_sets=gene_sets, **kwargs)

    if return_obj:
        return pre_res
    if hasattr(pre_res, "res2d"):
        return pre_res.res2d
    return pre_res


def run_gsea_within_arm(
    adata: AnnData,
    gene_sets: str | dict[str, list[str]],
    design: TrialDesign,
    arm: str,
    visits: tuple[str, str],
    layer: str | None = None,
    rank_by: str = "tstat",
    min_units: int = 4,
    return_obj: bool = False,
    **kwargs,
) -> pd.DataFrame | gp.Prerank:
    """Perform GSEA on longitudinal data using within-arm comparison.

    Similar to run_gsea_did but uses within_arm_comparison instead of DiD.
    Ranks genes by pre→post change within a single arm, then runs gseapy.prerank.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    gene_sets
        A library name (e.g. 'KEGG_2021_Human') or a dictionary mapping
        pathway names to gene lists.
    design
        A `TrialDesign` object.
    arm
        The arm to analyze (e.g., design.arm_treated).
    visits
        Tuple of (baseline, followup) visit labels.
    layer
        Layer to extract gene expression from.
    rank_by
        Metric for ranking genes: 'signed_confidence', 'beta', or 'tstat'.
    min_units
        Minimum number of participants required per gene.
    return_obj
        Whether to return the full gseapy object.
    **kwargs
        Additional parameters passed to gseapy.prerank.

    Returns
    -------
    pd.DataFrame or gseapy.Prerank
        Enrichment results or the gseapy result object.
    """
    _ensure_gseapy()

    genes = adata.var_names.tolist()
    res = within_arm_comparison(
        adata,
        arm=arm,
        features=genes,
        design=design,
        visits=visits,
        layer=layer,
        aggregate="participant_visit",
        standardize=True,
    )

    ranking = _rank_within_arm_results(res, rank_by=rank_by, min_units=min_units)
    pre_res = gp.prerank(rnk=ranking, gene_sets=gene_sets, **kwargs)

    if return_obj:
        return pre_res
    if hasattr(pre_res, "res2d"):
        return pre_res.res2d
    return pre_res


def run_gsea_did(
    adata: AnnData,
    gene_sets: str | dict[str, list[str]],
    design: TrialDesign,
    visits: tuple[str, str],
    layer: str | None = None,
    exclude_crossovers: bool = True,
    celltype: str | None = None,
    rank_by: str = "signed_confidence",
    use_bootstrap: bool = False,
    n_boot: int = 999,
    min_units: int = 4,
    return_obj: bool = False,
    **kwargs,
) -> pd.DataFrame | gp.Prerank:
    """Perform Gene Set Enrichment Analysis (GSEA) on trial-aware rankings.

    This function calculates Difference-in-Differences (DiD) effect sizes for
    all genes, ranks them, and performs GSEA using `gseapy.prerank`. This approach
    ensures that enriched pathways represent treatment effects rather than
    baseline differences.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    gene_sets
        A library name (e.g. 'KEGG_2021_Human') or a dictionary mapping
        pathway names to gene lists.
    design
        A `TrialDesign` object.
    visits
        A tuple of (baseline, followup) visit labels.
    layer
        Layer to extract gene expression from. Recommended to use a
        normalized layer (e.g., 'log1p_cpm').
    exclude_crossovers
        Whether to exclude crossover cells from the DiD ranking.
    rank_by
        Metric for ranking genes:

        - 'signed_confidence': sign(beta_DiD) * -log10(p_DiD). Highlights
          genes with high effect and high significance.
        - 'beta': ranks genes solely by the DiD effect size.
        - 'tstat': ranks genes by the t-statistic (beta_DiD / se_DiD).
    use_bootstrap
        Whether to use Wild Cluster Bootstrap for DiD p-values (used if
        rank_by is 'signed_confidence').
    n_boot
        Number of bootstrap permutations.
    min_units
        Minimum number of paired participants required for a gene to be
        included in the ranking. Genes with fewer participants return NaN
        and are filtered out before GSEA. Default is 4.
    return_obj
        Whether to return the full gseapy object. If False (default),
        returns the results DataFrame (`res2d`).
    **kwargs
        Additional parameters passed to `gseapy.prerank` (e.g., `permutation_num`,
        `outdir`, `min_size`, `max_size`).

    Returns
    -------
    pd.DataFrame or gseapy.Prerank
        A DataFrame of enrichment results (if return_obj=False) or the
        gseapy result object.

    Examples
    --------
    >>> res = run_gsea_did(adata, gene_sets="KEGG_2021_Human", design=design, visits=("V1", "V2"))
    >>> print(res.head())
    """
    _ensure_gseapy()

    # 1. Run DiD for all genes
    genes = adata.var_names.tolist()
    res = did_table(
        adata,
        features=genes,
        design=design,
        visits=visits,
        celltype=celltype,
        exclude_crossovers=exclude_crossovers,
        layer=layer,
        aggregate="participant_visit",
        use_bootstrap=use_bootstrap,
        n_boot=n_boot,
    )

    # 2. Filter genes with insufficient data
    # Genes with n_units < min_units will have NaN beta_DiD
    ranking = _rank_did_results(res, rank_by=rank_by, min_units=min_units)

    # 3. Run GSEA Prerank
    pre_res = gp.prerank(rnk=ranking, gene_sets=gene_sets, **kwargs)

    if return_obj:
        return pre_res

    # Handle both gseapy >= 1.0 (has .res2d) and potentially older versions
    if hasattr(pre_res, "res2d"):
        return pre_res.res2d
    return pre_res


def run_gsea_did_multi(
    adata: AnnData,
    gene_sets: dict[str, str | dict[str, list[str]]],
    design: TrialDesign,
    visits: tuple[str, str],
    **kwargs,
) -> dict[str, pd.DataFrame | gp.Prerank]:
    """Run GSEA across multiple gene-set collections.

    Parameters
    ----------
    adata
        AnnData object.
    gene_sets
        Dictionary of gene-set collections.
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    **kwargs
        Additional parameters passed to `gseapy.prerank` (e.g., `permutation_num`,
        `outdir`, `min_size`, `max_size`).

    Returns
    -------
    dict[str, pd.DataFrame | gp.Prerank]
        Dictionary of GSEA results.
    """
    _ensure_gseapy()
    results: dict[str, pd.DataFrame | gp.Prerank] = {}
    for label, gs in gene_sets.items():
        results[label] = run_gsea_did(
            adata,
            gene_sets=gs,
            design=design,
            visits=visits,
            **kwargs,
        )
    return results


def run_gsea_did_by_celltype(
    adata: AnnData,
    gene_sets: str | dict[str, list[str]],
    design: TrialDesign,
    visits: tuple[str, str],
    celltypes: list[str] | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame | gp.Prerank]:
    """Run GSEA on DiD rankings separately for each cell type.

    Parameters
    ----------
    adata
        AnnData object.
    gene_sets
        Dictionary of gene-set collections.
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    celltypes
        List of cell types to analyze. If None, uses all unique cell types in `design.celltype_col`.
    **kwargs
        Additional parameters passed to `gseapy.prerank`

    Returns
    -------
    dict[str, pd.DataFrame | gp.Prerank]
        Dictionary of GSEA results.
    """
    _ensure_gseapy()
    if design.celltype_col is None:
        raise ValueError("design.celltype_col must be set for celltype GSEA.")
    if celltypes is None:
        celltypes = sorted(adata.obs[design.celltype_col].dropna().unique())

    results: dict[str, pd.DataFrame | gp.Prerank] = {}
    for ct in celltypes:
        results[ct] = run_gsea_did(
            adata,
            gene_sets=gene_sets,
            design=design,
            visits=visits,
            celltype=ct,
            **kwargs,
        )
    return results


def run_gsea_pseudobulk(
    adata: AnnData,
    gene_sets: str | dict[str, list[str]],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    celltype_col: str | None = None,
    rank_by: str = "signed_confidence",
    min_units: int = 4,
    return_obj: bool = False,
    **kwargs,
) -> pd.DataFrame | gp.Prerank | dict[str, gp.Prerank]:
    """Run GSEA using pseudobulk DiD results.

    Parameters
    ----------
    adata
        AnnData object.
    gene_sets
        Dictionary of gene-set collections.
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    celltype_col
        If provided and results contain per-celltype rows, run GSEA separately for each cell type and concatenate results.
    rank_by
        Metric for ranking genes. One of ``'signed_confidence'``
        (sign(beta_DiD) * -log10(p_DiD), default), ``'beta'`` (DiD effect
        size), or ``'tstat'`` (t-statistic beta_DiD / se_DiD).
    min_units
        Minimum number of paired participants required for a gene to be included in the ranking.
    return_obj
        Whether to return the full gseapy object. If False (default),
        returns the results DataFrame (`res2d`).
    **kwargs
        Additional parameters passed to `gseapy.prerank` (e.g., `permutation_num`,
        `outdir`, `min_size`, `max_size`).

    Returns
    -------
    pd.DataFrame | gp.Prerank | dict[str, gp.Prerank]
        A DataFrame of enrichment results (if return_obj=False) or the
        gseapy result object.
    """
    _ensure_gseapy()

    res = pseudobulk_did(
        adata,
        genes=adata.var_names.tolist(),
        design=design,
        visits=visits,
        celltype_col=celltype_col,
    )

    # If celltype_col is provided and results contain per-celltype rows,
    # run GSEA separately for each cell type and concatenate results.
    if celltype_col is not None and "celltype" in res.columns:
        all_results: list[pd.DataFrame] = []
        obj_results: dict[str, gp.Prerank] = {}
        for ct, ct_res in res.groupby("celltype"):
            try:
                ct_ranking = _rank_did_results(ct_res, rank_by=rank_by, min_units=min_units)
            except ValueError:
                continue
            ct_pre = gp.prerank(rnk=ct_ranking, gene_sets=gene_sets, **kwargs)
            if return_obj:
                obj_results[ct] = ct_pre
            else:
                ct_df = ct_pre.res2d if hasattr(ct_pre, "res2d") else pd.DataFrame(ct_pre)
                ct_df["celltype"] = ct
                all_results.append(ct_df)
        if return_obj:
            return obj_results
        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, ignore_index=True)

    ranking = _rank_did_results(res, rank_by=rank_by, min_units=min_units)

    pre_res = gp.prerank(
        rnk=ranking,
        gene_sets=gene_sets,
        **kwargs,
    )

    if return_obj:
        return pre_res
    if hasattr(pre_res, "res2d"):
        return pre_res.res2d
    return pre_res
