from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

from ..adata_tools import subset_arms, subset_primary
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
    "rank_genes_cross_sectional",
    "rank_genes_did",
    "rank_genes_within_arm",
    "run_gsea_cross_sectional",
    "run_gsea_did",
    "run_gsea_did_by_celltype",
    "run_gsea_did_multi",
    "run_gsea_prerank",
    "run_gsea_pseudobulk",
    "run_gsea_within_arm",
]


def _ensure_gseapy() -> None:
    """Ensure gseapy is installed."""
    if gp is None:
        raise ImportError(
            "gseapy is required for GSEA functions. Install with 'pip install gseapy'."
        )


def _finalize_ranking(valid: pd.DataFrame) -> pd.DataFrame:
    """Return a clean feature/rank table safe for gseapy.prerank.

    ``.dropna()`` alone is not enough: a degenerate OLS (e.g. a zero-variance
    gene, or a small panel like AML's 11 participants with haemoglobin retained)
    can yield an infinite beta and hence an infinite rank, which gseapy turns
    into a "float division by zero". Duplicate gene identifiers likewise break
    prerank's running-sum normalisation. So drop non-finite ranks and collapse
    duplicate features (keeping the strongest-magnitude rank) before sorting.

    Genes with the same data across participants have the same statistic, and
    statistics that are equal in exact arithmetic can differ in the last bits from
    one run to the next. The enrichment score of a gene set depends on where its
    genes fall among such ties, which gseapy orders arbitrarily. The ranks are
    therefore rounded to ten significant digits and equal ranks are ordered by gene
    name, so the same data give the same ranking in every run.
    """
    out = valid[["feature", "rank"]].copy()
    out["rank"] = out["rank"].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["rank"])
    out["rank"] = [float(f"{value:.10g}") for value in out["rank"].astype(float)]
    out["feature"] = out["feature"].astype(str)
    if out["feature"].duplicated().any():
        strongest_first = out.assign(_size=out["rank"].abs()).sort_values(
            ["_size", "rank"], ascending=False, kind="mergesort"
        )
        out = strongest_first.drop_duplicates(subset="feature", keep="first").drop(columns="_size")
    return out.sort_values(["rank", "feature"], ascending=[False, True], kind="mergesort")


def _order_enrichment(res: pd.DataFrame) -> pd.DataFrame:
    """Rows as gseapy orders them, by decreasing absolute NES, with equal NES by term name.

    Gene sets that hold the same ranked genes have the same NES, and gseapy returns them in
    an order that changes from run to run.
    """
    if "NES" not in res.columns or "Term" not in res.columns:
        return res
    size = pd.to_numeric(res["NES"], errors="coerce").abs()
    ordered = res.assign(_size=size).sort_values(
        ["_size", "Term"], ascending=[False, True], kind="mergesort"
    )
    return ordered.drop(columns="_size").reset_index(drop=True)


class _NoRankableGenes(ValueError):
    """No gene of a results table can be ranked."""


def _rank_results(
    res: pd.DataFrame, rank_by: str, min_units: int, *, beta: str, se: str, p: str
) -> pd.DataFrame:
    """Rank the genes of a results table by ``rank_by``.

    A gene enters the ranking only with the quantities its rank is made of: the estimate,
    and its standard error (``"tstat"``) or p-value (``"signed_confidence"``). A gene whose
    model could not be fitted has no place in the ranked list; given a rank of zero it would
    sit in the middle of the list and count as a gene outside every set.
    """
    needs = {"signed_confidence": (beta, p), "beta": (beta,), "tstat": (beta, se)}
    if rank_by not in needs:
        raise ValueError(f"Unknown rank_by: {rank_by}")
    valid = res[res["n_units"] >= min_units]
    estimated = np.isfinite(valid[list(needs[rank_by])].to_numpy(dtype=float)).all(axis=1)
    valid = valid[estimated].copy()
    if len(valid) == 0:
        raise _NoRankableGenes(
            f"No genes have an estimate from at least min_units={min_units} participants. "
            f"Try reducing min_units or checking your data."
        )

    if rank_by == "signed_confidence":
        valid["rank"] = np.sign(valid[beta]) * -np.log10(valid[p] + 1e-12)
    elif rank_by == "beta":
        valid["rank"] = valid[beta]
    else:
        valid["rank"] = valid[beta] / (valid[se] + 1e-12)
    return _finalize_ranking(valid)


def _rank_did_results(res: pd.DataFrame, rank_by: str, min_units: int) -> pd.DataFrame:
    """Rank the DiD results."""
    return _rank_results(res, rank_by, min_units, beta="beta_DiD", se="se_DiD", p="p_DiD")


def _rank_between_arm_results(res: pd.DataFrame, rank_by: str, min_units: int) -> pd.DataFrame:
    """Rank the between-arm comparison results."""
    return _rank_results(res, rank_by, min_units, beta="beta_arm", se="se_arm", p="p_arm")


def _rank_within_arm_results(res: pd.DataFrame, rank_by: str, min_units: int) -> pd.DataFrame:
    """Rank the within-arm comparison results."""
    return _rank_results(res, rank_by, min_units, beta="beta_time", se="se_time", p="p_time")


def rank_genes_cross_sectional(
    adata: AnnData,
    design: TrialDesign,
    visit: str,
    layer: str | None = None,
    rank_by: str = "tstat",
    min_units: int = 4,
) -> pd.DataFrame:
    """Rank all genes by the between-arm difference at one visit.

    The gene-level table of `between_arm_comparison` is computed once. The ranking
    can then be given to `run_gsea_prerank` for any number of gene-set collections.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
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

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``rank``, by decreasing rank. Genes without an
        estimate are left out.
    """
    res = between_arm_comparison(
        adata,
        visit=visit,
        features=adata.var_names.tolist(),
        design=design,
        layer=layer,
        aggregate="participant_visit",
        standardize=True,
        method="ols",
    )
    return _rank_between_arm_results(res, rank_by=rank_by, min_units=min_units)


def rank_genes_within_arm(
    adata: AnnData,
    design: TrialDesign,
    arm: str,
    visits: tuple[str, str],
    layer: str | None = None,
    rank_by: str = "tstat",
    min_units: int = 4,
) -> pd.DataFrame:
    """Rank all genes by the change between two visits within one arm.

    The gene-level table of `within_arm_comparison` is computed once. The ranking
    can then be given to `run_gsea_prerank` for any number of gene-set collections.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
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

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``rank``, by decreasing rank. Genes without an
        estimate are left out.
    """
    res = within_arm_comparison(
        adata,
        arm=arm,
        features=adata.var_names.tolist(),
        design=design,
        visits=visits,
        layer=layer,
        aggregate="participant_visit",
        standardize=True,
    )
    return _rank_within_arm_results(res, rank_by=rank_by, min_units=min_units)


def rank_genes_did(
    adata: AnnData,
    design: TrialDesign,
    visits: tuple[str, str],
    layer: str | None = None,
    exclude_crossovers: bool = True,
    celltype: str | None = None,
    rank_by: str = "signed_confidence",
    use_bootstrap: bool = False,
    n_boot: int = 999,
    min_units: int = 4,
    cov_type: str = "cluster",
) -> pd.DataFrame:
    """Rank all genes by their Difference-in-Differences (DiD) estimate.

    The gene-level table of `did_table` is computed once. The ranking can then be
    given to `run_gsea_prerank` for any number of gene-set collections.

    Parameters
    ----------
    adata
        AnnData object containing expression data.
    design
        A `TrialDesign` object.
    visits
        A tuple of (baseline, followup) visit labels.
    layer
        Layer to extract gene expression from. Recommended to use a
        normalized layer (e.g., 'log1p_cpm').
    exclude_crossovers
        Whether to exclude crossover cells from the DiD ranking.
    celltype
        Restrict the DiD to one cell type of `design.celltype_col`.
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
        included in the ranking. Default is 4.
    cov_type
        Covariance of the gene-level standard error and p-value that enter the
        ranking (``'tstat'`` and ``'signed_confidence'``): ``'cluster'`` (default)
        or ``'nonrobust'``, as in `did_table`.

    Returns
    -------
    pd.DataFrame
        Columns ``feature`` and ``rank``, by decreasing rank. Genes without an
        estimate are left out.

    Examples
    --------
    >>> ranking = rank_genes_did(adata, design, visits=("V1", "V2"), rank_by="tstat")
    >>> kegg = run_gsea_prerank(ranking, "KEGG_2021_Human")
    >>> hallmark = run_gsea_prerank(ranking, "MSigDB_Hallmark_2020")
    """
    res = did_table(
        adata,
        features=adata.var_names.tolist(),
        design=design,
        visits=visits,
        celltype=celltype,
        exclude_crossovers=exclude_crossovers,
        layer=layer,
        aggregate="participant_visit",
        use_bootstrap=use_bootstrap,
        n_boot=n_boot,
        cov_type=cov_type,
    )
    return _rank_did_results(res, rank_by=rank_by, min_units=min_units)


def run_gsea_prerank(
    ranking: pd.DataFrame,
    gene_sets: str | dict[str, list[str]],
    return_obj: bool = False,
    **kwargs,
) -> pd.DataFrame | gp.Prerank:
    """Run `gseapy.prerank` on a gene ranking.

    Ranking the genes is the expensive step of a trial-aware GSEA, because it fits
    one model per gene. A ranking from `rank_genes_did`, `rank_genes_within_arm` or
    `rank_genes_cross_sectional` is computed once and tested here against one
    gene-set collection per call. Each collection is its own analysis, with its own
    permutations and its own FDR.

    Parameters
    ----------
    ranking
        Table with the columns ``feature`` (gene) and ``rank`` (ranking metric).
        Non-finite ranks are dropped, a repeated gene keeps its strongest rank and
        equal ranks are ordered by gene name; the table itself is not modified.
    gene_sets
        A library name (e.g. 'KEGG_2021_Human') or a dictionary mapping
        pathway names to gene lists.
    return_obj
        Whether to return the full gseapy object. If False (default),
        returns the results DataFrame (`res2d`).
    **kwargs
        Additional parameters passed to `gseapy.prerank` (e.g., `permutation_num`,
        `outdir`, `min_size`, `max_size`).

    Returns
    -------
    pd.DataFrame or gseapy.Prerank
        Enrichment results or the gseapy result object. The results table is ordered by
        decreasing absolute NES, gene sets with the same NES by name.
    """
    _ensure_gseapy()
    missing = {"feature", "rank"} - set(ranking.columns)
    if missing:
        raise ValueError(
            "ranking needs the columns 'feature' and 'rank'; missing: "
            + ", ".join(sorted(missing))
        )
    pre_res = gp.prerank(rnk=_finalize_ranking(ranking), gene_sets=gene_sets, **kwargs)

    # Handle both gseapy >= 1.0 (has .res2d) and potentially older versions
    if not hasattr(pre_res, "res2d"):
        return pre_res
    pre_res.res2d = _order_enrichment(pre_res.res2d)
    return pre_res if return_obj else pre_res.res2d


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
    difference at a fixed visit, then runs gseapy.prerank. For several gene-set
    collections, rank once with `rank_genes_cross_sectional` and call
    `run_gsea_prerank` per collection.

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
    ranking = rank_genes_cross_sectional(
        adata, design, visit, layer=layer, rank_by=rank_by, min_units=min_units
    )
    return run_gsea_prerank(ranking, gene_sets, return_obj=return_obj, **kwargs)


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
    For several gene-set collections, rank once with `rank_genes_within_arm` and
    call `run_gsea_prerank` per collection.

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
    ranking = rank_genes_within_arm(
        adata, design, arm, visits, layer=layer, rank_by=rank_by, min_units=min_units
    )
    return run_gsea_prerank(ranking, gene_sets, return_obj=return_obj, **kwargs)


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
    cov_type: str = "cluster",
    **kwargs,
) -> pd.DataFrame | gp.Prerank:
    """Perform Gene Set Enrichment Analysis (GSEA) on trial-aware rankings.

    This function calculates Difference-in-Differences (DiD) effect sizes for
    all genes, ranks them, and performs GSEA using `gseapy.prerank`. This approach
    ensures that enriched pathways represent treatment effects rather than
    baseline differences. For several gene-set collections, use
    `run_gsea_did_multi`, or rank once with `rank_genes_did` and call
    `run_gsea_prerank` per collection.

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
    celltype
        Restrict the DiD to one cell type of `design.celltype_col`.
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
    cov_type
        Covariance of the gene-level standard error and p-value that enter the
        ranking (``'tstat'`` and ``'signed_confidence'``): ``'cluster'`` (default)
        or ``'nonrobust'``, as in `did_table`.
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
    ranking = rank_genes_did(
        adata,
        design,
        visits,
        layer=layer,
        exclude_crossovers=exclude_crossovers,
        celltype=celltype,
        rank_by=rank_by,
        use_bootstrap=use_bootstrap,
        n_boot=n_boot,
        min_units=min_units,
        cov_type=cov_type,
    )
    return run_gsea_prerank(ranking, gene_sets, return_obj=return_obj, **kwargs)


def _two_arm_rows(adata: AnnData, design: TrialDesign, visits: tuple[str, str]) -> AnnData:
    """The rows of the two visits and the two named arms, for wrappers of the DiD ranking:
    excluded rows are reported once, to the caller of the wrapper."""
    in_visits = adata.obs[design.visit_col].isin(list(visits)).to_numpy()
    return subset_arms(adata[in_visits], design, stacklevel=4)


def run_gsea_did_multi(
    adata: AnnData,
    gene_sets: dict[str, str | dict[str, list[str]]],
    design: TrialDesign,
    visits: tuple[str, str],
    *,
    layer: str | None = None,
    exclude_crossovers: bool = True,
    celltype: str | None = None,
    rank_by: str = "signed_confidence",
    use_bootstrap: bool = False,
    n_boot: int = 999,
    min_units: int = 4,
    return_obj: bool = False,
    cov_type: str = "cluster",
    **kwargs,
) -> dict[str, pd.DataFrame | gp.Prerank]:
    """Run GSEA across multiple gene-set collections.

    The genes are ranked once and the same ranking is tested against every
    collection, each in its own `gseapy.prerank` run.

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
    layer, exclude_crossovers, celltype, rank_by, use_bootstrap, n_boot, min_units, cov_type
        The gene ranking, as in `run_gsea_did`.
    return_obj
        Whether to return the full gseapy objects instead of the results tables.
    **kwargs
        Additional parameters passed to `gseapy.prerank` (e.g., `permutation_num`,
        `outdir`, `min_size`, `max_size`).

    Returns
    -------
    dict[str, pd.DataFrame | gp.Prerank]
        Dictionary of GSEA results.
    """
    _ensure_gseapy()
    adata = _two_arm_rows(adata, design, visits)
    ranking = rank_genes_did(
        adata,
        design,
        visits,
        layer=layer,
        exclude_crossovers=exclude_crossovers,
        celltype=celltype,
        rank_by=rank_by,
        use_bootstrap=use_bootstrap,
        n_boot=n_boot,
        min_units=min_units,
        cov_type=cov_type,
    )
    return {
        label: run_gsea_prerank(ranking, gs, return_obj=return_obj, **kwargs)
        for label, gs in gene_sets.items()
    }


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
        List of cell types to analyze. If None, uses the cell types of
        `design.celltype_col` among the cells of the two arms and the two visits.
    **kwargs
        Additional parameters passed to `run_gsea_did`.

    Returns
    -------
    dict[str, pd.DataFrame | gp.Prerank]
        GSEA results keyed by cell type. A cell type in which the DiD cannot be
        estimated (an arm without a participant sampled at both visits, or fewer
        than `min_units` participants) is left out, with one warning.
    """
    _ensure_gseapy()
    if design.celltype_col is None:
        raise ValueError("design.celltype_col must be set for celltype GSEA.")

    adata = _two_arm_rows(adata, design, visits)
    if celltypes is None:
        celltypes = sorted(adata.obs[design.celltype_col].dropna().unique())

    # The DiD of a cell type needs, in each arm, a participant with cells at both visits,
    # counted on the rows the DiD will use.
    obs = subset_primary(
        adata, design, visits=visits, exclude_crossovers=kwargs.get("exclude_crossovers", True)
    ).obs
    visits_seen = obs.groupby(
        [design.celltype_col, design.arm_col, design.participant_col], observed=True
    )[design.visit_col].nunique()
    paired = visits_seen[visits_seen >= 2].reset_index()
    arms_seen = paired.groupby(design.celltype_col, observed=True)[design.arm_col].nunique()
    estimable = set(arms_seen.index[arms_seen >= 2])

    results: dict[str, pd.DataFrame | gp.Prerank] = {}
    skipped: list[str] = []
    for ct in celltypes:
        if ct not in estimable:
            skipped.append(ct)
            continue
        try:
            results[ct] = run_gsea_did(
                adata,
                gene_sets=gene_sets,
                design=design,
                visits=visits,
                celltype=ct,
                **kwargs,
            )
        except _NoRankableGenes:
            skipped.append(ct)
    if skipped:
        warnings.warn(
            f"{len(skipped)} cell types were skipped because the DiD cannot be estimated in "
            "them (each arm needs participants with cells at both visits, at least "
            "`min_units` in total): " + ", ".join(str(ct) for ct in skipped) + ".",
            UserWarning,
            stacklevel=2,
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
            except _NoRankableGenes:
                continue
            ct_pre = run_gsea_prerank(ct_ranking, gene_sets, return_obj=return_obj, **kwargs)
            if return_obj:
                obj_results[ct] = ct_pre
            else:
                ct_df = ct_pre if isinstance(ct_pre, pd.DataFrame) else pd.DataFrame(ct_pre)
                ct_df["celltype"] = ct
                all_results.append(ct_df)
        if return_obj:
            return obj_results
        if not all_results:
            return pd.DataFrame()
        return pd.concat(all_results, ignore_index=True)

    ranking = _rank_did_results(res, rank_by=rank_by, min_units=min_units)
    return run_gsea_prerank(ranking, gene_sets, return_obj=return_obj, **kwargs)
