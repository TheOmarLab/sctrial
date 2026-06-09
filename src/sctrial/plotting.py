"""Publication-quality plotting: forest plots, interaction plots, volcanoes, UMAPs, GSEA heatmaps."""

from __future__ import annotationsp

from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ._env import ensure_matplotlib_config_dir, ensure_numba_cache_dir
from .design import TrialDesign
from .stats._extract import extract_gene_vector

__all__ = [
    "did_volcano_frame",
    "signed_logp",
    "plot_trial_interaction",
    "plot_parallel_trends",
    "plot_did_forest",
    "plot_did_forest_interactive",
    "plot_did_volcano_interactive",
    "plot_within_arm_comparison",
    "plot_trial_umap",
    "plot_gsea_radar",
    "plot_trial_dotplot",
    "plot_abundance_interaction",
    "plot_trial_umap_panel",
    "plot_module_umap_panel",
    "plot_gsea_heatmap",
]

# Optional dependencies for plotting
# Ensure caches/config go to writable temp dirs to avoid numba/matplotlib failures.
ensure_matplotlib_config_dir()
ensure_numba_cache_dir()

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec as GridSpecType

# Initialize optional plotting dependencies to avoid type errors
plt: ModuleType | None = None
GridSpec: type[GridSpecType] | None = None
sns: ModuleType | None = None
sc: ModuleType | None = None
_scanpy_import_error: Exception | None = None

try:
    import matplotlib.pyplot as plt  # type: ignore[no-redef]
    from matplotlib.gridspec import GridSpec  # type: ignore[no-redef]
except ImportError:
    pass

try:
    import seaborn as sns  # type: ignore[no-redef]
except ImportError:
    pass

try:
    import scanpy as sc  # type: ignore[no-redef]
except (
    ImportError,
    RuntimeError,
    OSError,
) as e:  # ImportError or runtime errors (e.g., numba cache)
    _scanpy_import_error = e


def did_volcano_frame(
    df: pd.DataFrame,
    *,
    effect_col: str = "beta_DiD",
    p_col: str = "p_DiD",
    out_col: str = "neglog10p",
    p_floor: float = 1e-300,
) -> pd.DataFrame:
    """Return a copy with an added -log10(p) column for volcano plots.

    Parameters
    ----------
    df
        Input results DataFrame.
    effect_col
        Column name for effect sizes.
    p_col
        Column name for p-values.
    out_col
        Name of the output column to add.
    p_floor
        Minimum p-value used for log transform.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``out_col`` column.
    """
    if effect_col not in df.columns:
        raise KeyError(f"Missing effect_col='{effect_col}' in df.columns.")
    if p_col not in df.columns:
        raise KeyError(f"Missing p_col='{p_col}' in df.columns.")

    out = df.copy()
    p = pd.to_numeric(out[p_col], errors="coerce").astype(float)
    out[out_col] = -np.log10(p.clip(lower=p_floor))
    return out


def signed_logp(
    df: pd.DataFrame,
    *,
    effect_col: str = "beta_DiD",
    p_col: str = "p_DiD",
    p_floor: float = 1e-300,
) -> pd.Series:
    """Return sign(effect) * -log10(p) as a Series aligned to df.index.

    Parameters
    ----------
    df
        Input results DataFrame.
    effect_col
        Column name for effect sizes.
    p_col
        Column name for p-values.
    p_floor
        Minimum p-value used for log transform.

    Returns
    -------
    pd.Series
        Signed -log10(p) values aligned to ``df.index``.
    """
    if effect_col not in df.columns:
        raise KeyError(f"Missing effect_col='{effect_col}' in df.columns.")
    if p_col not in df.columns:
        raise KeyError(f"Missing p_col='{p_col}' in df.columns.")

    eff = pd.to_numeric(df[effect_col], errors="coerce").astype(float)
    p = pd.to_numeric(df[p_col], errors="coerce").astype(float)
    return np.sign(eff) * (-np.log10(p.clip(lower=p_floor)))


def plot_trial_interaction(
    adata: AnnData,
    feature: str,
    design: TrialDesign,
    visits: tuple[str, str] | None = None,
    layer: str | None = None,
    color_palette: dict | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot mean expression (interaction plot) by arm and visit.

    This visualizes the DiD effect: the change from baseline to follow-up
    across treatment arms.

    Parameters
    ----------
    adata
        AnnData object.
    feature
        Gene name or obs column to plot.
    design
        TrialDesign object.
    visits
        Tuple of (baseline, followup) visit labels.
    layer
        Layer for gene expression.
    color_palette
        Optional colour mapping for arms.
    ax
        Optional matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the interaction plot.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    if visits is None:
        visits = design.primary_visits()

    # extract data
    id_cols = [design.participant_col, design.visit_col]
    if design.arm_col is not None:
        id_cols.append(design.arm_col)
    obs = adata.obs[id_cols].copy()
    if feature in adata.obs.columns:
        obs[feature] = adata.obs[feature].values
    elif feature in adata.var_names:
        obs[feature] = extract_gene_vector(adata, feature, layer=layer)
    else:
        raise KeyError(f"Feature '{feature}' not found.")

    # subset to relevant visits
    obs = obs[obs[design.visit_col].isin(list(visits))].copy()

    # Aggregate to participant-level means to avoid pseudoreplication
    # (Zimmerman et al., 2021; Squair et al., 2021)
    grp_cols = [design.participant_col, design.visit_col]
    if design.arm_col is not None:
        grp_cols.append(design.arm_col)
    obs = obs.groupby(grp_cols, observed=True)[feature].mean().reset_index()
    obs[design.visit_col] = pd.Categorical(
        obs[design.visit_col], categories=list(visits), ordered=True
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.pointplot(
        data=obs,
        x=design.visit_col,
        y=feature,
        hue=design.arm_col,  # None → no hue coloring
        palette=color_palette,
        dodge=True,
        capsize=0.1,
        ax=ax,
    )
    ax.set_title(f"Trial interaction: {feature}")
    return ax


def plot_parallel_trends(
    adata: AnnData,
    feature: str,
    design: TrialDesign,
    visits: Sequence[str],
    *,
    layer: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot pre-treatment trends by arm to visually assess parallel trends.

    Produces a point-plot of mean feature values across visits, separately
    for each arm, to help evaluate the parallel-trends assumption.

    Parameters
    ----------
    adata
        AnnData object.
    feature
        Feature to plot (gene name in ``var_names`` or column in ``obs``).
    design
        TrialDesign object specifying arm and visit columns.
    visits
        Sequence of visit labels to include (should be pre-treatment visits).
    layer
        Layer for gene expression (``None`` uses ``adata.X``).
    ax
        Optional matplotlib axes; a new figure is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the parallel trends plot.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )

    id_cols = [design.participant_col, design.visit_col]
    if design.arm_col is not None:
        id_cols.append(design.arm_col)
    obs = adata.obs[id_cols].copy()
    if feature in adata.obs.columns:
        obs[feature] = adata.obs[feature].values
    elif feature in adata.var_names:
        obs[feature] = extract_gene_vector(adata, feature, layer=layer)
    else:
        raise KeyError(f"Feature '{feature}' not found.")

    obs = obs[obs[design.visit_col].isin(list(visits))].copy()

    # Aggregate to participant-level means to avoid pseudoreplication
    grp_cols = [design.participant_col, design.visit_col]
    if design.arm_col is not None:
        grp_cols.append(design.arm_col)
    obs = obs.groupby(grp_cols, observed=True)[feature].mean().reset_index()
    obs[design.visit_col] = pd.Categorical(
        obs[design.visit_col], categories=list(visits), ordered=True
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.pointplot(
        data=obs,
        x=design.visit_col,
        y=feature,
        hue=design.arm_col,
        dodge=True,
        capsize=0.1,
        ax=ax,
    )
    ax.set_title(f"Parallel trends (pre-treatment): {feature}")
    return ax


def plot_did_forest(
    df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    beta_col: str = "beta_DiD",
    se_col: str = "se_DiD",
    p_col: str = "p_DiD",
    alpha: float = 0.05,
    title: str = "DiD Effect Sizes",
    ax: Axes | None = None,
) -> Axes:
    """Plot a forest plot of DiD effect sizes with confidence intervals.

    Parameters
    ----------
    df
        DataFrame returned by did_table or abundance_did.
    feature_col
        Column name for features/cell types.
    beta_col
        Column name for effect sizes.
    se_col
        Column name for standard errors.
    p_col
        Column name for p-values (to indicate significance).
    alpha
        Significance threshold for highlighting.
    title
        Plot title.
    ax
        Optional matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the forest plot.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    if df.empty:
        # Gracefully handle empty dataframe
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No data to plot", ha="center")
        ax.set_title(title)
        return ax

    # Ensure necessary columns exist
    for col in [feature_col, beta_col, se_col]:
        if col not in df.columns:
            # Fallback for abundance_did which uses 'celltype'
            if col == feature_col and "celltype" in df.columns:
                feature_col = "celltype"
            else:
                raise KeyError(f"Missing column '{col}' in DataFrame.")

    df_plot = df.copy()
    # Filter rows with NaNs in beta or se
    df_plot = df_plot.dropna(subset=[beta_col, se_col])
    if df_plot.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No valid DiD estimates", ha="center")
        ax.set_title(title)
        return ax

    from scipy import stats as sp_stats

    if "n_units" in df_plot.columns:
        df_vals = (df_plot["n_units"] - 2).clip(lower=1)
        t_crit = df_vals.apply(lambda d: sp_stats.t.ppf(0.975, d))
        df_plot["ci"] = t_crit * df_plot[se_col]
    else:
        df_plot["ci"] = 1.96 * df_plot[se_col]
    df_plot = df_plot.sort_values(beta_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 0.5 * len(df_plot) + 1))

    # Plot zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Plot points and CIs
    y_pos = np.arange(len(df_plot))

    # Identify significant ones
    sig: pd.Series
    if p_col in df_plot.columns:
        sig = df_plot[p_col] < alpha
    else:
        sig = pd.Series(np.zeros(len(df_plot), dtype=bool), index=df_plot.index)

    # Standard points
    ax.errorbar(
        df_plot.loc[~sig, beta_col],
        y_pos[~sig],
        xerr=df_plot.loc[~sig, "ci"],
        fmt="o",
        color="gray",
        label=f"p >= {alpha}" if sig.any() else None,
        capsize=3,
    )

    # Significant points
    if sig.any():
        ax.errorbar(
            df_plot.loc[sig, beta_col],
            y_pos[sig],
            xerr=df_plot.loc[sig, "ci"],
            fmt="o",
            color="firebrick",
            label=f"p < {alpha}",
            capsize=3,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot[feature_col])
    ax.set_xlabel("Effect Size (beta_DiD)")
    ax.set_title(title)
    if sig.any():
        ax.legend()

    sns.despine(ax=ax)
    return ax


def plot_within_arm_comparison(
    adata: AnnData,
    arm: str,
    feature: str,
    design: TrialDesign,
    visits: tuple[str, str],
    layer: str | None = None,
    plot_type: Literal["box", "paired"] = "paired",
    ax: Axes | None = None,
) -> Axes:
    """Plot within-arm longitudinal change.

    Parameters
    ----------
    adata
        AnnData object.
    arm
        The arm to plot (e.g., design.arm_treated).
    feature
        Gene name or module score.
    design
        A `TrialDesign` object.
    visits
        Tuple of (pre, post) visit labels.
    layer
        Expression layer.
    plot_type
        - 'box': standard boxplot with points.
        - 'paired': lines connecting pre/post values for each participant.
    ax
        Optional matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the within-arm comparison plot.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    from .adata_tools import subset_cells

    ad = subset_cells(adata, design, arm=arm)
    ad = ad[ad.obs[design.visit_col].isin(visits)].copy()

    obs = ad.obs[[design.participant_col, design.visit_col]].copy()
    if feature in ad.obs.columns:
        obs[feature] = ad.obs[feature].values
    else:
        obs[feature] = extract_gene_vector(ad, feature, layer=layer)

    obs[design.visit_col] = pd.Categorical(
        obs[design.visit_col], categories=list(visits), ordered=True
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    # Aggregate to participant-visit means to avoid pseudoreplication
    # (Zimmerman et al., 2021; Squair et al., 2021)
    obs_agg = (
        obs.groupby([design.participant_col, design.visit_col], observed=True)[feature]
        .mean()
        .reset_index()
    )
    obs_agg[design.visit_col] = pd.Categorical(
        obs_agg[design.visit_col], categories=list(visits), ordered=True
    )

    if plot_type == "box":
        sns.boxplot(
            data=obs_agg, x=design.visit_col, y=feature, ax=ax, palette="Set2", showfliers=False
        )
        sns.stripplot(data=obs_agg, x=design.visit_col, y=feature, ax=ax, color="black", alpha=0.3)
    elif plot_type == "paired":
        df_paired = obs_agg

        # Plot lines
        for p in df_paired[design.participant_col].unique():
            tmp = df_paired[df_paired[design.participant_col] == p].sort_values(design.visit_col)
            if len(tmp) == 2:
                ax.plot(tmp[design.visit_col], tmp[feature], color="gray", alpha=0.5, linewidth=1)

        # Plot points
        sns.stripplot(
            data=df_paired,
            x=design.visit_col,
            y=feature,
            hue=design.visit_col,
            ax=ax,
            palette="Set2",
            size=6,
            legend=False,
        )

    ax.set_title(f"{arm}: {feature}")
    sns.despine(ax=ax)
    return ax


def plot_trial_umap(
    adata: AnnData,
    feature: str,
    design: TrialDesign,
    visits: tuple[str, str] | None = None,
    layer: str | None = None,
    cmap: str = "magma",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Create a panel of UMAPs stratified by arm and visit.

    This creates a 2x2 panel (Treated/Control x Baseline/Followup) showing
    the expression of a feature on the UMAP embedding.

    Parameters
    ----------
    adata
        AnnData object with 'X_umap' in obsm.
    feature
        Gene or module score.
    design
        A `TrialDesign` object.
    visits
        Tuple of (baseline, followup) visit labels.
    layer
        Expression layer.
    cmap
        Colormap.
    figsize
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install sctrial[plots]"
        )
    if visits is None:
        visits = design.primary_visits()

    arms = [design.arm_treated, design.arm_control]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

    # Get global vmin/vmax for consistent scaling
    if feature in adata.obs.columns:
        vals = adata.obs[feature].values
    else:
        vals = extract_gene_vector(adata, feature, layer=layer)

    vmin = np.nanpercentile(vals, 1)
    vmax = np.nanpercentile(vals, 99)

    for i, arm in enumerate(arms):
        for j, visit in enumerate(visits):
            ax = axes[i, j]
            mask = (adata.obs[design.arm_col] == arm) & (adata.obs[design.visit_col] == visit)
            sub = adata[mask].copy()

            if sub.n_obs > 0:
                if sc is not None:
                    sc.pl.umap(
                        sub,
                        color=feature,
                        ax=ax,
                        show=False,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                        title=f"{arm} - {visit}",
                        frameon=False,
                    )
                else:
                    if "X_umap" not in sub.obsm:
                        raise KeyError("UMAP coordinates not found in adata.obsm['X_umap'].")
                    vals_sub = (
                        sub.obs[feature].values
                        if feature in sub.obs.columns
                        else extract_gene_vector(sub, feature, layer=layer)
                    )
                    ax.scatter(
                        sub.obsm["X_umap"][:, 0],
                        sub.obsm["X_umap"][:, 1],
                        c=vals_sub,
                        s=6,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_title(f"{arm} - {visit}")
                    ax.set_axis_off()
            else:
                ax.set_title(f"{arm} - {visit} (no cells)")
                ax.axis("off")

    plt.suptitle(f"Trial UMAP: {feature}", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_gsea_radar(
    gsea_results: pd.DataFrame,
    term: str,
    pool_col: str = "pool",
    nes_col: str = "NES",
    term_col: str = "Term",
    fdr_col: str = "FDR q-val",
    title: str | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> Figure:
    """Radar (spider) plot of GSEA NES across pools/cell types.

    Parameters
    ----------
    gsea_results
        Merged GSEA results table.
    term
        The pathway term to plot.
    pool_col
        Column identifying cell types or pools.
    nes_col
        Column with NES values.
    term_col
        Column containing pathway term names.
    fdr_col
        Column containing FDR q-values (used for disambiguation).
    title
        Plot title.
    figsize
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the radar plot.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install sctrial[plots]"
        )
    from math import pi

    df_term = gsea_results[gsea_results[term_col].str.contains(term, case=False, na=False)]
    if df_term.empty:
        raise ValueError(f"Term '{term}' not found in results.")

    # If multiple matches, take the best one
    if df_term[term_col].nunique() > 1:
        best_term = df_term.groupby(term_col)[fdr_col].min().idxmin()
        df_term = df_term[df_term[term_col] == best_term]

    term_name = df_term[term_col].iloc[0]

    vals = df_term.set_index(pool_col)[nes_col]
    categories = vals.index.tolist()
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    values = vals.values.tolist()
    values += values[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle="solid")
    ax.fill(angles, values, alpha=0.3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    if title is None:
        title = f"NES for {term_name}"
    ax.set_title(title, pad=20)

    return fig


def plot_trial_dotplot(
    adata: AnnData,
    features: Sequence[str],
    design: TrialDesign,
    visits: tuple[str, str] | None = None,
    use_raw: bool | None = None,
    standard_scale: str | None = None,
    cmap: str = "Reds",
    **kwargs,
) -> Any:
    """Dotplot of features across cell types and trial arms.

    Replicates the 'celltype_treatment' dotplot pattern.

    Parameters
    ----------
    adata
        AnnData object.
    features
        List of gene names to include.
    design
        TrialDesign object (must have ``celltype_col``).
    visits
        Optional (baseline, followup) visit tuple to subset.
    use_raw
        Whether to use ``adata.raw``. Defaults to ``True`` if raw exists.
    standard_scale
        Scanpy standard_scale parameter (``"var"`` or ``"group"``).
    cmap
        Colormap name.
    **kwargs
        Additional arguments passed to ``scanpy.pl.dotplot``.

    Returns
    -------
    scanpy.pl.DotPlot
        The Scanpy dotplot object (allows further customization).
    """
    if sc is None:
        raise ImportError(
            "scanpy is required for plotting. "
            "Install with: pip install sctrial[plots]"
            + (f" (scanpy import failed: {_scanpy_import_error})" if _scanpy_import_error else "")
        )
    if design.celltype_col is None:
        raise ValueError("TrialDesign.celltype_col must be set for plot_trial_dotplot.")

    ad = adata.copy()
    if visits:
        ad = ad[ad.obs[design.visit_col].isin(visits)].copy()

    # Create combined variable
    if design.arm_col is not None:
        ad.obs["_ct_arm"] = (
            ad.obs[design.celltype_col].astype(str) + "_" + ad.obs[design.arm_col].astype(str)
        )
    else:
        ad.obs["_ct_arm"] = ad.obs[design.celltype_col].astype(str)

    # Sorting
    cts = sorted(ad.obs[design.celltype_col].unique())
    arms = [design.arm_control, design.arm_treated]
    categories = []
    for ct in cts:
        for arm in arms:
            cat = f"{ct}_{arm}"
            if cat in ad.obs["_ct_arm"].values:
                categories.append(cat)

    ad.obs["_ct_arm"] = pd.Categorical(ad.obs["_ct_arm"], categories=categories, ordered=True)

    if use_raw is None:
        use_raw = adata.raw is not None

    return sc.pl.dotplot(
        ad,
        var_names=features,
        groupby="_ct_arm",
        use_raw=use_raw,
        standard_scale=standard_scale,
        color_map=cmap,
        **kwargs,
    )


def plot_abundance_interaction(
    adata: AnnData,
    celltype: str,
    design: TrialDesign,
    visits: tuple[str, str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot cell type abundance (proportion) by arm and visit.

    Parameters
    ----------
    adata
        AnnData object.
    celltype
        Cell type to plot.
    design
        TrialDesign object (must have ``celltype_col``).
    visits
        Tuple of (baseline, followup) visit labels.
    ax
        Optional matplotlib axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the abundance plot.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    if design.celltype_col is None:
        raise ValueError("TrialDesign must have celltype_col defined.")
    if visits is None:
        visits = design.primary_visits()

    id_cols = [design.participant_col, design.visit_col]
    if design.arm_col is not None:
        id_cols.append(design.arm_col)
    obs = adata.obs[id_cols + [design.celltype_col]].copy()
    obs = obs[obs[design.visit_col].isin(list(visits))].copy()

    # Calculate proportions per participant-visit, ensuring true zeros are
    # represented (a participant with no cells of a given type has proportion 0,
    # not a missing row).
    counts = (
        obs.groupby(
            id_cols + [design.celltype_col],
            observed=True,
        )
        .size()
        .reset_index(name="n")
    )
    totals = (
        obs.groupby([design.participant_col, design.visit_col], observed=True)
        .size()
        .reset_index(name="total")
    )
    counts = pd.merge(counts, totals, on=[design.participant_col, design.visit_col], how="right")
    counts["n"] = counts["n"].fillna(0)
    counts["prop"] = counts["n"] / counts["total"]

    # Filter for specific celltype — use merge to keep participant-visits
    # that have zero cells of this type.
    pv_frame = obs.groupby(id_cols, observed=True).size().reset_index(name="_n").drop(columns="_n")
    ct_counts = counts[counts[design.celltype_col] == celltype].copy()
    df_plot = pd.merge(pv_frame, ct_counts, on=id_cols, how="left")
    df_plot["prop"] = df_plot["prop"].fillna(0.0)
    df_plot[design.visit_col] = pd.Categorical(
        df_plot[design.visit_col], categories=list(visits), ordered=True
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.pointplot(
        data=df_plot,
        x=design.visit_col,
        y="prop",
        hue=design.arm_col,  # None → no hue coloring
        dodge=True,
        capsize=0.1,
        ax=ax,
    )
    ax.set_title(f"Abundance: {celltype}")
    ax.set_ylabel("Proportion")
    return ax


def plot_trial_umap_panel(
    adata: AnnData,
    feature: str,
    design: TrialDesign,
    visits: tuple[str, str] | None = None,
    layer: str | None = None,
    cmap: str = "magma",
    figsize: tuple[float, float] = (16, 8),
    title: str | None = None,
) -> Figure:
    """Combined UMAP panel: Cell Types + 4 Trial-stratified UMAPs.

    Produces a 1×3 grid: a large cell-type reference UMAP on the left and a
    2×2 grid of feature UMAPs (Treated/Control × Baseline/Followup) on the
    right.

    Parameters
    ----------
    adata
        AnnData object with ``X_umap`` in ``obsm``.
    feature
        Gene name or ``obs`` column to display.
    design
        TrialDesign object (must have ``celltype_col`` set).
    visits
        Tuple of ``(baseline, followup)`` visit labels; uses
        ``design.primary_visits()`` if ``None``.
    layer
        Expression layer for gene features (``None`` uses ``adata.X``).
    cmap
        Matplotlib colormap for the feature panels.
    figsize
        Figure size ``(width, height)``.
    title
        Optional suptitle; defaults to ``"Trial UMAP Panel: {feature}"``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the UMAP panel.
    """
    if plt is None or GridSpec is None:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install sctrial[plots]"
        )
    if visits is None:
        visits = design.primary_visits()

    ad = adata.copy()
    if feature in ad.obs.columns:
        pass
    elif feature in ad.var_names:
        ad.obs[feature] = extract_gene_vector(ad, feature, layer=layer)
        # Avoid 'both .var_names and .obs.columns' error in scanpy
        ad = ad[:, [g for g in ad.var_names if g != feature]].copy()
    else:
        raise KeyError(f"Feature '{feature}' not found.")

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.5, 1, 1])

    # 1. Big Cell Type UMAP on the left
    ax_big = fig.add_subplot(gs[:, 0])
    if design.celltype_col and design.celltype_col in ad.obs.columns:
        if sc is not None:
            sc.pl.umap(
                ad,
                color=design.celltype_col,
                ax=ax_big,
                show=False,
                frameon=False,
                title="Cell Types",
            )
        else:
            if "X_umap" not in ad.obsm:
                raise KeyError("UMAP coordinates not found in adata.obsm['X_umap'].")
            ct = ad.obs[design.celltype_col].astype("category")
            codes = ct.cat.codes
            ax_big.scatter(
                ad.obsm["X_umap"][:, 0],
                ad.obsm["X_umap"][:, 1],
                c=codes,
                s=6,
                cmap="tab20",
            )
            ax_big.set_title("Cell Types")
            ax_big.set_axis_off()
    else:
        if "X_umap" not in ad.obsm:
            raise KeyError("UMAP coordinates not found in adata.obsm['X_umap'].")
        ax_big.scatter(
            ad.obsm["X_umap"][:, 0],
            ad.obsm["X_umap"][:, 1],
            s=6,
            c="grey",
            alpha=0.3,
        )
        ax_big.set_title("All Cells")
        ax_big.set_axis_off()

    # 2. 2x2 Grid on the right
    # Calculate global vmin/vmax for consistent color scale
    vals = ad.obs[feature].values
    vmin = np.nanpercentile(vals, 1)
    vmax = np.nanpercentile(vals, 99)

    if design.arm_col is not None:
        positions = {
            (design.arm_treated, visits[0]): (0, 1),
            (design.arm_treated, visits[1]): (0, 2),
            (design.arm_control, visits[0]): (1, 1),
            (design.arm_control, visits[1]): (1, 2),
        }
    else:
        # Single-arm: one row, two columns (pre/post)
        positions = {
            ("All", visits[0]): (0, 1),
            ("All", visits[1]): (0, 2),
        }

    for (arm, visit), (r, c) in positions.items():
        ax = fig.add_subplot(gs[r, c])
        mask = ad.obs[design.visit_col] == visit
        if design.arm_col is not None:
            mask = mask & (ad.obs[design.arm_col] == arm)
        sub = ad[mask].copy()

        if sub.n_obs > 0:
            if sc is not None:
                sc.pl.umap(
                    sub,
                    color=feature,
                    ax=ax,
                    show=False,
                    frameon=False,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    title=f"{arm} - {visit}",
                )
            else:
                if "X_umap" not in sub.obsm:
                    raise KeyError("UMAP coordinates not found in adata.obsm['X_umap'].")
                vals_sub = (
                    sub.obs[feature].values
                    if feature in sub.obs.columns
                    else extract_gene_vector(sub, feature, layer=layer)
                )
                ax.scatter(
                    sub.obsm["X_umap"][:, 0],
                    sub.obsm["X_umap"][:, 1],
                    c=vals_sub,
                    s=6,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"{arm} - {visit}")
                ax.set_axis_off()
        else:
            ax.set_title(f"{arm} - {visit} (no cells)")
            ax.axis("off")

    if title is None:
        title = f"Trial UMAP Panel: {feature}"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_module_umap_panel(
    adata: AnnData,
    module_cols: Sequence[str],
    celltype_col: str = "celltype",
    umap_key: str = "X_umap",
    n_cols: int = 2,
    cmap: str = "magma",
    figsize: tuple[float, float] = (12, 10),
    point_size: float = 6,
    alpha: float = 0.7,
    label_fontsize: int = 8,
) -> Figure:
    """Plot cell-type UMAP plus module score UMAP panels.

    Creates a multi-panel figure with one cell-type reference UMAP and one
    UMAP per module score column, all sharing the same embedding coordinates.

    Parameters
    ----------
    adata
        AnnData object with a UMAP embedding in ``obsm[umap_key]``.
    module_cols
        Column names in ``adata.obs`` containing module scores to plot.
    celltype_col
        Column in ``adata.obs`` with cell-type labels (used in reference panel).
    umap_key
        Key in ``adata.obsm`` for UMAP coordinates.
    n_cols
        Number of columns in the subplot grid.
    cmap
        Matplotlib colormap for module score panels.
    figsize
        Figure size ``(width, height)``.
    point_size
        Scatter point size.
    alpha
        Point transparency.
    label_fontsize
        Font size for cell-type labels on the reference panel.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the UMAP panel.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")
    if umap_key not in adata.obsm:
        raise KeyError(f"{umap_key} not found in adata.obsm")
    for m in module_cols:
        if m not in adata.obs.columns:
            raise KeyError(f"Module score '{m}' not found in adata.obs")
    if celltype_col not in adata.obs.columns:
        raise KeyError(f"{celltype_col} not found in adata.obs")

    coords = adata.obsm[umap_key]
    x = coords[:, 0]
    y = coords[:, 1]

    n_panels = 1 + len(module_cols)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # Panel 1: cell types with labels
    ax0 = axes[0]
    celltypes = pd.Categorical(adata.obs[celltype_col])
    cats = list(celltypes.categories)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(cats))))
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(cats)}
    for c in cats:
        mask = adata.obs[celltype_col] == c
        ax0.scatter(x[mask], y[mask], s=point_size, alpha=alpha, color=color_map[c], label=c)
        # label at centroid
        ax0.text(np.median(x[mask]), np.median(y[mask]), str(c), fontsize=label_fontsize)
    ax0.set_title("Cell Types")
    ax0.set_axis_off()

    # Module panels
    for i, m in enumerate(module_cols, start=1):
        ax = axes[i]
        vals = adata.obs[m].values
        vmin = np.nanpercentile(vals, 1)
        vmax = np.nanpercentile(vals, 99)
        sca = ax.scatter(x, y, c=vals, s=point_size, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(m)
        ax.set_axis_off()
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)

    # Hide extra axes
    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def plot_gsea_heatmap(
    gsea_results: Any,
    collection: str | None = None,
    fdr_thresh: float = 0.25,
    top_n: int = 30,
    pool_col: str = "pool",
    term_col: str = "Term",
    nes_col: str = "NES",
    fdr_col: str = "FDR q-val",
    figsize: tuple[float, float] = (12, 10),
    title: str | None = None,
) -> Axes:
    """Heatmap of GSEA NES across pools (cell types).

    Parameters
    ----------
    gsea_results
        Merged GSEA results table.
    collection
        If provided, subset to this collection (e.g. 'HALLMARK').
    fdr_thresh
        Only include pathways that are significant (FDR < thresh) in at least one pool.
    top_n
        Number of top pathways to show (ranked by minimum FDR across pools).
    figsize
        Figure size.
    title
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    # If it's a gseapy Prerank object, extract res2d
    df: pd.DataFrame
    if hasattr(gsea_results, "res2d"):
        df = gsea_results.res2d.copy()
    else:
        df = pd.DataFrame(gsea_results).copy()

    if pool_col not in df.columns:
        # Assume global result if pool column is missing
        df[pool_col] = "Global"

    if collection:
        if "collection" in df.columns:
            df = df[df["collection"] == collection]

    # 1. Filter significant
    df_sig = df[df[fdr_col] <= fdr_thresh].copy()
    if df_sig.empty:
        # Fallback to top_n without FDR threshold if none significant
        top_terms = df.groupby(term_col)[fdr_col].min().sort_values().head(top_n).index.tolist()
    else:
        # 2. Rank pathways by minimum FDR
        top_terms = df_sig.groupby(term_col)[fdr_col].min().sort_values().head(top_n).index.tolist()

    if not top_terms:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No pathways to plot", ha="center")
        return ax

    # 3. Pivot
    df_top = df[df[term_col].isin(top_terms)].copy()
    mat = df_top.pivot_table(index=term_col, columns=pool_col, values=nes_col, aggfunc="mean")

    # Sort terms by original min FDR ranking (use reindex to handle missing terms gracefully)
    mat = mat.reindex(top_terms)

    # Ensure numeric
    mat = mat.astype(float)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        mat, cmap="RdBu_r", center=0, linewidths=0.5, linecolor="gray", cbar_kws={"label": "NES"}
    )

    if title is None:
        title = f"GSEA NES Heatmap: {collection if collection else 'Top Pathways'}"
    plt.title(title)

    plt.xlabel("Pool")
    plt.ylabel("Pathway")
    plt.tight_layout()
    return ax


def plot_did_forest_interactive(
    df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    beta_col: str = "beta_DiD",
    se_col: str = "se_DiD",
    p_col: str = "p_DiD",
    alpha: float = 0.05,
    title: str = "DiD Effect Sizes",
) -> Any:
    """Interactive forest plot using Plotly.

    Parameters
    ----------
    df
        DataFrame with DiD results (from ``did_table`` or ``abundance_did``).
    feature_col
        Column name for feature labels (y-axis).
    beta_col
        Column name for effect size estimates.
    se_col
        Column name for standard errors (used for 95 % CI error bars).
    p_col
        Column name for p-values (used to color significant points).
    alpha
        Significance threshold for coloring points.
    title
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive forest plot.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotly is required for interactive plots") from e

    if df.empty:
        raise ValueError("Empty DataFrame provided.")
    # Fallback for abundance_did which uses 'celltype' instead of 'feature'
    if feature_col not in df.columns:
        if feature_col == "feature" and "celltype" in df.columns:
            feature_col = "celltype"
        else:
            raise KeyError(f"Missing column '{feature_col}'")

    from scipy import stats as sp_stats

    df_plot = df.dropna(subset=[beta_col, se_col]).copy()
    if "n_units" in df_plot.columns:
        df_vals = (df_plot["n_units"] - 2).clip(lower=1)
        t_crit = df_vals.apply(lambda d: sp_stats.t.ppf(0.975, d))
        df_plot["ci"] = t_crit * df_plot[se_col]
    else:
        df_plot["ci"] = 1.96 * df_plot[se_col]
    df_plot = df_plot.sort_values(beta_col)

    sig = df_plot[p_col] < alpha if p_col in df_plot.columns else False
    color = np.where(sig, "crimson", "gray")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot[beta_col],
            y=df_plot[feature_col],
            error_x=dict(type="data", array=df_plot["ci"], visible=True),
            mode="markers",
            marker=dict(color=color),
        )
    )
    fig.update_layout(title=title, xaxis_title="Effect (beta)", yaxis_title="Feature")
    return fig


def plot_did_volcano_interactive(
    df: pd.DataFrame,
    *,
    beta_col: str = "beta_DiD",
    p_col: str = "p_DiD",
    title: str = "DiD Volcano Plot",
) -> Any:
    """Interactive volcano plot using Plotly.

    Parameters
    ----------
    df
        DataFrame with DiD results.
    beta_col
        Column name for effect size estimates (x-axis).
    p_col
        Column name for p-values (y-axis as ``-log10(p)``).
    title
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive volcano plot.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotly is required for interactive plots") from e

    df_plot = df.dropna(subset=[beta_col, p_col]).copy()
    df_plot["neglog10p"] = -np.log10(df_plot[p_col].clip(lower=1e-12))

    fig = go.Figure(
        data=go.Scatter(
            x=df_plot[beta_col],
            y=df_plot["neglog10p"],
            mode="markers",
        )
    )
    fig.update_layout(title=title, xaxis_title="Effect (beta)", yaxis_title="-log10(p)")
    return fig
