from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

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
    "plot_did_forest",
    "plot_within_arm_comparison",
    "plot_trial_umap",
    "plot_gsea_radar",
    "plot_trial_dotplot",
    "plot_abundance_interaction",
    "plot_trial_umap_panel",
    "plot_gsea_heatmap",
]

# Optional dependencies for plotting
# Ensure caches/config go to writable temp dirs to avoid numba/matplotlib failures.
ensure_matplotlib_config_dir()
ensure_numba_cache_dir()

# Initialize optional plotting dependencies to avoid type errors
plt: Any = None
GridSpec: Any = None
sns: Any = None
sc: Any = None
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
except (ImportError, RuntimeError, OSError) as e:  # ImportError or runtime errors (e.g., numba cache)
    _scanpy_import_error = e

def did_volcano_frame(
        df: pd.DataFrame,
        *,
        effect_col: str = "beta_DiD",
        p_col: str = "p_DiD",
        out_col: str = "neglog10p",
        p_floor: float = 1e-300,
) -> pd.DataFrame:
    """Return a copy with an added -log10(p) column for volcano plots."""
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
    """Return sign(effect) * -log10(p) as a Series aligned to df.index."""
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
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot mean expression (interaction plot) by arm and visit.

    This visualizes the DiD effect: the change from baseline to follow-up
    across treatment arms.
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    if visits is None:
        visits = design.primary_visits()

    # extract data
    obs = adata.obs[[design.arm_col, design.visit_col]].copy()
    if feature in adata.obs.columns:
        obs[feature] = adata.obs[feature].values
    elif feature in adata.var_names:
        obs[feature] = extract_gene_vector(adata, feature, layer=layer)
    else:
        raise KeyError(f"Feature '{feature}' not found.")

    # subset to relevant visits
    obs = obs[obs[design.visit_col].isin(list(visits))].copy()
    obs[design.visit_col] = pd.Categorical(obs[design.visit_col], categories=list(visits), ordered=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.pointplot(
        data=obs,
        x=design.visit_col,
        y=feature,
        hue=design.arm_col,
        palette=color_palette,
        dodge=True,
        capsize=0.1,
        ax=ax,
    )
    ax.set_title(f"Trial interaction: {feature}")
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
    ax: plt.Axes | None = None,
) -> plt.Axes:
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

    df_plot["ci"] = 1.96 * df_plot[se_col]
    df_plot = df_plot.sort_values(beta_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 0.5 * len(df_plot) + 1))

    # Plot zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Plot points and CIs
    y_pos = np.arange(len(df_plot))

    # Identify significant ones
    sig: Any
    if p_col in df_plot.columns:
        sig = df_plot[p_col] < alpha
    else:
        sig = np.zeros(len(df_plot), dtype=bool)

    # Standard points
    ax.errorbar(
        df_plot.loc[~sig, beta_col],
        y_pos[~sig],
        xerr=df_plot.loc[~sig, "ci"],
        fmt="o",
        color="gray",
        label=f"p >= {alpha}" if sig.any() else None,
        capsize=3
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
            capsize=3
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
    ax: plt.Axes | None = None,
) -> plt.Axes:
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

    obs[design.visit_col] = pd.Categorical(obs[design.visit_col], categories=list(visits), ordered=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    if plot_type == "box":
        sns.boxplot(data=obs, x=design.visit_col, y=feature, ax=ax, palette="Set2", showfliers=False)
        sns.stripplot(data=obs, x=design.visit_col, y=feature, ax=ax, color="black", alpha=0.3)
    elif plot_type == "paired":
        # Group by participant and visit, then plot lines
        # First, aggregate to participant-visit mean if multiple cells
        df_paired = obs.groupby([design.participant_col, design.visit_col], observed=True)[feature].mean().reset_index()

        # Plot lines
        for p in df_paired[design.participant_col].unique():
            tmp = df_paired[df_paired[design.participant_col] == p].sort_values(design.visit_col)
            if len(tmp) == 2:
                ax.plot(tmp[design.visit_col], tmp[feature], color="gray", alpha=0.5, linewidth=1)

        # Plot points
        sns.stripplot(data=df_paired, x=design.visit_col, y=feature, hue=design.visit_col, ax=ax, palette="Set2", size=6, legend=False)

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
) -> plt.Figure:
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
    if plt is None or sc is None:
        raise ImportError(
            "matplotlib and scanpy are required for plotting. "
            "Install with: pip install sctrial[plots]"
            + (f" (scanpy import failed: {_scanpy_import_error})" if _scanpy_import_error else "")
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
                sc.pl.umap(
                    sub,
                    color=feature,
                    ax=ax,
                    show=False,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    title=f"{arm} - {visit}",
                    frameon=False
                )
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
    title: str | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> plt.Figure:
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
    title
        Plot title.
    figsize
        Figure size.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    from math import pi
    df_term = gsea_results[gsea_results["Term"].str.contains(term, case=False, na=False)]
    if df_term.empty:
        raise ValueError(f"Term '{term}' not found in results.")

    # If multiple matches, take the best one
    if df_term["Term"].nunique() > 1:
        best_term = df_term.groupby("Term")["FDR q-val"].min().idxmin()
        df_term = df_term[df_term["Term"] == best_term]

    term_name = df_term["Term"].iloc[0]

    vals = df_term.set_index(pool_col)[nes_col]
    categories = vals.index.tolist()
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    values = vals.values.tolist()
    values += values[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
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
    **kwargs
):
    """Dotplot of features across cell types and trial arms.

    Replicates the 'celltype_treatment' dotplot pattern.
    """
    if sc is None:
        raise ImportError(
            "scanpy is required for plotting. "
            "Install with: pip install sctrial[plots]"
            + (f" (scanpy import failed: {_scanpy_import_error})" if _scanpy_import_error else "")
        )
    ad = adata.copy()
    if visits:
        ad = ad[ad.obs[design.visit_col].isin(visits)].copy()

    # Create combined variable
    ad.obs["_ct_arm"] = (
        ad.obs[design.celltype_col].astype(str) +
        "_" +
        ad.obs[design.arm_col].astype(str)
    )

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
        **kwargs
    )


def plot_abundance_interaction(
    adata: AnnData,
    celltype: str,
    design: TrialDesign,
    visits: tuple[str, str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot cell type abundance (proportion) by arm and visit."""
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    if design.celltype_col is None:
        raise ValueError("TrialDesign must have celltype_col defined.")
    if visits is None:
        visits = design.primary_visits()

    obs = adata.obs[[design.participant_col, design.arm_col, design.visit_col, design.celltype_col]].copy()
    obs = obs[obs[design.visit_col].isin(list(visits))].copy()

    # Calculate proportions per participant-visit
    counts = obs.groupby([design.participant_col, design.visit_col, design.arm_col, design.celltype_col], observed=True).size().reset_index(name="n")
    totals = counts.groupby([design.participant_col, design.visit_col], observed=True)["n"].transform("sum")
    counts["prop"] = counts["n"] / totals

    # Filter for specific celltype
    df_plot = counts[counts[design.celltype_col] == celltype].copy()
    df_plot[design.visit_col] = pd.Categorical(df_plot[design.visit_col], categories=list(visits), ordered=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.pointplot(
        data=df_plot,
        x=design.visit_col,
        y="prop",
        hue=design.arm_col,
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
) -> plt.Figure:
    """Combined UMAP panel: Cell Types + 4 Trial-stratified UMAPs.

    Replicates the layout: [Large Cell Type UMAP] [2x2 Grid of Feature UMAPs].
    """
    if plt is None or sc is None or GridSpec is None:
        raise ImportError(
            "matplotlib and scanpy are required for plotting. "
            "Install with: pip install sctrial[plots]"
            + (f" (scanpy import failed: {_scanpy_import_error})" if _scanpy_import_error else "")
        )
    if visits is None:
        visits = design.primary_visits()

    # Pre-extract feature
    from .stats._extract import extract_gene_vector
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
    sc.pl.umap(ad, color=design.celltype_col, ax=ax_big, show=False, frameon=False, title="Cell Types")

    # 2. 2x2 Grid on the right
    # Calculate global vmin/vmax for consistent color scale
    vals = ad.obs[feature].values
    vmin = np.nanpercentile(vals, 1)
    vmax = np.nanpercentile(vals, 99)

    positions = {
        (design.arm_treated, visits[0]): (0, 1),
        (design.arm_treated, visits[1]): (0, 2),
        (design.arm_control, visits[0]): (1, 1),
        (design.arm_control, visits[1]): (1, 2),
    }

    for (arm, visit), (r, c) in positions.items():
        ax = fig.add_subplot(gs[r, c])
        sub = ad[(ad.obs[design.arm_col] == arm) & (ad.obs[design.visit_col] == visit)]

        if sub.n_obs > 0:
            sc.pl.umap(
                sub, color=feature, ax=ax, show=False, frameon=False,
                vmin=vmin, vmax=vmax, cmap=cmap, title=f"{arm} - {visit}"
            )
        else:
            ax.set_title(f"{arm} - {visit} (no cells)")
            ax.axis("off")

    if title is None:
        title = f"Trial UMAP Panel: {feature}"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_gsea_heatmap(
    gsea_results: pd.DataFrame,
    collection: str | None = None,
    fdr_thresh: float = 0.25,
    top_n: int = 30,
    pool_col: str = "pool",
    term_col: str = "Term",
    nes_col: str = "NES",
    fdr_col: str = "FDR q-val",
    figsize: tuple[float, float] = (12, 10),
    title: str | None = None,
) -> plt.Axes:
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
    """
    if plt is None or sns is None:
        raise ImportError(
            "matplotlib and seaborn are required for plotting. "
            "Install with: pip install sctrial[plots]"
        )
    # If it's a gseapy Prerank object, extract res2d
    df: Any
    if hasattr(gsea_results, "res2d"):
        df = gsea_results.res2d.copy()
    else:
        df = gsea_results.copy()

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
    ax = sns.heatmap(mat, cmap="RdBu_r", center=0, linewidths=0.5, linecolor="gray", cbar_kws={"label": "NES"})

    if title is None:
        title = f"GSEA NES Heatmap: {collection if collection else 'Top Pathways'}"
    plt.title(title)

    plt.xlabel("Pool")
    plt.ylabel("Pathway")
    plt.tight_layout()
    return ax
