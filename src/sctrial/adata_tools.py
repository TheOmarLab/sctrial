"""AnnData manipulation utilities: subsetting, merging, pseudobulk helpers."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from .design import TrialDesign

__all__ = ["subset_primary", "subset_arms", "subset_cells", "profile_features"]

logger = logging.getLogger(__name__)


def _require_cols(obs: pd.DataFrame, cols: list[str]) -> None:
    """Require specific columns in the DataFrame."""
    missing = [c for c in cols if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}. Available: {list(obs.columns)}")


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Best-effort conversion of a metadata column to boolean.

    Handles bool, numeric (any non-zero → True), and string/categorical
    columns (recognises "true", "t", "yes", "y", "1" as truthy).

    Non-finite numeric values (NaN, inf) are treated as ``False``.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # numeric: any non-zero finite value is truthy
    if pd.api.types.is_numeric_dtype(s):
        n_nonfinite = int(s.isna().sum() + np.isinf(s.fillna(0)).sum())
        if n_nonfinite > 0:
            logger.warning(
                "%d non-finite value(s) (NaN/inf) in boolean column '%s'; treating as False.",
                n_nonfinite,
                s.name,
            )
        filled = s.fillna(0)
        finite_mask = np.isfinite(filled)
        return (finite_mask & (filled != 0)).astype(bool)

    # strings / categoricals
    ss = s.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y"}
    out = ss.isin(true_vals)
    return out.fillna(False).astype(bool)


def subset_primary(
    adata: AnnData,
    design: TrialDesign,
    visits: tuple[str, str],
    exclude_crossovers: bool = True,
) -> AnnData:
    """Subset AnnData to the primary (baseline, followup) visits.

    Parameters
    ----------
    visits:
        Tuple of (baseline_visit, followup_visit), e.g. ("3/T0", "6/T12w").
    exclude_crossovers:
        If True and design.crossover_col is provided, drop rows where crossover_col is truthy.

    Returns
    -------
    AnnData
        Subsetted AnnData object.
    """
    obs = adata.obs
    _require_cols(obs, [design.visit_col])

    mask = obs[design.visit_col].isin(list(visits)).to_numpy(dtype=bool)

    if exclude_crossovers and design.crossover_col:
        _require_cols(obs, [design.crossover_col])
        cross = _to_bool_series(obs[design.crossover_col]).to_numpy(dtype=bool)
        mask &= ~cross

    return adata[mask].copy()


_MAX_LABELS_IN_MESSAGE = 10


def subset_arms(
    data: AnnData | pd.DataFrame, design: TrialDesign, *, stacklevel: int = 2
) -> AnnData | pd.DataFrame:
    """Keep the rows of the two arms named by the design.

    A two-arm analysis compares ``design.arm_treated`` with ``design.arm_control``.
    A row whose arm is any other label, or is missing, belongs to neither group and
    is removed. One ``UserWarning`` names the labels removed and counts the
    participants that carry them. The two-arm functions of the package build their
    model frame through this function.

    Data that hold rows of other labels but no row of one named arm do not match the
    design (for example ``arm_control="Control"`` on data labeled ``"Placebo"``), and
    raise. Data that hold only one of the two named arms and nothing else are returned
    unchanged; the estimators report a missing estimate for them.

    Parameters
    ----------
    data
        AnnData, or a DataFrame that holds ``design.arm_col`` (``adata.obs`` or a
        table with one row per participant and visit).
    design
        Trial design with ``arm_col``, ``arm_treated`` and ``arm_control``.
    stacklevel
        Passed to ``warnings.warn``.

    Returns
    -------
    AnnData or pd.DataFrame
        ``data`` itself when every row belongs to a named arm, otherwise a copy
        restricted to the rows of the two named arms.

    Raises
    ------
    ValueError
        If the design has no arm column, if ``arm_treated`` equals ``arm_control``,
        or if the data hold rows of other labels and no row of one named arm.
    KeyError
        If ``design.arm_col`` is not a column of the data.

    Examples
    --------
    >>> two_arms = subset_arms(adata, design)
    >>> res = did_table(two_arms, features=features, design=design, visits=("V1", "V2"))
    """
    if design.arm_col is None:
        raise ValueError(
            "A two-arm analysis requires design.arm_col. For a single-arm design "
            "(arm_col=None) use the within-arm functions."
        )
    if design.arm_treated == design.arm_control:
        raise ValueError(
            "A two-arm analysis requires two distinct arm labels, but "
            f"arm_treated == arm_control == {design.arm_treated!r}. For one arm use "
            "within_arm_comparison or a design with arm_col=None."
        )
    obs = data.obs if isinstance(data, AnnData) else data
    _require_cols(obs, [design.arm_col])
    if len(obs) == 0:
        return data

    arm = obs[design.arm_col]
    is_treated = arm.isin([design.arm_treated]).to_numpy(dtype=bool)
    is_control = arm.isin([design.arm_control]).to_numpy(dtype=bool)
    keep = is_treated | is_control
    if keep.all():
        return data

    absent = [
        label
        for label, rows in ((design.arm_treated, is_treated), (design.arm_control, is_control))
        if not rows.any()
    ]
    if absent:
        present = sorted(str(v) for v in pd.unique(arm.dropna()))
        raise ValueError(
            f"No rows of arm {' or '.join(repr(a) for a in absent)} in obs[{design.arm_col!r}] "
            f"(labels present: {present}). The design names arm_treated="
            f"{design.arm_treated!r} and arm_control={design.arm_control!r}; a two-arm "
            f"analysis uses rows with these two labels only."
        )

    outside = obs.loc[~keep]
    by_participant = design.participant_col in outside.columns

    def _count(rows: pd.DataFrame) -> int:
        return int(rows[design.participant_col].nunique()) if by_participant else len(rows)

    labels = outside[design.arm_col]
    named = sorted(pd.unique(labels.dropna()), key=str)
    parts = [
        f"{label!r} ({_count(outside[labels.isin([label]).to_numpy()])})"
        for label in named[:_MAX_LABELS_IN_MESSAGE]
    ]
    if len(named) > _MAX_LABELS_IN_MESSAGE:
        parts.append(f"{len(named) - _MAX_LABELS_IN_MESSAGE} more labels")
    if labels.isna().any():
        parts.append(f"missing label ({_count(outside[labels.isna().to_numpy()])})")
    warnings.warn(
        f"obs[{design.arm_col!r}] holds rows outside the two arms of the design "
        f"({design.arm_treated!r} and {design.arm_control!r}). They are excluded from the "
        f"two-arm analysis. {'Participants' if by_participant else 'Rows'} per excluded "
        f"label: {', '.join(parts)}.",
        UserWarning,
        stacklevel=stacklevel,
    )
    if isinstance(data, AnnData):
        return data[keep].copy()
    return data.loc[keep].copy()


def subset_cells(
    adata: AnnData,
    design: TrialDesign,
    arm: str | None = None,
    visit: str | None = None,
    celltype: str | None = None,
    exclude_crossovers: bool = False,
) -> AnnData:
    """General-purpose subsetting helper by arm/visit/celltype (+ optional crossover exclusion).

    Parameters
    ----------
    adata
        AnnData object.
    design
        TrialDesign object.
    arm
        Arm to subset by.
    visit
        Visit to subset by.
    celltype
        Celltype to subset by.
    exclude_crossovers
        If True, exclude crossovers.

    Returns
    -------
    AnnData
        Subsetted AnnData object.

    """
    obs = adata.obs

    required = []
    if arm is not None and design.arm_col is not None:
        required.append(design.arm_col)
    if visit is not None:
        required.append(design.visit_col)
    if celltype is not None:
        if not design.celltype_col:
            raise ValueError("celltype_col must be set in TrialDesign when filtering by celltype.")
        required.append(design.celltype_col)
    if exclude_crossovers and design.crossover_col:
        required.append(design.crossover_col)

    if required:
        _require_cols(obs, required)

    mask = np.ones(obs.shape[0], dtype=bool)

    if arm is not None and design.arm_col is not None:
        mask &= obs[design.arm_col].to_numpy() == arm

    if visit is not None:
        mask &= obs[design.visit_col].to_numpy() == visit

    if celltype is not None:
        mask &= obs[design.celltype_col].to_numpy() == celltype

    if exclude_crossovers and design.crossover_col:
        cross = _to_bool_series(obs[design.crossover_col]).to_numpy(dtype=bool)
        mask &= ~cross

    return adata[mask].copy()


def profile_features(
    adata: AnnData,
    features: Sequence[str],
    groupby: str,
    layer: str | None = None,
    agg: str = "mean",
) -> pd.DataFrame:
    """Calculate aggregate expression of features across groups.

    Useful for profiling marker sets across clusters or trial arms.

    Parameters
    ----------
    adata
        AnnData object.
    features
        Genes or obs columns to aggregate.
    groupby
        Column in `adata.obs` to group by.
    layer
        Expression layer to use for genes.
    agg
        Aggregation function ('mean', 'median', etc. supported by pandas).

    Returns
    -------
    pd.DataFrame
        Table with index `groupby` and columns `features`, containing
        aggregated feature values.

    Notes
    -----
    Aggregation is performed at the **cell level**.  When grouping by
    treatment arm or participant, groups with more cells will dominate
    the aggregate.  For participant-level or balanced comparisons, use
    :func:`~sctrial.stats.pseudobulk.pseudobulk_expression` or
    pre-aggregate to pseudobulk before calling this function.
    """
    from .stats._extract import extract_gene_vector

    _require_cols(adata.obs, [groupby])

    res = {}
    for feat in features:
        if feat in adata.obs.columns:
            res[feat] = adata.obs[feat].values
        elif feat in adata.var_names:
            res[feat] = extract_gene_vector(adata, feat, layer=layer)
        else:
            raise KeyError(f"Feature '{feat}' not found in obs or var_names.")

    df = pd.DataFrame(res, index=adata.obs_names)
    df[groupby] = adata.obs[groupby].values

    return df.groupby(groupby, observed=True).agg(agg)
