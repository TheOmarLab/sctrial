"""Fit the public score models on their participant-level sufficient input."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd
from anndata import AnnData

from ..design import TrialDesign
from ..stats.comparisons import between_arm_comparison, within_arm_comparison
from ..stats.did import did_table

METHODS = {
    "did": "Participant fixed-effects difference-in-differences",
    "paired": "Paired participant fixed-effects change",
    "cross": "Participant-level ordinary least squares",
}


def _finite(value: Any) -> float | None:
    return float(value) if value is not None and pd.notna(value) and math.isfinite(value) else None


def _setting(request: dict, key: str) -> str:
    value = request.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Choose a valid {key} for this comparison.")
    return value


def fit_scores(study: dict, summary: pd.DataFrame, request: dict) -> dict:
    mode = request.get("design")
    if not isinstance(mode, str) or mode not in METHODS:
        raise ValueError("Choose a paired, two-group longitudinal, or cross-sectional comparison.")
    first, other = (_setting(request, k) for k in ("arm", "reference"))
    baseline, visit, population = (
        _setting(request, k) for k in ("baseline", "visit", "population")
    )
    groups = [person["arm"] for person in study["participants"]]
    if first not in groups or (mode != "paired" and (first == other or other not in groups)):
        raise ValueError(
            "Choose a study group and, for a two-group comparison, a different reference group."
        )
    if population not in study["populations"] or visit not in study["visits"]:
        raise ValueError("Choose a population and visit from the confirmed study.")
    visits = [visit] if mode == "cross" else [baseline, visit]
    if mode != "cross" and (baseline not in study["visits"] or baseline == visit):
        raise ValueError("Choose two different observed visits.")
    arms = [first] if mode == "paired" else [first, other]
    keep = summary.population.eq(population) & summary.arm.isin(arms) & summary.visit.isin(visits)
    selected = request.get("participants")
    if selected is not None:
        people = {person["id"] for person in study["participants"]}
        if not isinstance(selected, list) or any(
            not isinstance(p, str) or p not in people for p in selected
        ):
            raise ValueError("The selection contains participants outside this study.")
        keep &= summary.participant.isin(selected)
    data = summary.loc[keep].copy()
    if mode != "cross":
        paired_ids = data.groupby("participant", observed=True).visit.nunique()
        data = data[data.participant.isin(paired_ids.index[paired_ids == 2])].copy()
        # The core fixed-effects fit has one design column per participant.
        if len(data) * (data.participant.nunique() + 2) * 8 > 32 * 1024**2:
            raise ValueError(
                "This fixed-effects design exceeds the local interactive fit budget. Use the Python workflow for this cohort."
            )
    data.index = pd.Index([str(i) for i in range(len(data))])
    features = study["features"]
    counts: dict[str, tuple[int, int]] = {}
    candidates = []
    for feature in features:
        key = feature["id"]
        valid = data[data[key].notna()]
        if mode != "cross":
            both = valid.groupby("participant", observed=True).visit.nunique()
            valid = valid[valid.participant.isin(both.index[both == 2])]
        per_arm = valid.groupby("arm", observed=True).participant.nunique()
        n_first, n_other = (
            int(per_arm.get(first, 0)),
            int(per_arm.get(other, 0)) if mode != "paired" else 0,
        )
        counts[key] = (n_first, n_other)
        if n_first and (mode == "paired" or n_other):
            candidates.append(key)
    suffix = {"did": "DiD", "paired": "time", "cross": "arm"}[mode]
    table = pd.DataFrame()
    notices: list[str] = []
    if candidates:
        # Every row is already a participant-visit mean for one selected population.
        design = TrialDesign(
            participant_col="participant",
            arm_col="arm",
            visit_col="visit",
            arm_treated=first,
            arm_control=other or "",
            celltype_col=None,
        )
        adata = AnnData(obs=data[["participant", "arm", "visit", *candidates]])
        common = dict(
            adata=adata,
            features=candidates,
            design=design,
            standardize=False,
            aggregate="participant_visit",
        )
        with warnings.catch_warnings(record=True) as collected:
            warnings.simplefilter("always", UserWarning)
            if mode == "did":
                table = did_table(**common, visits=(baseline, visit))
            elif mode == "paired":
                table = within_arm_comparison(**common, arm=first, visits=(baseline, visit))
            else:
                table = between_arm_comparison(**common, visit=visit, method="ols")
        table = table.set_index("feature")
        notices = list(
            dict.fromkeys(
                str(item.message) for item in collected if issubclass(item.category, UserWarning)
            )
        )
    rows = []
    for feature in features:
        key = feature["id"]
        record = table.loc[key] if key in table.index else {}
        n_first, n_other = counts[key]
        estimate = _finite(record.get("beta_" + suffix))
        p = _finite(record.get("p_" + suffix))
        covariance = record.get(
            "cov_type_used", "nonrobust" if mode == "cross" and estimate is not None else None
        )
        rows.append(
            {
                **feature,
                "estimate": estimate,
                "se": _finite(record.get("se_" + suffix)),
                "lo": _finite(record.get("ci_lo_" + suffix)),
                "hi": _finite(record.get("ci_hi_" + suffix)),
                "p": p,
                "q": _finite(record.get("FDR_" + suffix)),
                "n": int(record.get("n_units", n_first + n_other)),
                "nFirst": n_first,
                "nOther": n_other if mode != "paired" else None,
                "covariance": covariance if pd.notna(covariance) else None,
                "reason": "Contrast is not estimable."
                if estimate is None
                else "The fit returned no valid test."
                if p is None
                else None,
            }
        )
    return {
        "studyId": study["studyId"],
        "rows": rows,
        "unit": study["unit"],
        "method": METHODS[mode],
        "warnings": notices,
    }
