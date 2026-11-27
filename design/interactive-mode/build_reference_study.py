"""Build the synthetic study and its fitted contrasts with sctrial 1.1.0.

Run against commit 0315adfb91023d774b595cf5185d72874cd3b5ba.
The output contains participant-visit means and model results. Individual cells
are used to construct the AnnData input and are not included in the JSON.
"""

import argparse
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import sctrial as st
import statsmodels.formula.api as smf

ARMS = ("Arm A", "Arm B")
VISITS = ("Baseline", "Week 4", "Week 12")
TYPES = ("CD8 T cells", "B cells", "CD4 T cells")
POOLED = "All annotated cells"
POPULATIONS = (POOLED, *TYPES[:2])
FEATURES = {
    "interferon": "Interferon response",
    "cytotoxicity": "Cytotoxicity",
    "inflammation": "Inflammatory response",
    "proliferation": "Proliferation",
}
SOURCES = {
    "did": "Participant fixed effects with participant-clustered covariance. The 95% interval comes from the matching fitted model.",
    "paired": "Participant fixed effects with participant-clustered covariance. The 95% interval is returned by within_arm_comparison.",
    "cross": "Participant-level ordinary least squares. The 95% interval is returned by between_arm_comparison.",
}
ABSENT_SPECIMENS = {"Week 4": {"A05", "B03"}, "Week 12": {"A02", "A05", "B04", "B06"}}
ABSENT_TYPES = {
    "CD8 T cells": {"Week 4": {"A06", "B05"}, "Week 12": {"A03", "B02"}},
    "B cells": {"Baseline": {"B01"}, "Week 4": {"A04", "B06"}, "Week 12": {"A01", "B05"}},
}
SEED = 62834
COMMIT = "0315adfb91023d774b595cf5185d72874cd3b5ba"


def study():
    rng = np.random.default_rng(SEED)
    baseline = np.array([[0.28, 0.64, 0.18, 0.14], [0.20, 0.05, 0.32, 0.17], [0.26, 0.19, 0.23, 0.21]])
    arm_a = np.array([[0.60, 0.43, -0.10, 0.13], [0.29, 0.03, -0.20, 0.07], [0.48, 0.19, -0.08, 0.11]])
    shared_change = np.array([0.15, 0.08, 0.03, 0.04])
    records = []
    people = []
    for arm in ARMS:
        for i in range(1, 7):
            person = arm[-1] + f"{i:02d}"
            people.append({"id": person, "arm": arm})
            offset = rng.normal(0, .14, (3, 4))
            slopes = rng.normal(0, .17, (3, 4))
            for visit_index, visit in enumerate(VISITS):
                if person in ABSENT_SPECIMENS.get(visit, set()):
                    continue
                for type_index, celltype in enumerate(TYPES):
                    if person in ABSENT_TYPES.get(celltype, {}).get(visit, set()):
                        continue
                    count = int(rng.integers(23, 87))
                    timing = [0.0, 1.0, 0.66][visit_index]
                    levels = baseline[type_index] + offset[type_index] + timing * (
                        slopes[type_index] + shared_change + (arm_a[type_index] if arm == "Arm A" else 0)
                    )
                    levels += rng.normal(0, .055, 4)
                    cell_scores = rng.normal(levels, .21, (count, 4))
                    frame = pd.DataFrame(cell_scores, columns=FEATURES)
                    frame = frame.assign(participant_id=person, arm=arm, visit=visit, cell_type=celltype)
                    records.append(frame)
    cells = pd.concat(records, ignore_index=True)
    cells.index = [f"cell-{i:05d}" for i in range(len(cells))]
    return ad.AnnData(obs=cells, X=np.zeros((len(cells), 1), dtype=np.float32)), people


def observation_summaries(data):
    records = []
    for population in (POOLED, *TYPES):
        frame = data.obs if population == POOLED else data.obs[data.obs.cell_type == population]
        groups = frame.groupby(["participant_id", "arm", "visit"], sort=False, observed=True)
        for (person, arm, visit), samples in groups:
            records.append({
                "id": person, "arm": arm, "visit": visit, "population": population, "cells": len(samples),
                "scores": {key: float(samples[key].mean()) for key in FEATURES},
            })
    return records


def did_bounds(aggregates, ids, first, visit, feature, row):
    # Reconstruct the same participant fixed-effects model to read its interval.
    # The public DiD table returns an analytical estimate, SE and p-value.
    model_data = aggregates.loc[aggregates.id.isin(ids) & aggregates.visit.isin([VISITS[0], visit])].copy()
    model_data["post"] = (model_data.visit == visit).astype(int)
    model_data["arm_indicator"] = (model_data.arm == first).astype(int)
    model_data["score"] = [s[feature] for s in model_data.scores]
    result = smf.ols("score ~ post + post:arm_indicator + C(id)", data=model_data).fit(
        cov_type="cluster", cov_kwds={"groups": model_data.id}
    )
    term = "post:arm_indicator"
    np.testing.assert_allclose(
        [result.params[term], result.bse[term], result.pvalues[term]],
        [row.beta_DiD, row.se_DiD, row.p_DiD],
        rtol=1e-8, atol=1e-12,
    )
    return list(map(float, result.conf_int(alpha=.05).loc[term]))


def fitted_tables(data, observations):
    summaries = pd.DataFrame(observations)
    tables = {}
    for population in POPULATIONS:
        selected = data if population == POOLED else data[data.obs.cell_type == population].copy()
        aggregates = summaries[summaries.population == population]
        for first, second in (ARMS, ARMS[::-1]):
            design = st.TrialDesign(
                participant_col="participant_id", visit_col="visit", arm_col="arm",
                arm_treated=first, arm_control=second, celltype_col="cell_type",
            )
            common = dict(adata=selected, features=list(FEATURES), design=design, standardize=False, aggregate="participant_visit")
            for mode, suffix in (("did", "DiD"), ("paired", "time"), ("cross", "arm")):
                for visit in VISITS if mode == "cross" else VISITS[1:]:
                    observed = aggregates[aggregates.visit == visit]
                    if mode != "cross":
                        observed = observed[observed.id.isin(aggregates[aggregates.visit == VISITS[0]].id)]
                    if mode == "paired":
                        observed = observed[observed.arm == first]
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=r"Only \d+ clusters .*", category=UserWarning)
                        if mode == "did":
                            fits = st.did_table(**common, visits=(VISITS[0], visit))
                        elif mode == "paired":
                            fits = st.within_arm_comparison(**common, arm=first, visits=(VISITS[0], visit))
                        else:
                            fits = st.between_arm_comparison(**common, visit=visit, method="ols")
                    fits = fits.set_index("feature")
                    rows = []
                    for feature in FEATURES:
                        row = fits.loc[feature]
                        assert int(row.n_units) == len(observed)
                        if mode == "did":
                            lo, hi = did_bounds(aggregates, observed.id, first, visit, feature, row)
                        else:
                            lo, hi = float(row["ci_lo_"+suffix]), float(row["ci_hi_"+suffix])
                        score_at_visit = observed.set_index("id").scores.map(lambda scores: scores[feature])
                        if mode != "cross":
                            baseline = aggregates[aggregates.visit == VISITS[0]].set_index("id").scores
                            score_at_visit -= baseline.loc[score_at_visit.index].map(lambda scores: scores[feature])
                        by_arm = observed.set_index("id").arm
                        first_values = score_at_visit[by_arm == first]
                        other_values = score_at_visit[by_arm == second]
                        estimate = float(row["beta_"+suffix])
                        expected = first_values.mean() if mode == "paired" else first_values.mean() - other_values.mean()
                        np.testing.assert_allclose(estimate, expected, atol=1e-9)
                        rows.append({
                            "feature": feature, "estimate": estimate, "lo": lo, "hi": hi,
                            "p": float(row["p_"+suffix]), "q": float(row["FDR_"+suffix]),
                            "n": int(row.n_units), "nFirst": len(first_values), "nOther": len(other_values),
                        })
                    tables["|".join((mode, first, visit, population))] = rows
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("fixture.json"))
    args = parser.parse_args()
    if version("sctrial") != "1.1.0":
        raise SystemExit(f"Install sctrial at {COMMIT} before building the fixture.")
    data, people = study()
    obs = observation_summaries(data)
    result = fitted_tables(data, obs)
    assert len(result) == 42
    payload = {
        "study": {
            "name": "Longitudinal immune profiling", "source": "Synthetic study",
            "unit": "Mean module score", "seed": SEED, "cells": data.n_obs, "commit": COMMIT,
            "package": "sctrial 1.1.0", "interval": "95% analytical confidence interval", "methods": SOURCES,
            "multiplicity": "Benjamini–Hochberg adjustment across four scores for the selected contrast and population.",
            "pooled": "All annotated cells pools CD8 T cells, B cells, and CD4 T cells.",
            "smallSample": "Small synthetic cohort. Analytical intervals and p-values are illustrative.",
            "versions": {k: version(k) for k in ("anndata","numpy","pandas","scipy","statsmodels")},
        },
        "participants": people, "visits": VISITS, "populations": POPULATIONS,
        "cellTypes": TYPES, "features": [{"id": key, "label": label} for key, label in FEATURES.items()],
        "observations": obs, "results": result,
    }
    args.output.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
