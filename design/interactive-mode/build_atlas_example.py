"""Rebuild the peripheral immune atlas from seeded synthetic single-cell counts.

Run from a checkout with the sctrial dependencies installed. The output contains
participant summaries, binned cell distributions and fitted contrasts. All gene
sets and changes between visits are illustrative simulation parameters.
"""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from sctrial.interactive._fit import METHODS, fit_scores
from sctrial.interactive._study import _summarize, cell_distributions

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "src/sctrial/interactive/assets/atlas-example.json"
SEED = 81027
ARMS = ("Arm A", "Arm B")
VISITS = ("Baseline", "Week 4", "Week 12")
POOLED = "All annotated cells"
TYPES = ("CD8 T cells", "CD4 T cells", "Monocytes", "B cells", "NK cells")
GENES = ("GZMB", "IFIT1", "NKG7", "ISG15")
SET_MEMBERS = {"interferon": ("IFIT1", "ISG15"), "cytotoxic": ("GZMB", "NKG7")}
OUTCOME_IDS = ("interferon", "cytotoxicity")
# Baseline count simulation parameters, with one row per type and one column per gene.
MEANS = np.array(
    [
        [2.6, 0.28, 3.2, 0.64],
        [0.06, 0.32, 0.16, 0.78],
        [0.02, 1.52, 0.10, 2.00],
        [0.04, 0.61, 0.05, 1.30],
        [6.30, 0.42, 7.00, 0.87],
    ]
)
UNMEASURED_VISITS = {("A05", "Week 4"), ("A02", "Week 12"), ("B06", "Week 12")}


def features() -> list[dict]:
    outcomes = []
    for key, label in zip(OUTCOME_IDS, ("Interferon response", "Cytotoxicity")):
        outcomes.append(
            {
                "id": key,
                "label": label,
                "kind": "module",
                "unit": "Control-adjusted score",
                "description": "Per-cell mean log1p expression of the two example set members "
                "minus the mean log1p expression of three simulated control features.",
            }
        )
    outcomes.extend(
        {
            "id": "gene_" + gene,
            "label": gene,
            "kind": "gene",
            "unit": "log1p simulated counts",
            "description": "Natural logarithm of one plus the simulated gene count in each cell.",
        }
        for gene in GENES
    )
    for key, members in SET_MEMBERS.items():
        outcomes.append(
            {
                "id": "set_" + key,
                "label": key.title() + " example set",
                "kind": "pathway",
                "members": list(members),
                "unit": "Mean log1p simulated counts",
                "description": "Illustrative gene set. Per-cell arithmetic mean of log1p "
                + " and ".join(members)
                + " expression.",
            }
        )
    return outcomes


def simulate() -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(SEED)
    blocks = []
    people = []
    # Observed abundance follows a multinomial draw with donor- and visit-specific composition.
    baseline = np.array([0.26, 0.45, 0.15, 0.09, 0.05])
    for arm in ARMS:
        for donor in range(1, 7):
            person = arm[-1] + f"{donor:02}"
            people.append({"id": person, "arm": arm})
            donor_composition = baseline * np.exp(rng.normal(0, 0.28, len(TYPES)))
            donor_levels = rng.normal(0, 0.38, (len(TYPES), len(GENES)))
            for time, visit in enumerate(VISITS):
                if (person, visit) in UNMEASURED_VISITS:
                    continue
                shift = (
                    [0.32, -0.33, 0.08, -0.12, 0.34]
                    if arm == "Arm A"
                    else [-0.06, 0.02, 0.08, 0.06, -0.06]
                )
                weights = donor_composition * np.exp(time * np.array(shift))
                sample_size = int(rng.integers(330, 810))
                counts = rng.multinomial(sample_size, weights / weights.sum())
                for i, (population, n) in enumerate(zip(TYPES, counts)):
                    if not n:
                        continue
                    # A cell-specific latent factor gives each set's two members shared variability.
                    latent = rng.lognormal(0, 0.68, (n, 2))
                    sample_effect = rng.normal(0, 0.19, len(GENES))
                    time_effect = np.array([0.09, 0.28, 0.08, 0.23]) if arm == "Arm A" else 0.04
                    rate = MEANS[i] * np.exp(donor_levels[i] + sample_effect + time * time_effect)
                    rates = latent[:, [0, 1, 0, 1]] * rate
                    expression = np.log1p(rng.negative_binomial(2, 2 / (2 + rates)))
                    control_mean = np.log1p(rng.poisson(1.15, (n, 3))).mean(axis=1)
                    block = pd.DataFrame(expression, columns=["gene_" + g for g in GENES])
                    for (key, members), module_id in zip(SET_MEMBERS.items(), OUTCOME_IDS):
                        score = block[["gene_" + g for g in members]].mean(axis=1)
                        block["set_" + key] = score
                        block[module_id] = score - control_mean
                    blocks.append(
                        block.assign(
                            participant=person, arm=arm, visit=visit, population=population
                        )
                    )
    return pd.concat(blocks, ignore_index=True), people


def build() -> dict:
    cells, people = simulate()
    outcomes = features()
    ids = [f["id"] for f in outcomes]
    summary = _summarize(cells, ids, POOLED, True)
    package_version = re.search(
        r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE
    )[1]
    study = {
        "studyId": "peripheral-immune-atlas-v1",
        "unit": "Recorded measurement",
        "study": {
            "name": "Peripheral immune atlas",
            "source": "Synthetic study",
            "seed": SEED,
            "unit": "Recorded measurement",
            "cells": len(cells),
            "package": "sctrial " + package_version,
            "methods": METHODS,
            "interval": "95% analytical confidence interval",
            "fdrFamily": "Eight gene, pathway and module outcomes",
            "multiplicity": "Benjamini–Hochberg adjustment across all eight outcomes for "
            "each contrast and population.",
            "pooled": "All annotated cells pools the five disjoint simulated populations.",
            "smallSample": "Illustrative synthetic cohort. Cell frequencies, expression and "
            "changes between visits are simulated.",
            "versions": {
                k: version(k) for k in ("anndata", "numpy", "pandas", "scipy", "statsmodels")
            },
        },
        "participants": people,
        "visits": list(VISITS),
        "populations": [POOLED, *TYPES],
        "cellTypes": list(TYPES),
        "features": outcomes,
        "results": {},
        "resultWarnings": {},
        "cellDistributions": cell_distributions(cells, ids),
        "observations": [
            {
                "id": r.participant,
                "arm": r.arm,
                "visit": r.visit,
                "population": r.population,
                "cells": int(r.cells),
                "scores": {key: getattr(r, key) for key in ids},
            }
            for r in summary.itertuples(index=False)
        ],
    }
    for population in study["populations"]:
        for arm in ARMS:
            for design in METHODS:
                for visit in VISITS if design == "cross" else VISITS[1:]:
                    key = "|".join((design, arm, visit, population))
                    request = {
                        "population": population,
                        "arm": arm,
                        "design": design,
                        "baseline": VISITS[0],
                        "visit": visit,
                        "reference": next(a for a in ARMS if a != arm),
                    }
                    fit = fit_scores(study, summary, request)
                    if fit["warnings"]:
                        study["resultWarnings"][key] = fit["warnings"]
                    fields = (
                        "estimate",
                        "se",
                        "lo",
                        "hi",
                        "p",
                        "q",
                        "n",
                        "nFirst",
                        "nOther",
                        "covariance",
                        "reason",
                    )
                    study["results"][key] = [
                        {"feature": r["id"], **{field: r[field] for field in fields}}
                        for r in fit["rows"]
                    ]
    return study


if __name__ == "__main__":
    OUTPUT.write_text(
        json.dumps(build(), separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
