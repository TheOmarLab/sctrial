"""Rebuild the synthetic gene observations and fitted reference contrasts.

Run this file in a checkout with the project's statistical dependencies installed.
The input and output paths refer to the packaged interactive example. Nothing is
downloaded. Generated cell expression exists only while the fixture is built.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from sctrial.interactive._fit import fit_scores

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "src" / "sctrial" / "interactive" / "assets"
SEED = 325011
GENE_UNIT = "Simulated expression (arbitrary units)"
SET_UNIT = "Mean simulated expression (arbitrary units)"
POOLED = "All annotated cells"
CD4 = "CD4 T cells"
# Relative means vary by gene and cell type. They describe the simulation.
GENES = {
    "IFIT1": ("interferon", 1.10),
    "ISG15": ("interferon", 1.55),
    "MX1": ("interferon", 0.89),
    "PRF1": ("cytotoxic", 0.74),
    "NKG7": ("cytotoxic", 1.39),
    "GZMB": ("cytotoxic", 1.02),
}
SETS = {
    "interferon": ("Interferon example set", ["IFIT1", "ISG15", "MX1"]),
    "cytotoxic": ("Cytotoxic example set", ["PRF1", "NKG7", "GZMB"]),
}
CELL_MEANS = {
    "interferon": {"CD8 T cells": 1.15, "B cells": 1.55, "CD4 T cells": 1.23},
    "cytotoxic": {"CD8 T cells": 2.78, "B cells": 0.12, "CD4 T cells": 0.36},
}
SLOPES = {
    "interferon": {"Arm A": 0.29, "Arm B": 0.13},
    "cytotoxic": {"Arm A": -0.12, "Arm B": 0.08},
}


def generator(*key: str) -> np.random.Generator:
    """Use independent stable streams for each participant, gene and sample."""
    digest = hashlib.sha256(json.dumps([SEED, *key]).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def features() -> list[dict]:
    genes = [
        {
            "id": "gene_" + symbol,
            "label": symbol,
            "kind": "gene",
            "unit": GENE_UNIT,
            "description": "Mean simulated expression per cell in this participant, visit and population.",
        }
        for symbol in GENES
    ]
    sets = [
        {
            "id": "set_" + family,
            "label": label,
            "kind": "pathway",
            "unit": SET_UNIT,
            "members": members,
            "description": "Illustrative gene set. The score is the unweighted arithmetic mean of "
            + ", ".join(members)
            + " in the same cells.",
        }
        for family, (label, members) in SETS.items()
    ]
    return genes + sets


def gene_totals(row: dict, time: float) -> np.ndarray:
    """Simulate each annotated cell once and retain its population gene totals."""
    person, arm, visit, population = (row[k] for k in ("id", "arm", "visit", "population"))
    totals = []
    for symbol, (family, weight) in GENES.items():
        donor = generator("donor", person, family).normal(0, 0.26)
        slope = SLOPES[family][arm] + generator("slope", person, family).normal(0, 0.16)
        sample = generator("sample", person, visit, population, family).normal(0, 0.15)
        mean = CELL_MEANS[family][population] * weight * math.exp(donor + time * slope + sample)
        rng = generator("cell", person, visit, population, symbol)
        # A two-part distribution on a nonnegative illustrative expression scale.
        detected = rng.random(row["cells"]) >= 0.24
        signal = rng.gamma(shape=2.2, scale=mean / (2.2 * 0.76), size=row["cells"])
        totals.append(float(np.sum(signal[detected])))
    return np.array(totals)


def summarize(study: dict) -> tuple[list[dict], pd.DataFrame]:
    times = {"Baseline": 0, "Week 4": 1 / 3, "Week 12": 1}
    per_type = {}
    pooled_totals = defaultdict(lambda: np.zeros(len(GENES)))
    pooled_counts = defaultdict(int)
    for row in study["observations"]:
        if row["population"] == POOLED:
            continue
        sample = (row["id"], row["visit"])
        totals = gene_totals(row, times[row["visit"]])
        per_type[sample + (row["population"],)] = totals
        pooled_totals[sample] += totals
        pooled_counts[sample] += row["cells"]

    observations, summary = [], []
    for row in study["observations"]:
        person, visit, population = (row[k] for k in ("id", "visit", "population"))
        sample = (person, visit)
        if population == POOLED:
            if pooled_counts[sample] != row["cells"]:
                raise ValueError(f"Annotated populations do not cover the pooled sample {sample!r}")
            totals = pooled_totals[sample]
        else:
            totals = per_type[sample + (population,)]
        scores = {
            "gene_" + symbol: float(total / row["cells"]) for symbol, total in zip(GENES, totals)
        }
        for family, (_, members) in SETS.items():
            scores["set_" + family] = math.fsum(scores["gene_" + g] for g in members) / len(members)
        observations.append(
            {"id": person, "visit": visit, "population": population, "scores": scores}
        )
        summary.append(
            {
                "participant": person,
                "visit": visit,
                "population": population,
                "arm": row["arm"],
                **scores,
            }
        )
    return observations, pd.DataFrame(summary)


def cd4_measurements(study: dict) -> list[dict]:
    """Check all CD4 cells and original module means against the pooled complement."""
    by_sample = defaultdict(dict)
    for row in study["observations"]:
        by_sample[(row["id"], row["visit"])][row["population"]] = row
    rows = []
    for key, populations in by_sample.items():
        pooled = populations[POOLED]
        others = [r for pop, r in populations.items() if pop not in (POOLED, CD4)]
        count = pooled["cells"] - sum(r["cells"] for r in others)
        cd4 = populations.get(CD4)
        if count == 0 and cd4 is None:
            continue
        if count <= 0 or cd4 is None or count != cd4["cells"]:
            raise ValueError(f"Inconsistent CD4 coverage at {key!r}")
        for feature in study["features"]:
            name = feature["id"]
            residual_sum = pooled["cells"] * pooled["scores"][name] - math.fsum(
                r["cells"] * r["scores"][name] for r in others
            )
            if not math.isclose(
                residual_sum / count, cd4["scores"][name], rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError(f"Inconsistent pooled module score for {name} at {key!r}")
        rows.append(cd4)
    return rows


def fit_reference(
    study: dict, feature_list: list[dict], summary: pd.DataFrame, keys: list[str], unit: str
) -> tuple[dict, dict]:
    config = {
        **study,
        "studyId": "synthetic-molecular-example",
        "features": feature_list,
        "unit": unit,
    }
    arms = list(dict.fromkeys(person["arm"] for person in study["participants"]))
    results = {}
    warnings = {}
    for key in keys:
        design, arm, visit, population = key.split("|")
        other = next(group for group in arms if group != arm)
        request = {
            "design": design,
            "arm": arm,
            "reference": other,
            "baseline": study["visits"][0],
            "visit": visit,
            "population": population,
        }
        fitted = fit_scores(config, summary, request)
        if fitted["warnings"]:
            warnings[key] = fitted["warnings"]
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
        results[key] = [
            {
                "feature": row["id"],
                "kind": row.get("kind", "module"),
                **{field: row[field] for field in fields},
            }
            for row in fitted["rows"]
        ]
    return results, warnings


def build() -> dict:
    raw = (ASSETS / "example.json").read_bytes()
    study = json.loads(raw)
    outcomes = features()
    observations, summary = summarize(study)
    cd4 = cd4_measurements(study)
    original_keys = list(study["results"])
    cd4_keys = [
        key.rsplit("|", 1)[0] + "|" + CD4 for key in original_keys if key.endswith("|" + POOLED)
    ]
    config = {**study, "populations": [*study["populations"], CD4]}
    fits, notices = fit_reference(config, outcomes, summary, [*original_keys, *cd4_keys], GENE_UNIT)
    module_summary = pd.DataFrame(
        [
            {
                "participant": row["id"],
                "arm": row["arm"],
                "visit": row["visit"],
                "population": CD4,
                **row["scores"],
            }
            for row in cd4
        ]
    )
    module_fits, module_notices = fit_reference(
        config, study["features"], module_summary, cd4_keys, study["study"]["unit"]
    )
    for key in cd4_keys:
        fits[key] = [*module_fits[key], *fits[key]]
        combined = list(dict.fromkeys([*module_notices.get(key, []), *notices.get(key, [])]))
        if combined:
            notices[key] = combined
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_version = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)[1]
    return {
        "features": outcomes,
        "observations": observations,
        "additionalPopulations": [CD4],
        "results": fits,
        "resultWarnings": notices,
        "provenance": {
            "source": "Synthetic gene expression on the bundled participant, arm, visit and annotated-cell sampling structure.",
            "seed": SEED,
            "package": "sctrial " + package_version,
            "baseSha256": hashlib.sha256(raw).hexdigest(),
            "aggregation": "Gene values are cell means. All annotated cells is a cell-count-weighted mean over CD8 T cells, B cells and CD4 T cells.",
            "sets": "Hand-selected example gene sets. Each set score is the arithmetic mean of its three simulated gene values. These scores describe the expression of the listed genes.",
            "fdrFamily": "Benjamini–Hochberg adjustment across six simulated genes and two illustrative sets for each contrast and population.",
            "fit": "sctrial participant-level score models on participant-visit population means, with standardize=False.",
            "versions": {
                name: version(name)
                for name in ("anndata", "numpy", "pandas", "scipy", "statsmodels")
            },
        },
    }


if __name__ == "__main__":
    output = json.dumps(build(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (ASSETS / "molecular-demo.json").write_text(output + "\n", encoding="utf-8")
