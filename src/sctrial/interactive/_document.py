"""Assemble a self-contained workspace for both the local server and offline use."""

from __future__ import annotations

import json
from importlib.resources import files

_STYLES = (
    "font",
    "dialog",
    "cohort",
    "results",
    "pins",
    "canvas",
    "mapping",
    "studio",
    "sheets",
    "editor",
    "evidence",
    "matrix",
    "app-menu",
    "atlas",
)
_SCRIPTS = (
    "csv",
    "cohort",
    "results",
    "pins",
    "canvas",
    "mapping",
    "chrome",
    "sheets",
    "matrix",
    "evidence",
    "app-menu",
    "cell-art",
    "gene-sets",
    "atlas-data",
    "atlas-scene",
    "atlas",
    "studio",
)


def build_workspace() -> bytes:
    """Return the offline document with its data, scripts, styles and font embedded."""
    assets = files("sctrial.interactive").joinpath("assets")

    def read(name: str) -> str:
        return assets.joinpath(name).read_text(encoding="utf-8")

    sample = json.loads(read("example.json"))
    molecular = json.loads(read("molecular-demo.json"))
    for feature in sample["features"]:
        feature.update(kind="module", unit="Mean module score")
    sample["features"].extend(molecular["features"])
    sample["populations"].extend(molecular.get("additionalPopulations", []))
    # The molecular example uses the original study's participant, visit, and population keys.
    scores = {
        (row["id"], row["visit"], row["population"]): row["scores"]
        for row in molecular["observations"]
    }
    for row in sample["observations"]:
        row["scores"].update(scores[(row["id"], row["visit"], row["population"])])
    for contrast, rows in molecular["results"].items():
        sample["results"].setdefault(contrast, []).extend(rows)
    sample["resultWarnings"] = molecular["resultWarnings"]
    sample["study"]["molecular"] = {
        **molecular["provenance"],
        "additionalPopulations": molecular["additionalPopulations"],
    }
    fixture = json.dumps(sample, separators=(",", ":"), ensure_ascii=False)

    style = "\n".join(read(name + ".css") for name in _STYLES)
    scripts = "\n".join(read(name + ".js") for name in _SCRIPTS)
    return (
        read("shell.html")
        .replace("/*STYLE*/", style)
        .replace("/*SCRIPT*/", scripts)
        .replace("/*FIXTURE*/", fixture)
        .encode("utf-8")
    )
