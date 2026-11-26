"""Read selected observation columns and aggregate at the participant-visit level."""

from __future__ import annotations

import secrets
from pathlib import Path
from threading import RLock
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sctrial import __version__

try:
    from anndata.io import read_elem
except ImportError:  # AnnData 0.9 and 0.10
    from anndata.experimental import read_elem

from ._fit import fit_scores

ROLES = ("participant", "arm", "visit", "population")
ALIASES = {
    "participant": (
        "participant_id",
        "patient_id",
        "donor_id",
        "subject_id",
        "participant",
        "patient",
        "donor",
        "subject",
    ),
    "arm": ("arm", "treatment_arm", "group", "condition"),
    "visit": ("visit", "visit_id", "timepoint", "time_point", "collection_time"),
    "population": ("cell_type", "celltype", "cell_type_label", "annotation", "population"),
}
MAX_SCORES = 12
MAX_SUMMARIES = 50000


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _numeric(node: h5py.Dataset | h5py.Group) -> bool:
    if isinstance(node, h5py.Dataset):
        return node.dtype.kind in "iuf"
    return _text(node.attrs.get("encoding-type", "")) in {"nullable-integer", "nullable-float"}


def _schema(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        obs = handle.get("obs")
        if not isinstance(obs, h5py.Group) or "_index" not in obs.attrs:
            raise ValueError(
                "Expected an H5AD observation table. Save the study with AnnData.write_h5ad()."
            )
        names = [_text(value) for value in obs.attrs.get("column-order", [])]
        columns: list[dict[str, Any]] = [
            {"name": name, "numeric": _numeric(obs[name])} for name in names
        ]
        index = obs[_text(obs.attrs["_index"])]
        if isinstance(index, h5py.Group):
            index = index["codes"] if "codes" in index else index["values"]
        count = len(index)
    by_lower = {name.lower(): name for name in names}
    suggested: dict[str, Any] = {
        role: next((by_lower[v] for v in ALIASES[role] if v in by_lower), "") for role in ROLES
    }
    suggested["scores"] = [
        col["name"]
        for col in columns
        if col["numeric"]
        and any(word in col["name"].lower() for word in ("score", "signature", "module"))
        and col["name"] not in suggested.values()
    ][:MAX_SCORES]
    return {"filename": path.name, "nCells": count, "columns": columns, "suggested": suggested}


def _mapping(schema: dict, request: dict) -> tuple[dict, list[str]]:
    roles = {role: request.get(role) or "" for role in ROLES}
    columns = {entry["name"] for entry in schema["columns"]}
    numeric = {entry["name"] for entry in schema["columns"] if entry["numeric"]}
    if not roles["participant"]:
        raise ValueError("Select a participant ID column.")
    chosen = [value for value in roles.values() if value]
    if any(not isinstance(name, str) or name not in columns for name in chosen):
        raise ValueError("Select study fields from the observation table.")
    if len(chosen) != len(set(chosen)):
        raise ValueError("Assign a different observation column to each study field.")
    scores = request.get("scores")
    if not isinstance(scores, list) or not 1 <= len(scores) <= MAX_SCORES:
        raise ValueError(f"Choose between 1 and {MAX_SCORES} numeric observation scores.")
    if any(not isinstance(name, str) or name not in numeric or name in chosen for name in scores):
        raise ValueError(
            "Each outcome must be a numeric observation column separate from the study fields."
        )
    if len(scores) != len(set(scores)):
        raise ValueError("Select each score once.")
    return roles, scores


def _labels(values: Any, name: str) -> pd.Categorical:
    labels = pd.Series(values)
    if labels.isna().any():
        raise ValueError(
            f"'{name}' contains missing study labels. Complete these labels before mapping."
        )
    labels = labels.astype("category")
    categories = pd.Index([_text(v) for v in labels.cat.categories])
    if not categories.is_unique or any(not v.strip() for v in categories):
        raise ValueError(f"'{name}' contains blank or ambiguous study labels.")
    return labels.cat.rename_categories(categories).array


def _summarize(
    frame: pd.DataFrame, scores: list[str], pooled: str, has_types: bool
) -> pd.DataFrame:
    keys = ["participant", "arm", "visit"]
    specifications = {
        "cells": ("participant", "size"),
        **{score: (score, "mean") for score in scores},
    }
    total = frame.groupby(keys, sort=False, observed=True).agg(**specifications).reset_index()
    if len(total) > MAX_SUMMARIES:
        raise ValueError("The workspace supports up to 50,000 participant-visit summaries.")
    total["population"] = pooled
    if not has_types:
        return total
    groups = frame.groupby([*keys, "population"], sort=False, observed=True)
    if len(total) + groups.ngroups > MAX_SUMMARIES:
        raise ValueError(
            "This mapping exceeds 50,000 summaries. Clear the population field or select a study subset."
        )
    by_population = groups.agg(**specifications).reset_index()
    return pd.concat([total, by_population], ignore_index=True)


class LocalStudy:
    """Keep source data read-only and serve the last explicitly confirmed mapping."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve(strict=True)
        self._stamp = self._file_stamp()
        self._schema = _schema(self.path)
        self._lock = RLock()
        self._state: tuple[dict, pd.DataFrame] | None = None
        self._mapping_key: tuple | None = None

    def _file_stamp(self) -> tuple[int, int, int]:
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def schema(self) -> dict:
        return self._schema

    def prepare(self, request: dict) -> dict:
        roles, names = _mapping(self._schema, request)
        key = (*roles.values(), *names)
        with self._lock:
            if self._file_stamp() != self._stamp:
                raise ValueError("The source file changed. Restart the workspace to reopen it.")
            if self._state is not None and key == self._mapping_key:
                return self._state[0]
            study_id = secrets.token_urlsafe(16)
            features = [{"id": f"score_{i + 1}", "label": name} for i, name in enumerate(names)]
            score_ids = [f["id"] for f in features]
            data: dict[str, Any] = {}
            defaults = {"arm": "All participants", "visit": "Collection", "population": "All cells"}
            with h5py.File(self.path, "r") as handle:
                obs = handle["obs"]
                for role, name in roles.items():
                    data[role] = _labels(read_elem(obs[name]), name) if name else defaults[role]
                for feature in features:
                    name = feature["label"]
                    values = pd.Series(read_elem(obs[name])).to_numpy(dtype=float, na_value=np.nan)
                    if np.isinf(values).any():
                        raise ValueError(
                            f"'{name}' contains infinite scores. Use finite values or missing observations."
                        )
                    data[feature["id"]] = values
            frame = pd.DataFrame(data)
            if frame.empty:
                raise ValueError("This study contains no cells.")
            people = frame[["participant", "arm"]].drop_duplicates()
            if people.participant.duplicated().any():
                raise ValueError(
                    "A participant maps to more than one group. Use a fixed group assignment."
                )
            if people.arm.nunique() > 2:
                raise ValueError("Select one or two groups in the input study for this workspace.")
            visits = frame.visit.drop_duplicates().tolist()
            types = frame.population.drop_duplicates().tolist() if roles["population"] else []
            pooled = "All cells"
            while pooled in types:
                pooled += " (pooled)"
            summary = _summarize(frame, score_ids, pooled, bool(types))
            rows = []
            for record in summary.itertuples(index=False):
                rows.append(
                    {
                        "id": record.participant,
                        "arm": record.arm,
                        "visit": record.visit,
                        "population": record.population,
                        "cells": int(record.cells),
                        "scores": {
                            score: None
                            if pd.isna(value := getattr(record, score))
                            else float(value)
                            for score in score_ids
                        },
                    }
                )
            payload = {
                "studyId": study_id,
                "title": self.path.name,
                "source": "h5ad",
                "version": __version__,
                "mapping": {
                    **{role: name or None for role, name in roles.items()},
                    "scores": list(names),
                },
                "unit": "Recorded score",
                "participants": people.rename(columns={"participant": "id"}).to_dict("records"),
                "visits": visits,
                "populations": [pooled, *types],
                "features": features,
                "observations": rows,
                "results": {},
            }
            if self._file_stamp() != self._stamp:
                raise ValueError("The source file changed during mapping. Restart the workspace.")
            self._state, self._mapping_key = (payload, summary), key
            return payload

    def fit(self, request: dict) -> dict:
        with self._lock:
            if self._state is None or request.get("studyId") != self._state[0]["studyId"]:
                raise ValueError("This study mapping is no longer current. Reopen the local study.")
            payload, summary = self._state
            return fit_scores(payload, summary, request)
