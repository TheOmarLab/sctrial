"""Aggregate selected observation scores and expression columns at the participant-visit level."""

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
    from anndata.experimental import read_elem  # type: ignore[no-redef]

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
CELL_BINS = 64


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _numeric(node: h5py.Dataset | h5py.Group) -> bool:
    if isinstance(node, h5py.Dataset):
        return node.dtype.kind in "iuf"
    return _text(node.attrs.get("encoding-type", "")) in {"nullable-integer", "nullable-float"}


def _meta(name: str, kind: str | None = None) -> dict:
    prefix, separator, suffix = name.partition(":")
    typed = separator and prefix.lower() in {"gene", "pathway", "module"}
    label = suffix.strip() if typed else name
    if not label:
        raise ValueError(f"Add a name after the molecular outcome prefix in '{name}'.")
    kind = kind or (prefix.lower() if typed else "score")
    units = {
        "gene": "Recorded expression",
        "module": "Module score",
        "pathway": "Pathway score",
        "score": "Recorded score",
    }
    return {"kind": kind, "label": label, "unit": units[kind]}


def _gene_values(
    handle: h5py.File, names: list[str], var_names: tuple[str, ...], cells: int
) -> np.ndarray:
    """Extract requested X columns. Sparse storage zeros remain genuine zeros."""
    if "X" not in handle or "var" not in handle:
        raise ValueError("This file has no X matrix with named variables.")
    requested = set(names)
    lookup: dict[str, int] = {}
    for i, name in enumerate(var_names):
        if name in requested:
            if name in lookup:
                raise ValueError(f"'{name}' occurs twice in var_names. Use unique variable names.")
            lookup[name] = i
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError("Unknown X variable (match var_names exactly): " + ", ".join(missing))
    if cells * len(names) > 12_000_000:
        raise ValueError("Select fewer genes or subset the study (12 million cell–gene values).")
    x = handle["X"]
    shape = x.shape if isinstance(x, h5py.Dataset) else tuple(x.attrs.get("shape", ()))
    if shape != (cells, len(var_names)):
        raise ValueError("The X matrix dimensions do not match obs and var.")
    result = np.zeros((cells, len(names)), dtype=float)
    indexes = [lookup[name] for name in names]
    if isinstance(x, h5py.Dataset):
        if x.dtype.kind not in "iuf":
            raise ValueError("X must store real numeric expression values.")
        for out, col in enumerate(indexes):
            for start in range(0, cells, 65536):
                end = min(start + 65536, cells)
                result[start:end, out] = x[start:end, col]
        return result
    encoding = _text(x.attrs.get("encoding-type", ""))
    if encoding not in {"csr_matrix", "csc_matrix"} or x["data"].dtype.kind not in "iuf":
        raise ValueError("Use a numeric dense, CSR or CSC expression matrix in X.")
    data, indices, pointers = x["data"], x["indices"], x["indptr"]
    if encoding == "csc_matrix":
        for out, col in enumerate(indexes):
            start, stop = pointers[col : col + 2]
            for left in range(int(start), int(stop), 65536):
                right = min(left + 65536, int(stop))
                np.add.at(result[:, out], indices[left:right], data[left:right])
        return result
    # CSR stores rows contiguously. Scan column indices and retrieve only matched values.
    ptr = pointers[:]
    order = np.argsort(indexes)
    columns = np.array(indexes)[order]
    for left in range(0, len(indices), 65536):
        block = indices[left : left + 65536]
        positions = np.searchsorted(columns, block).clip(max=len(columns) - 1)
        matched = np.flatnonzero(columns[positions] == block)
        for start in range(0, len(matched), 512):
            offsets = matched[start : start + 512]
            absolute = left + offsets
            rows = np.searchsorted(ptr, absolute, side="right") - 1
            # Contiguous runs avoid HDF5's relatively expensive point selections.
            if absolute[-1] - absolute[0] + 1 == len(absolute):
                values = data[absolute[0] : absolute[-1] + 1]
            else:
                values = data[absolute]
            np.add.at(result, (rows, order[positions[offsets]]), values)
    return result


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
        gene_input = "var" in handle and "X" in handle
    by_lower = {name.lower(): name for name in names}
    suggested: dict[str, Any] = {
        role: next((by_lower[v] for v in ALIASES[role] if v in by_lower), "") for role in ROLES
    }
    suggested["scores"] = [
        col["name"]
        for col in columns
        if col["numeric"]
        and any(
            word in col["name"].lower()
            for word in ("score", "signature", "module", "gene:", "pathway:")
        )
        and col["name"] not in suggested.values()
    ][:MAX_SCORES]
    return {
        "filename": path.name,
        "nCells": count,
        "columns": columns,
        "suggested": suggested,
        "geneInput": gene_input,
    }


def _mapping(schema: dict, request: dict) -> tuple[dict, list[str], list[str], dict]:
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
    scores = request.get("scores", [])
    genes = request.get("genes", [])
    if not isinstance(genes, list) or any(not isinstance(g, str) or not g.strip() for g in genes):
        raise ValueError("Enter X variable names as a list of nonempty strings.")
    if not isinstance(scores, list) or not 1 <= len(scores) + len(genes) <= MAX_SCORES:
        raise ValueError(
            f"Choose between 1 and {MAX_SCORES} observation scores and genes in total."
        )
    if len(genes) != len(set(genes)):
        raise ValueError("Select each X variable once.")
    kinds = request.get("scoreKinds", {})
    if not isinstance(kinds, dict) or any(
        key not in scores or kind not in ("gene", "pathway", "module", "score")
        for key, kind in kinds.items()
    ):
        raise ValueError("Assign measurement types only to selected observation scores.")
    if any(not isinstance(name, str) or name not in numeric or name in chosen for name in scores):
        raise ValueError(
            "Each outcome must be a numeric observation column separate from the study fields."
        )
    if len(scores) != len(set(scores)):
        raise ValueError("Select each score once.")
    return roles, scores, genes, kinds


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


def cell_distributions(frame: pd.DataFrame, scores: list[str]) -> dict:
    """Count measured cells on a shared scale without retaining cell records or coordinates."""
    keys = ["arm", "visit", "population"]
    grouped = frame.groupby(keys, sort=False, observed=True)
    group_ids = grouped.ngroup().to_numpy(dtype=np.intp)
    groups = [
        {**dict(zip(keys, labels)), "bins": {}, "zeros": {}}
        for labels in grouped.size().index
    ]
    features = {}
    cuts = np.arange(1, CELL_BINS) / CELL_BINS
    for score in scores:
        raw = frame[score].to_numpy(dtype=float)
        measured = np.isfinite(raw)
        if not measured.any():
            continue
        values = raw[measured]
        low, high = float(values.min()), float(values.max())
        features[score] = [low, high]
        # Interpolation avoids overflowing the span for large opposite-sign values.
        edges = low * (1 - cuts) + high * cuts
        index = np.searchsorted(edges, values, side="right") if low != high else 0
        members = group_ids[measured]
        counts = np.bincount(members * CELL_BINS + index, minlength=len(groups) * CELL_BINS)
        zeros = np.bincount(members[values == 0], minlength=len(groups))
        for i, histogram in enumerate(counts.reshape(-1, CELL_BINS)):
            if histogram.any():
                groups[i]["bins"][score] = histogram.tolist()
                groups[i]["zeros"][score] = int(zeros[i])
    return {"features": features, "groups": groups}


class LocalStudy:
    """Keep source data read-only and serve the last explicitly confirmed mapping."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve(strict=True)
        self._stamp = self._file_stamp()
        self._schema = _schema(self.path)
        self._lock = RLock()
        self._state: tuple[dict, pd.DataFrame] | None = None
        self._mapping_key: tuple | None = None
        self._variables: tuple[str, ...] | None = None

    def _file_stamp(self) -> tuple[int, int, int]:
        stat = self.path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def schema(self) -> dict:
        return self._schema

    def _var_names(self, handle: h5py.File) -> tuple[str, ...]:
        if self._variables is None:
            var = handle["var"]
            labels = np.asarray(read_elem(var[_text(var.attrs["_index"])]))
            if labels.ndim != 1:
                raise ValueError("Expected a one-dimensional var_names index.")
            self._variables = tuple(_text(label) for label in labels)
        return self._variables

    def find_genes(self, request: dict) -> dict:
        query = request.get("query")
        if not isinstance(query, str) or not 2 <= len(query.strip()) <= 160:
            raise ValueError("Type 2 to 160 characters of a variable name.")
        query = query.strip().casefold()
        with self._lock:
            if self._file_stamp() != self._stamp:
                raise ValueError("The source file changed. Restart the workspace to reopen it.")
            if not self._schema["geneInput"]:
                return {"genes": []}
            with h5py.File(self.path, "r") as handle:
                variables = self._var_names(handle)
            matches = []
            for name in variables:
                if name.casefold().startswith(query):
                    matches.append(name)
                    if len(matches) == 12:
                        break
            return {"genes": matches}

    def prepare(self, request: dict) -> dict:
        roles, names, genes, kinds = _mapping(self._schema, request)
        key = (*roles.values(), tuple(names), tuple(genes), tuple(sorted(kinds.items())))
        with self._lock:
            if self._file_stamp() != self._stamp:
                raise ValueError("The source file changed. Restart the workspace to reopen it.")
            if self._state is not None and key == self._mapping_key:
                return self._state[0]
            study_id = secrets.token_urlsafe(16)
            features = [
                {
                    "id": f"score_{i + 1}",
                    "inputName": name,
                    "inputSource": "obs",
                    **_meta(name, kinds.get(name)),
                }
                for i, name in enumerate(names)
            ]
            features.extend(
                {
                    "id": f"gene_{i + 1}",
                    "kind": "gene",
                    "inputName": name,
                    "inputSource": "X",
                    "label": name,
                    "unit": "Expression (X scale)",
                    "description": "Mean of stored X values for each participant, visit, and population.",
                }
                for i, name in enumerate(genes)
            )
            display_names = [f["label"] for f in features]
            for feature in features:
                if display_names.count(feature["label"]) > 1:
                    source = (
                        "X" if feature["inputSource"] == "X" else "obs: " + feature["inputName"]
                    )
                    feature["label"] += f" [{source}]"
            score_ids = [f["id"] for f in features]
            data: dict[str, Any] = {}
            defaults = {"arm": "All participants", "visit": "Collection", "population": "All cells"}
            with h5py.File(self.path, "r") as handle:
                obs = handle["obs"]
                for role, name in roles.items():
                    data[role] = _labels(read_elem(obs[name]), name) if name else defaults[role]
                for name, feature in zip(names, features):
                    values = pd.Series(read_elem(obs[name])).to_numpy(dtype=float, na_value=np.nan)
                    if np.isinf(values).any():
                        raise ValueError(
                            f"'{name}' contains infinite scores. Use finite values or missing observations."
                        )
                    data[feature["id"]] = values
                if genes:
                    if not self._schema["geneInput"]:
                        raise ValueError("This file has no X matrix with named variables.")
                    expression = _gene_values(
                        handle, genes, self._var_names(handle), self._schema["nCells"]
                    )
                    for i, feature in enumerate(features[len(names) :]):
                        values = expression[:, i]
                        if np.isinf(values).any():
                            raise ValueError(f"'{genes[i]}' contains infinite expression values.")
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
            visit_column = frame["visit"]
            ordered_visits = isinstance(visit_column.dtype, pd.CategoricalDtype) and (
                visit_column.cat.ordered
            )
            visits = (
                visit_column.cat.remove_unused_categories().cat.categories.tolist()
                if ordered_visits
                else visit_column.drop_duplicates().tolist()
            )
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
                    **({"genes": list(genes)} if genes else {}),
                    **({"scoreKinds": kinds} if kinds else {}),
                },
                "unit": "Recorded score",
                "participants": people.rename(columns={"participant": "id"}).to_dict("records"),
                "visits": visits,
                "orderedVisits": ordered_visits,
                "populations": [pooled, *types],
                "features": features,
                "observations": rows,
                "cellDistributions": cell_distributions(frame, score_ids),
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
