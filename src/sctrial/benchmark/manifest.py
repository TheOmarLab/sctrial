"""The immutable identity of a benchmark run.

Every result file carries the hash of the configuration that produced it, and
figure and table generation REFUSES to combine files whose hashes differ.

This exists because of what has already happened here, not as ceremony. A
benchmark was calibrated from a deleted scratch file; a runtime fix landed in the
writer while the figures kept reading a CSV three months older; a resume defect
was fixed in one of two near-identical drivers; a calibration was measured,
documented in the Methods and never threaded into the generator. Every one of
those produced plausible output and none raised an error. A hash that must match
turns "the numbers came from somewhere" into a checkable claim.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

__all__ = [
    "manifest_hash",
    "verify_manifest",
    "assert_single_manifest",
    "source_tree_sha256",
    "SOURCE_TREE_PATHS",
]

# What the benchmark actually executes. Hashed content, not a commit id.
SOURCE_TREE_PATHS = ("src", "scripts", "pyproject.toml")

_REPO = Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO), *args],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - git absent or not a repo
        return "unknown"


def _r_versions() -> dict:
    """Versions of the R packages whose results enter the benchmark."""
    code = (
        'ip <- installed.packages()[, "Version"]; '
        'cat(paste(sapply(c("dreamlet","limma","edgeR","nebula","variancePartition"), '
        'function(p) paste0(p, "=", if (p %in% names(ip)) ip[[p]] else "absent")), '
        'collapse="\\n"))'
    )
    try:
        out = subprocess.run(
            ["Rscript", "-e", code], capture_output=True, text=True, timeout=120, check=True
        ).stdout
        return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    except Exception:
        return {"_error": "Rscript unavailable"}


def manifest_hash(manifest: dict) -> str:
    """Stable hash of a manifest, independent of key order."""
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


# NOTE: there is deliberately NO build_manifest here. The producer lives in
# scripts/calibrate_simulator.py:_build_manifest, which also hashes the eligible
# gene set and the calibration artifacts. A second producer would be exactly the
# duplication this project keeps removing: two implementations drift, and the one
# that drifts is the one nobody is reading.
#
# This module is the CONSUMER side -- verification and the mixed-manifest guard --
# which had no implementation anywhere.


def verify_manifest(manifest: dict, artifacts: dict[str, Path] | None = None) -> None:
    """Raise if the recorded hashes no longer describe what is on disk."""
    recorded = manifest.get("manifest_sha256", manifest.get("config_sha256"))
    if recorded is not None and "manifest_sha256" in manifest:
        if recorded != manifest_hash(manifest):
            raise RuntimeError(
                "manifest_sha256 does not match the manifest contents; it was "
                "edited after being written"
            )
    for name, path in (artifacts or {}).items():
        key = f"artifact_{name}_sha256"
        if key not in manifest:
            continue
        path = Path(path)
        actual = _sha256_file(path) if path.exists() else "missing"
        if actual != manifest[key]:
            raise RuntimeError(
                f"artifact {name} changed since the manifest was written "
                f"({path}). Recorded {manifest[key][:12]}, found {actual[:12]}. "
                "Re-run the benchmark rather than mixing artifacts."
            )


def assert_single_manifest(df: pd.DataFrame, context: str = "") -> str:
    """Refuse a table whose rows came from different runs.

    Combining results across manifests is how a corrected run gets silently
    averaged with the run it was meant to replace. Loaders call this before
    plotting anything.
    """
    if "manifest_sha256" not in df.columns:
        raise ValueError(
            f"{context or 'results'} carry no manifest_sha256. They predate "
            "provenance tracking and cannot be verified; re-run the benchmark."
        )
    seen = sorted(set(df["manifest_sha256"].dropna().astype(str)))
    if len(seen) != 1:
        raise ValueError(
            f"{context or 'results'} mix {len(seen)} manifests: "
            f"{[h[:12] for h in seen]}. These are different benchmark runs and "
            "must not be combined."
        )
    return seen[0]


def source_tree_sha256(repo: Path | None = None, paths=SOURCE_TREE_PATHS) -> str:
    """Deterministic hash of the source that will actually run.

    A commit SHA says what SHOULD be there; this says what IS there. The
    difference is not hypothetical here: the cluster spent this project with its
    HEAD pinned at one commit while rsync had overwritten the files with code many
    commits newer, so the nominal commit described nothing that was executing.

    It also needs no git, which matters because git is absent from this cluster's
    compute nodes -- so a job can verify its own source at run time, which is
    exactly where verification is worth having.

    Sorted relative paths, content-hashed, excluding bytecode and egg-info so the
    hash is stable across installs.
    """
    repo = Path(repo) if repo is not None else _REPO
    h = hashlib.sha256()
    files: list[Path] = []
    for rel in paths:
        target = repo / rel
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(
                f for f in target.rglob("*")
                if f.is_file()
                and "__pycache__" not in f.parts
                and not f.name.endswith((".pyc", ".pyo"))
                and ".egg-info" not in str(f)
            )
    for f in sorted(files, key=lambda x: str(x.relative_to(repo))):
        h.update(str(f.relative_to(repo)).encode())
        h.update(b"\0")
        h.update(_sha256_file(f).encode())
        h.update(b"\n")
    return h.hexdigest()
