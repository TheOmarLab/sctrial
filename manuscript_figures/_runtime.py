"""Recorded wall-clock time of one participant-level analysis per clinical dataset.

Wall-clock time differs from run to run, so a panel that measures it while it is
drawn cannot be reproduced byte for byte. The measurements are recorded once into a
table kept in the repository (``recorded/dataset_runtime.csv``) and the panel reads
that table. To record again, on the cluster and with single-threaded BLAS::

    python -m manuscript_figures._runtime

Each dataset is analyzed ``N_REPLICATES`` times with the participant-level function
of its design (``did_table``, ``within_arm_comparison`` or ``between_arm_comparison``
on the immune signatures); the panel shows the median and the interquartile range.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ._shared import between_arm_comparison, did_table, within_arm_comparison

RECORDED = Path(__file__).resolve().parent / "recorded" / "dataset_runtime.csv"
N_REPLICATES = 5
METHOD_FAMILY = {"two_arm_did": "did_table", "paired": "within_arm_comparison",
                 "cross_sectional": "between_arm_comparison"}
COLUMNS = ["dataset", "design_type", "method", "n_cells", "n_participants", "n_features",
           "replicate", "time_s", "peak_mb"]


def measure(datasets) -> pd.DataFrame:
    """Time the participant-level analysis of each dataset, ``N_REPLICATES`` times.

    ``datasets`` holds ``(name, adata, design, visits, features, design_type)`` tuples,
    as ``figure3_robustness_benchmarking._load_dataset_by_index`` returns them.
    """
    rows = []
    for name, adata, design, visits, sigs, dtype in datasets:
        n_participants = int(adata.obs[design.participant_col].nunique())
        for rep in range(N_REPLICATES):
            gc.collect()
            tracemalloc.start()
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if dtype == "cross_sectional":
                    between_arm_comparison(adata, visit=visits[0], features=sigs, design=design,
                                           aggregate="participant_visit", standardize=True)
                elif dtype == "paired":
                    within_arm_comparison(adata, arm="All", features=sigs, design=design,
                                          visits=visits, aggregate="participant_visit",
                                          standardize=True)
                else:
                    did_table(adata, features=sigs, design=design, visits=visits,
                              aggregate="participant_visit", standardize=True)
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append({"dataset": name, "design_type": dtype, "method": METHOD_FAMILY[dtype],
                         "n_cells": int(adata.n_obs), "n_participants": n_participants,
                         "n_features": len(sigs), "replicate": rep, "time_s": elapsed,
                         "peak_mb": peak / 1024**2})
        times = [r["time_s"] for r in rows if r["dataset"] == name]
        print(f"    {name}: median {np.median(times):.2f} s over {N_REPLICATES} runs")
    return pd.DataFrame(rows, columns=COLUMNS)


def load_recorded(path: Path = RECORDED) -> pd.DataFrame:
    """The recorded measurements (lines starting with ``#`` hold their provenance)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Record it with: python -m manuscript_figures._runtime")
    table = pd.read_csv(path, comment="#")
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path.name} lacks columns {missing}")
    return table


def check_recorded(recorded: pd.DataFrame, name: str, *, n_cells: int, n_participants: int,
                   n_features: int) -> None:
    """Refuse a recording made on a different version of a dataset."""
    rows = recorded[recorded["dataset"] == name]
    if rows.empty:
        raise ValueError(f"no recorded runtime for {name}; record again (see {__name__})")
    seen = (int(rows["n_cells"].iloc[0]), int(rows["n_participants"].iloc[0]),
            int(rows["n_features"].iloc[0]))
    if seen != (n_cells, n_participants, n_features):
        raise ValueError(
            f"recorded runtime for {name} was measured on {seen[0]:,} cells, {seen[1]} "
            f"participants and {seen[2]} features; the dataset now has {n_cells:,}, "
            f"{n_participants} and {n_features}. Record again (see {__name__}).")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,  # noqa: S603, S607
                              cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    except Exception:
        return os.environ.get("SCTRIAL_COMMIT", "unknown")


def record(path: Path = RECORDED) -> pd.DataFrame:
    """Measure all six datasets and write the table with its provenance."""
    from .main.figure3_robustness_benchmarking import _load_dataset_by_index

    frames = []
    for idx in range(6):
        ds = _load_dataset_by_index(idx)
        if ds is None:
            raise RuntimeError(f"dataset {idx} could not be loaded; nothing was recorded")
        frames.append(measure([ds]))
        del ds
        gc.collect()
    table = pd.concat(frames, ignore_index=True)
    threads = {k: os.environ.get(k, "unset") for k in
               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    header = [
        "# Wall-clock time of one participant-level analysis per dataset (seconds) and peak",
        "# traced Python memory (MB). Written by manuscript_figures._runtime.record().",
        f"# recorded: {time.strftime('%Y-%m-%d')}",
        f"# commit: {_commit()}",
        f"# host: {platform.node()} ({platform.machine()}), python {platform.python_version()}",
        f"# threads: {threads}",
        f"# replicates per dataset: {N_REPLICATES}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        table.to_csv(fh, index=False)
    print(f"  Recorded {len(table)} measurements to {path}")
    return table


if __name__ == "__main__":
    record()
