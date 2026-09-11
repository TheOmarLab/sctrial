"""Guard against loading datasets from anywhere but the canonical loaders.

Two production bugs motivated this file, both silent:

1. ``scripts/run_benchmark.py`` calibrated the benchmark simulator from
   ``try/GSE169246/tnbc_processed.h5ad`` -- a hand-placed scratch copy outside
   the dataset layout. When that file was deleted the phase aborted; while it
   existed it calibrated the simulator from pre-reprocessing TNBC data even
   though every other analysis used the current object.

2. ``figure3`` / ``supp_fig5`` located the benchmark CSV with
   ``Path(__file__).parents[4]``, which encodes the local checkout depth. On the
   HPC the repo sits one level shallower, so the path resolved outside the
   project, the file was not found, and four benchmark panels rendered blank
   with no error.

Both classes fail silently and produce plausible-looking output, so they are
caught here rather than by review.

Rules enforced:
  * Only ``src/sctrial/datasets.py`` may call ``read_h5ad``. Everything else must
    go through a loader (``load_tnbc_zhang()``, ``get_aml()``, ...).
  * No module may reference a scratch/personal directory.
  * No module may reach outside the repo with ``parents[N]`` path arithmetic to
    locate data; derive from ``MANUSCRIPT_DIR`` instead.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCAN_DIRS = ["src", "scripts", "manuscript_figures"]

# Only this file may read h5ad directly -- it *is* the canonical loader layer.
READ_H5AD_ALLOWLIST = {REPO / "src" / "sctrial" / "datasets.py"}

SCRATCH_PAT = re.compile(
    r'["\'](?:[^"\']*/)?(?:try|scratch|tmp_data|Downloads|OneDrive)/[^"\']*\.h5ad["\']'
    r'|/\s*["\'](?:try|scratch|Downloads)["\']'
)
# parents[4] and deeper reaches above the repo root from any module in the tree.
DEEP_PARENTS_PAT = re.compile(r"parents\[([4-9])\]")


def _py_files():
    for d in SCAN_DIRS:
        root = REPO / d
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


def _strip_comments(text: str) -> str:
    """Drop whole-line comments so documentation of a past bug is not a hit."""
    return "\n".join(
        "" if line.lstrip().startswith("#") else line for line in text.splitlines()
    )


def test_read_h5ad_only_in_datasets_module():
    offenders = []
    for p in _py_files():
        if p in READ_H5AD_ALLOWLIST:
            continue
        src = _strip_comments(p.read_text(encoding="utf-8", errors="ignore"))
        for i, line in enumerate(src.splitlines(), 1):
            if "read_h5ad(" in line:
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "read_h5ad() outside src/sctrial/datasets.py. Load datasets through the "
        "canonical loaders (load_tnbc_zhang(), get_aml(), ...) so every analysis "
        "sees the same reprocessed object:\n  " + "\n  ".join(offenders)
    )


def test_no_scratch_directory_dataset_paths():
    offenders = []
    for p in _py_files():
        src = _strip_comments(p.read_text(encoding="utf-8", errors="ignore"))
        for i, line in enumerate(src.splitlines(), 1):
            if SCRATCH_PAT.search(line):
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "Dataset read from a scratch/personal directory. Data must live under "
        "datasets/<name>/ and be loaded via its loader:\n  " + "\n  ".join(offenders)
    )


def test_no_deep_parents_path_arithmetic():
    offenders = []
    for p in _py_files():
        src = _strip_comments(p.read_text(encoding="utf-8", errors="ignore"))
        for i, line in enumerate(src.splitlines(), 1):
            m = DEEP_PARENTS_PAT.search(line)
            if m:
                offenders.append(f"{p.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, (
        "parents[4+] path arithmetic encodes the local checkout depth and "
        "resolves outside the project on the HPC (silently blanking panels). "
        "Derive paths from MANUSCRIPT_DIR / the loader layer instead:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("name", ["load_tnbc_zhang", "load_aml", "load_cart"])
def test_canonical_loaders_are_importable(name):
    """The guard above is only meaningful if the loaders it points to exist."""
    import sctrial.datasets as ds

    assert hasattr(ds, name), f"canonical loader {name}() missing from sctrial.datasets"
