"""Seeds must be identical ACROSS PROCESSES, not merely within one.

This file exists because a determinism bug survived every previous test.
`calibration.py` seeded the split-half partition from

    abs(hash((participant, visit, stratum))) % (2**32)

and CPython salts `hash()` of strings with `PYTHONHASHSEED`, which is randomised
per interpreter process. So the partition that makes the participant and
participant-by-visit variance components identifiable was different on every run,
and with it sigma_b, sigma_u, sigma_e, the frozen calibration derived from them,
the bootstrap acceptance tolerance and the gate ledger.

Every determinism test written before this compared two calls INSIDE one process
-- exactly where `hash()` is stable -- and passed. The defect only surfaced when
two gate runs under the identical commit produced acceptance tolerances differing
by up to 3%, and was confirmed by four processes whose bootstrap digests agreed
only when `PYTHONHASHSEED` was pinned.

So these tests SPAWN SUBPROCESSES with deliberately different `PYTHONHASHSEED`
values. A same-process assertion cannot detect this class and must not be
substituted for one.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from sctrial.benchmark.seeds import stable_seed  # noqa: E402


def _run(code: str, hashseed: str) -> dict:
    """Execute `code` in a fresh interpreter with a given PYTHONHASHSEED."""
    import os

    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=str(SRC))
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise AssertionError(f"subprocess failed (PYTHONHASHSEED={hashseed}):\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_SEED_PROBE = """
    import json
    from sctrial.benchmark.seeds import stable_seed
    print(json.dumps({
        "split_half": stable_seed("calibration_split_half_v1", "P016", "Post", "P016|Post"),
        "scenario":   stable_seed("scenario_replicate_v1", 2024, "two_arm__null_n60", 37, bits=32),
        "unicode":    stable_seed("ns", "participant\\u00e9", "Pre", None),
    }))
"""


def test_stable_seed_is_identical_across_processes():
    """The whole point: different PYTHONHASHSEED, same seed.

    `hash()` would fail this. That is the difference this file exists to pin.
    """
    a = _run(_SEED_PROBE, "1")
    b = _run(_SEED_PROBE, "987654321")
    c = _run(_SEED_PROBE, "random")
    d = _run(_SEED_PROBE, "random")
    assert a == b == c == d, (
        "stable_seed differs across processes; a seed derived this way cannot "
        f"reproduce a calibration.\n1: {a}\n987654321: {b}\nrandom: {c}\nrandom: {d}"
    )


def test_builtin_hash_would_fail_this_test():
    """Demonstrate the defect this module guards against, rather than asserting it abstractly.

    If this ever stops differing, CPython has changed its hashing and the guard
    above has become vacuous -- which is worth knowing, because a test that
    cannot fail is not a test.
    """
    code = """
        import json
        print(json.dumps({"h": abs(hash(("P016", "Post", "s"))) % (2**32)}))
    """
    a = _run(code, "1")
    b = _run(code, "987654321")
    assert a != b, (
        "builtin hash() no longer varies with PYTHONHASHSEED; the seed-stability "
        "guard may now be vacuous and should be re-examined"
    )


def test_seed_distinguishes_every_component():
    """Changing any identifying part must change the seed."""
    base = stable_seed("ns", "P01", "Pre", "s1")
    assert stable_seed("ns", "P02", "Pre", "s1") != base
    assert stable_seed("ns", "P01", "Post", "s1") != base
    assert stable_seed("ns", "P01", "Pre", "s2") != base
    assert stable_seed("other", "P01", "Pre", "s1") != base, "namespace must separate uses"


def test_serialisation_is_unambiguous():
    """Concatenation must not let different inputs collide.

    ("12","3","45") and ("1","23","45") differ; a naive join would give "12345"
    for both, and a collision here silently assigns two strata the same partition.
    """
    assert stable_seed("ns", "12", "3", "45") != stable_seed("ns", "1", "23", "45")
    assert stable_seed("ns", "a", "b") != stable_seed("ns", "ab")


def test_seed_rejects_bad_arguments():
    with pytest.raises(ValueError, match="namespace"):
        stable_seed("", "x")
    with pytest.raises(ValueError, match="bits"):
        stable_seed("ns", "x", bits=7)


_CALIBRATION_PROBE = """
    import json
    import numpy as np
    from sctrial.benchmark.calibration import SummaryAccumulator

    G = 60
    rng = np.random.default_rng(0)          # fixture data, explicitly seeded
    acc = SummaryAccumulator(n_genes=G, gene_names=[f"g{i}" for i in range(G)])
    for i in range(8):
        pid = f"P{i:02d}"
        arm = "Treated" if i % 2 == 0 else "Control"
        for visit in ("Pre", "Post"):
            counts = rng.poisson(4.0, size=(40, G)).astype(float)
            acc.add_block(counts, pid, visit, arm)

    # The split-half assignment itself, and everything derived from it.
    rows = sorted(acc.pv_rows, key=lambda r: (r["participant"], r["visit"]))
    split = [[r["participant"], r["visit"], int(r["n_cells_a"]),
              float(np.asarray(r["counts_a"]).sum()),
              float(np.asarray(r["counts_b"]).sum())] for r in rows]
    vc = {k: (float(v) if isinstance(v, (int, float)) else str(v))
          for k, v in sorted(acc.variance_components().items())}
    print(json.dumps({"split": split, "variance_components": vc}))
"""


def test_split_half_and_variance_components_are_identical_across_processes():
    """The actual defect: the calibration estimator must not depend on the interpreter.

    Asserts the split-half assignment AND the variance components derived from
    it, because the assignment is what feeds sigma_b, sigma_u and sigma_e, and
    those feed the frozen simulator configuration.
    """
    a = _run(_CALIBRATION_PROBE, "1")
    b = _run(_CALIBRATION_PROBE, "987654321")
    c = _run(_CALIBRATION_PROBE, "random")

    assert a["split"] == b["split"] == c["split"], (
        "the split-half partition depends on PYTHONHASHSEED, so the calibration "
        "cannot be reproduced from code plus data plus declared seeds"
    )
    assert a["variance_components"] == b["variance_components"] == c["variance_components"], (
        "variance components differ across processes; the frozen calibration "
        "would not be reproducible"
    )
    # The fixture must actually exercise a non-trivial split.
    assert any(r[2] > 0 for r in a["split"])
    assert len(a["split"]) == 16


def test_no_hash_based_seeding_remains_in_the_package():
    """No module may derive a seed from builtin `hash()`.

    A source check, because this is one of the few properties with no runtime
    signature: the wrong idiom produces a perfectly good number every time and
    only misbehaves across processes.
    """
    import ast

    # PARSED, not grepped. A regex over source text cannot tell code from prose:
    # the first version flagged `manifest_hash(` (substring), the second flagged a
    # docstring that merely explains the defect. The AST answers the actual
    # question -- is builtin `hash` CALLED anywhere -- with no false positives and,
    # more importantly, no temptation to loosen the pattern until it goes quiet.
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "seeds.py":
            continue                      # documents the defect in prose
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert not offenders, (
        "builtin hash() used where a stable seed is required:\n  " + "\n  ".join(offenders)
    )
