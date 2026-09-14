"""Stable, process-independent seeds. The ONLY way to derive a seed from a name.

WHY THIS MODULE EXISTS
----------------------
`calibration.py` seeded the split-half partition with

    np.random.default_rng(abs(hash((participant, visit, stratum))) % (2**32))

`hash()` on strings is salted by `PYTHONHASHSEED`, which CPython randomises for
every interpreter process. So the split-half assignment -- the device that makes
the participant and participant-by-visit variance components identifiable -- was
different on every run, and with it every variance component, the frozen
calibration derived from them, the bootstrap acceptance tolerance, and the gate
ledger.

The failure was invisible to every determinism test written so far because those
compared two calls INSIDE one process, which is exactly where `hash()` is stable.
It only appeared when two gate runs under the identical commit produced
acceptance tolerances differing by up to 3%.

Two things follow, and both are enforced here rather than by convention:

1. **Never derive a seed from `hash()`.** Use `stable_seed`, which is SHA-256 of
   a canonical serialisation and therefore identical across processes, machines,
   Python versions and `PYTHONHASHSEED` settings.

2. **Never fix the problem by pinning `PYTHONHASHSEED`.** That would make this
   particular bug reproducible while leaving scientific randomness dependent on
   an interpreter setting, and would hide the defective idiom for the next
   author. The seed must be a property of the code, not of the environment.

CANONICAL SERIALISATION
-----------------------
Parts are joined with an ASCII unit separator (0x1f), which cannot occur in the
identifiers used here. Naive concatenation is ambiguous -- ("12", "3", "45") and
("1", "23", "45") would collide -- and a collision here silently gives two
different strata the same partition.

The namespace prevents collisions between unrelated uses: the same
(participant, visit) must not produce the same seed for a split-half partition
and for a scenario replicate.
"""
from __future__ import annotations

import hashlib

__all__ = ["stable_seed", "SEP"]

# ASCII unit separator. Cannot appear in a participant id, visit label, cell type
# or scenario name, so the serialisation is unambiguous.
SEP = "\x1f"


def stable_seed(namespace: str, *parts: object, bits: int = 64) -> int:
    """A deterministic seed derived from a namespace and identifying parts.

    Identical across processes, interpreters and `PYTHONHASHSEED` values, which
    is the entire point: `hash()` is not.

    Parameters
    ----------
    namespace
        Distinguishes unrelated uses of the same identifiers. Include a version
        suffix (``"..._v1"``) when a change to the derived quantity should
        deliberately produce different draws.
    parts
        Identifying components. Converted with ``str`` and joined with an
        unambiguous separator.
    bits
        Width of the returned integer. 64 by default; NumPy accepts arbitrary
        non-negative integers as seeds, and 64 bits is far past any collision
        concern at this scale.
    """
    if not namespace:
        raise ValueError("namespace must be a non-empty string")
    if bits % 8 or not 8 <= bits <= 256:
        raise ValueError(f"bits must be a multiple of 8 in [8, 256], got {bits}")
    payload = SEP.join([namespace, *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")
