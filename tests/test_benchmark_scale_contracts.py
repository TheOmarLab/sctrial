"""Scale contracts for the benchmark: every method must report on the SAME scale.

Motivated by a defect that reached the manuscript. limma/voom/dreamlet/edgeR report
**log2** fold-changes, while the simulator injects effects on the **natural log**
scale and NEBULA/sctrial_did/wilcoxon_paired report natural-log betas. The runners
harvested ``logFC`` unconverted into the same ``estimated_beta`` column as
natural-log truth, inflating dreamlet's effects by 1/ln2 = 1.4427.

That single missing conversion manufactured the paper's "substantial effect-size
bias on signal genes" finding: measured dreamlet signal-gene beta was 0.7157
against 0.5/ln2 = 0.7213, while every natural-log method sat at 0.498-0.504.
Correcting it flips all 16 bias cells in Figure 3C and collapses dreamlet's RMSE
from 0.230 to 0.048 -- indistinguishable from sctrial.

The failure was invisible because each half was correct in isolation: R really does
return log2, and the truth really is natural log. Only the seam was wrong. These
tests pin the seam.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

RUNNERS = Path(__file__).resolve().parent.parent / "src" / "sctrial" / "benchmark" / "runners"

# Runners whose upstream R package reports log2 fold-changes and therefore MUST
# convert to natural log before populating `beta`/`ci_lo`/`ci_hi`.
LOG2_RUNNERS = {
    "dreamlet_runner.py": ["logFC", "CI.L", "CI.R"],
    "limma_voom.py": ["logFC", "CI.L", "CI.R"],
    "edger_qlf.py": ["logFC"],
}


@pytest.mark.parametrize("fname,fields", sorted(LOG2_RUNNERS.items()))
def test_log2_runners_convert_to_natural_log(fname, fields):
    """Every log2 field harvested from R must be scaled by ln2."""
    path = RUNNERS / fname
    if not path.exists():  # pragma: no cover - runner removed
        pytest.skip(f"{fname} not present")
    src = path.read_text(encoding="utf-8")

    assert "_LN2" in src, (
        f"{fname} harvests log2 fold-changes but defines no ln2 conversion. "
        "Natural-log truth and natural-log competitors share the estimated_beta "
        "column; an unconverted logFC inflates effects by 1.4427x."
    )

    for field in fields:
        # Find the harvest expression and require the scaling. Allow the wrapping
        # call's closing paren(s) between row.get(...) and `* _LN2`, e.g.
        #   float(row.get("logFC", np.nan)) * _LN2
        pat = re.compile(
            r'row\.get\(\s*["\']' + re.escape(field) + r'["\']\s*,[^)]*\)[\s)]*\*\s*_LN2'
        )
        assert pat.search(src), (
            f"{fname}: `{field}` is harvested without `* _LN2`. It is a log2 value "
            f"being written into a natural-log column."
        )


def test_ln2_constant_is_correct():
    """A wrong constant would be worse than none - pin the value."""
    for fname in LOG2_RUNNERS:
        path = RUNNERS / fname
        if not path.exists():  # pragma: no cover
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_LN2" for t in node.targets
            ):
                found = True
                # must be log(2), not log2 of something or a literal typo
                seg = ast.unparse(node.value)
                assert "log(2" in seg, f"{fname}: _LN2 is `{seg}`, expected np.log(2.0)"
        assert found, f"{fname}: _LN2 assigned nowhere"
    assert np.isclose(float(np.log(2.0)), 0.6931471805599453)


def test_natural_log_runners_do_not_convert():
    """NEBULA/sctrial/wilcoxon are already natural-log; scaling them would break parity."""
    for fname in ("nebula_runner.py", "sctrial_did.py", "wilcoxon_paired.py"):
        path = RUNNERS / fname
        if not path.exists():  # pragma: no cover
            continue
        src = path.read_text(encoding="utf-8")
        assert "_LN2" not in src, (
            f"{fname} reports natural-log betas already; applying an ln2 factor "
            "would desynchronise it from the simulator truth."
        )


def test_conversion_recovers_injected_effect():
    """Numeric check: a log2 estimate of a 0.5 natural-log effect is 0.5/ln2."""
    true_effect_natural = 0.5
    r_reported_log2 = true_effect_natural / np.log(2.0)  # what voom/dreamlet returns
    assert np.isclose(r_reported_log2, 0.7213475204444817)
    # the runner's conversion must bring it back to the truth
    converted = r_reported_log2 * float(np.log(2.0))
    assert np.isclose(converted, true_effect_natural)
    # and the uncorrected value is exactly the spurious "bias" that was published
    assert np.isclose(r_reported_log2 - true_effect_natural, 0.2213475204444817)
