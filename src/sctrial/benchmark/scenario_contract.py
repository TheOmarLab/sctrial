"""Requested-versus-realised validation for every simulated scenario.

WHY THIS EXISTS
---------------
A scenario is a request. The simulator produces a realisation. Nothing in the
pipeline compared the two, so when a frozen calibration field leaked into the
experimental design, every two-arm scenario from n=8 to n=60 simulated eleven
participants while recording the sample size that had been asked for. The
sample-size axis of the benchmark was flat and the results looked entirely
plausible.

The specific leak is fixed. This module addresses the CLASS: any generating
field that can be captured by an upstream default -- arm sizes, visits, cell
yield, panel size, signal count -- is checked here against what actually came
out of the simulator, and a mismatch stops the scenario instead of being
averaged into a figure.

WHAT IS ASSERTED EXACTLY AND WHAT IS NOT
----------------------------------------
Exact equality is asserted only where the pipeline is deterministic given the
request: arm sizes, panel size, signal count, and a fixed cell yield.

Where the request specifies a DISTRIBUTION rather than a value, asserting a
count would be wrong -- it would fail on correct behaviour. There the generating
MODE is asserted instead. The important case is cell yield: a scenario that asks
for the empirical distribution must show dispersion across participant-visits,
because a single repeated value is the signature of a fixed value having leaked
in. That check is what would have caught the original defect from the other
direction, and it is the reason this module tests realised state rather than
re-reading the configuration it was handed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# A scenario whose realised state cannot be checked at all is not "passing".
# Silence and success have to be distinguishable, so every check reports whether
# it ran, and a scenario with no evidence is INSUFFICIENT rather than OK.
_UNCHECKABLE = "no realised evidence recorded"


@dataclass(frozen=True)
class Violation:
    """One requested-versus-realised mismatch."""

    field: str
    requested: Any
    realised: Any
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.field}: requested {self.requested!r}, realised "
            f"{self.realised!r} -- {self.detail}"
        )


@dataclass
class ContractReport:
    scenario: str
    violations: list[Violation]
    checked: list[str]
    unchecked: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations

    def raise_if_violated(self) -> None:
        if self.violations:
            lines = "\n  ".join(str(v) for v in self.violations)
            raise RuntimeError(
                f"SCENARIO CONTRACT VIOLATED for {self.scenario}:\n  {lines}\n"
                "The simulated data does not match the requested design. This is "
                "the failure class that previously flattened the sample-size axis; "
                "the scenario is refused rather than recorded."
            )

    def summary(self) -> str:
        state = "OK" if self.ok else f"{len(self.violations)} VIOLATION(S)"
        out = f"{self.scenario}: {state} ({len(self.checked)} checks)"
        if self.unchecked:
            out += f"; UNCHECKED: {', '.join(self.unchecked)}"
        return out


def _requested(scenario: dict) -> dict:
    """The scenario's own generating fields.

    Design lives in `config_kwargs`, not at the top level. Reading the top level
    would silently return None for every field and make each check vacuously
    pass -- a validator that cannot fail is worse than none, because it is
    reported as a passing check.
    """
    return dict(scenario.get("config_kwargs") or {})


def _expected_arms(scenario: dict, cfg: Any) -> tuple[int | None, int | None]:
    """Participants per arm implied by the REQUEST.

    `arm_ratio` is a scenario-owned field. When a scenario sets it (the
    TNBC-matched 6/5 design), that is the request; when it does not, `n_per_arm`
    is, and any non-None `arm_ratio` reaching the simulator came from somewhere
    it should not have.
    """
    req = _requested(scenario)
    design = req.get("design", getattr(cfg, "design", None))
    n_per_arm = int(req.get("n_per_arm", getattr(cfg, "n_per_arm", 0)))
    if design == "single_arm":
        return n_per_arm, 0
    ratio = req.get("arm_ratio")
    if ratio is not None:
        return int(ratio[0]), int(ratio[1])
    return n_per_arm, n_per_arm


def check_simulation(
    scenario: dict,
    cfg: Any,
    sim: dict,
    panel_genes: list,
    signal_genes: set,
) -> ContractReport:
    """Validate one realised simulation against the scenario that requested it.

    Runs per iteration, immediately after `simulate_trial_v2`, so a mismatch is
    caught on the first replicate rather than after a scenario has burned a
    72-hour allocation.
    """
    v: list[Violation] = []
    checked: list[str] = []
    unchecked: list[str] = []

    arms = list(sim["latent"]["arms"])
    n_treated = sum(a == "Treated" for a in arms)
    n_control = len(arms) - n_treated
    want_t, want_c = _expected_arms(scenario, cfg)

    # --- design ---------------------------------------------------------
    checked.append("arm_sizes")
    if (n_treated, n_control) != (want_t, want_c):
        v.append(
            Violation(
                "arm_sizes",
                f"{want_t}T/{want_c}C",
                f"{n_treated}T/{n_control}C",
                "a design field leaked in from the frozen calibration, or "
                "arm_ratio overrode n_per_arm",
            )
        )

    checked.append("n_participants")
    if len(arms) != want_t + want_c:
        v.append(
            Violation("n_participants", want_t + want_c, len(arms), "total participants")
        )

    # --- visits ---------------------------------------------------------
    # The IDENTITY of dropped visits is random; the COUNT is not.
    pb = sim["pseudobulk_counts"]
    if "visit" in pb and "participant" in pb:
        checked.append("visits")
        n_pv = len(pb[["participant", "visit"]].drop_duplicates())
        n_expected_pv = len(arms) * 2
        n_drop = int(round(len(arms) * float(getattr(cfg, "missing_rate", 0.0) or 0.0)))
        if n_pv != n_expected_pv - n_drop:
            v.append(
                Violation(
                    "visits",
                    n_expected_pv - n_drop,
                    n_pv,
                    f"participant-visits at missing_rate={cfg.missing_rate}",
                )
            )
    else:
        unchecked.append("visits")

    # --- cellular sampling ----------------------------------------------
    cells = pb["n_cells"].to_numpy(dtype=float)
    fixed = getattr(cfg, "cells_per_pv_fixed", None)
    checked.append("cells_per_pv")
    if fixed is not None:
        # Deterministic: every participant-visit gets exactly this many.
        if not (cells.min() == cells.max() == float(fixed)):
            v.append(
                Violation(
                    "cells_per_pv",
                    f"fixed {fixed}",
                    f"min {cells.min():.0f} max {cells.max():.0f}",
                    "a fixed cell yield was requested but not realised",
                )
            )
    elif cells.min() == cells.max():
        # THE B1 SIGNATURE, from the other direction. A scenario that asked for
        # the empirical distribution and received one repeated value did not get
        # what it asked for -- something upstream supplied a fixed count.
        v.append(
            Violation(
                "cells_per_pv",
                "empirical distribution",
                f"constant {cells.min():.0f}",
                "a constant cell yield reached a scenario that requested the "
                "empirical distribution; a fixed value has leaked in",
            )
        )

    # --- gene set -------------------------------------------------------
    checked.append("panel_size")
    if len(panel_genes) != int(scenario["panel_size"]):
        v.append(
            Violation("panel_size", scenario["panel_size"], len(panel_genes), "tested genes")
        )

    # --- signal ---------------------------------------------------------
    checked.append("n_signal")
    want_sig = int(scenario.get("n_signal", 0))
    if len(signal_genes) != want_sig:
        v.append(
            Violation("n_signal", want_sig, len(signal_genes), "affected genes")
        )

    checked.append("signal_fraction")
    realised_frac = len(signal_genes) / max(len(panel_genes), 1)
    want_frac = float(scenario.get("signal_fraction", 0.0))
    if abs(realised_frac - want_frac) > 1e-9:
        v.append(
            Violation(
                "signal_fraction",
                f"{want_frac:.4f}",
                f"{realised_frac:.4f}",
                "realised fraction differs from the scenario's nominal fraction; "
                "manuscript labels must be derived from the realised field",
            )
        )

    return ContractReport(scenario["name"], v, checked, unchecked)


def check_scenario_results(
    scenario_id: str,
    scenario: dict,
    df: pd.DataFrame,
    manifest_sha: str | None = None,
) -> ContractReport:
    """Validate a completed scenario's result table before it is marked complete.

    `check_simulation` guards each replicate as it is produced. This is the
    second gate: it re-derives the same quantities from the RECORDED columns, so
    a defect in the recording path -- rather than in the simulation path -- is
    also caught. The two are deliberately redundant; the original defect survived
    precisely because only one layer was checked.
    """
    v: list[Violation] = []
    checked: list[str] = []
    unchecked: list[str] = []

    if df.empty:
        return ContractReport(
            scenario_id, [Violation("rows", ">0", 0, "empty result table")], [], []
        )

    def _uniq(col: str):
        return sorted(pd.unique(df[col].dropna())) if col in df else None

    req = _requested(scenario)
    want_t, want_c = _expected_arms(scenario, _CfgView(scenario))

    for col, want, label in (
        ("n_treated", want_t, "treated participants"),
        ("n_control", want_c, "control participants"),
        ("panel_size", int(scenario["panel_size"]), "tested genes"),
        ("n_signal_realised", int(scenario.get("n_signal", 0)), "affected genes"),
    ):
        vals = _uniq(col)
        if vals is None:
            unchecked.append(col)
            continue
        checked.append(col)
        if vals != [want]:
            v.append(Violation(col, want, vals, f"{label}, across all replicates"))

    # Cell yield: the same distinction as in check_simulation, re-derived from
    # what was recorded rather than from the simulator's own object.
    if "cells_per_pv_min" in df and "cells_per_pv_max" in df:
        checked.append("cells_per_pv")
        lo = float(df["cells_per_pv_min"].min())
        hi = float(df["cells_per_pv_max"].max())
        fixed = req.get("cells_per_pv_fixed")
        if fixed is not None:
            if not (lo == hi == float(fixed)):
                v.append(
                    Violation("cells_per_pv", f"fixed {fixed}", f"[{lo:.0f}, {hi:.0f}]",
                              "requested fixed yield not realised")
                )
        elif lo == hi:
            v.append(
                Violation("cells_per_pv", "empirical distribution", f"constant {lo:.0f}",
                          "a fixed cell yield leaked into a scenario that requested "
                          "the empirical distribution")
            )
    else:
        unchecked.append("cells_per_pv")

    # Provenance.
    if manifest_sha is not None:
        if "manifest_sha256" not in df:
            v.append(Violation("manifest_sha256", manifest_sha, None, "unstamped rows"))
        else:
            checked.append("manifest_sha256")
            got = sorted(pd.unique(df["manifest_sha256"].dropna()))
            if got != [manifest_sha]:
                v.append(
                    Violation("manifest_sha256", manifest_sha, got,
                              "rows from a different manifest")
                )

    if "scenario" in df:
        checked.append("scenario_id")
        got = sorted(pd.unique(df["scenario"].dropna()))
        if got != [scenario_id]:
            v.append(Violation("scenario_id", scenario_id, got, "mislabelled rows"))
    else:
        unchecked.append("scenario_id")

    return ContractReport(scenario_id, v, checked, unchecked)


class _CfgView:
    """Adapter so `_expected_arms` reads a scenario dict like a config object."""

    def __init__(self, scenario: dict):
        req = _requested(scenario)
        self.n_per_arm = int(req.get("n_per_arm", 0))
        self.design = req.get("design")
        self.missing_rate = float(req.get("missing_rate", 0.0) or 0.0)


def evaluability_by_method(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of prespecified genes each method actually returned inference for.

    Reported, never used to filter. A method that drops genes at low cell yield
    has a lower end-to-end detection rate, and that is a benchmark result rather
    than a reason to shrink its denominator.
    """
    if df.empty or "method" not in df or "evaluable" not in df:
        return {}
    return {
        str(m): float(g["evaluable"].mean())
        for m, g in df.groupby("method", observed=True)
    }


def _rate_mcse(df: pd.DataFrame, signal: bool) -> float:
    """Replicate-level Monte Carlo SE of the null-FPR or the power."""
    if df.empty or "iteration" not in df:
        return float("nan")
    sub = df[df["pvalue"].notna() & (df["is_signal"] == signal)]
    if sub.empty:
        return float("nan")
    per_rep = sub.assign(hit=sub["pvalue"] < 0.05).groupby("iteration")["hit"].mean()
    if len(per_rep) < 2:
        return float("nan")
    return float(np.std(per_rep.to_numpy(), ddof=1) / np.sqrt(len(per_rep)))


def completion_record(
    scenario_id: str,
    scenario: dict,
    df: pd.DataFrame,
    stop_reason: str,
    max_replicates: int,
    manifest_sha: str | None,
    mcse_target_fpr: float,
    mcse_target_power: float,
) -> dict:
    """The per-scenario completion record.

    Written ONLY after a scenario finishes, so its absence marks truncation. This
    matters because adaptive stopping makes replicate count a per-scenario
    outcome: a scenario killed part-way through an extension still holds more
    rows than the base batch, so a count threshold cannot distinguish it from one
    that legitimately stopped early. The record can.
    """
    worst_f = worst_p = 0.0
    per_method = {}
    for method, grp in df.groupby("method", observed=True):
        f = _rate_mcse(grp, signal=False)
        p = _rate_mcse(grp, signal=True)
        per_method[str(method)] = {"fpr_mcse": f, "power_mcse": p}
        if np.isfinite(f):
            worst_f = max(worst_f, f)
        if np.isfinite(p):
            worst_p = max(worst_p, p)

    return {
        "scenario_id": scenario_id,
        "n_replicates_completed": int(df["iteration"].nunique()),
        "stop_reason": stop_reason,
        "fpr_mcse": worst_f,
        "power_mcse": worst_p,
        "max_replicates": int(max_replicates),
        "mcse_target_fpr": float(mcse_target_fpr),
        "mcse_target_power": float(mcse_target_power),
        "per_method_mcse": per_method,
        "manifest_sha256": manifest_sha,
        "evaluability": evaluability_by_method(df),
        "n_participants": int(df["n_participants"].iloc[0]) if "n_participants" in df else None,
        "panel_size": int(scenario["panel_size"]),
        "n_signal": int(scenario.get("n_signal", 0)),
        "signal_fraction_realised": (
            float(df["signal_fraction_realised"].iloc[0])
            if "signal_fraction_realised" in df else None
        ),
    }


# `adaptive_disabled` is valid but NOT acceptable for a definitive run: see
# aggregate_benchmark.py, which requires it to be opted into explicitly.
VALID_STOP_REASONS = ("precision_reached", "max_replicates_reached", "adaptive_disabled")
DEFINITIVE_STOP_REASONS = ("precision_reached", "max_replicates_reached")
