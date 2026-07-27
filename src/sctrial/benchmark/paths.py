"""Manifest-scoped result layout.

    results/benchmark/<manifest_sha>/
        scenarios/    one CSV per scenario, written by producer jobs
        completion/   one JSON per scenario, written last, marks it complete
        combined/     the single aggregated CSV, written only by the aggregator
        logs/         SLURM stdout/stderr

WHY THE MANIFEST IS IN THE PATH
-------------------------------
Every stale-result incident in this project had the same shape: results produced
under one configuration were read back under another, because they shared a
directory and the newer file simply overwrote or sat beside the older one. A
resume adopted them, a figure averaged them, or a summary counted them.

Putting the manifest hash in the PATH makes that physically impossible rather
than merely forbidden. Two configurations cannot collide, a resume cannot adopt
results from a different configuration, and deleting a superseded run is one
directory removal instead of a hunt for which files belong to it.

The corollary is that nothing here resolves a "latest" or "current" directory.
Convenience lookups are exactly how the wrong results get read: a glob that
picks the newest CSV will happily pick up a development run that finished after
the definitive one. Callers pass a manifest SHA or a completion record, or they
get an error naming what is available.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SHA_RE = re.compile(r"^[0-9a-f]{8,64}$")

SCENARIOS = "scenarios"
COMPLETION = "completion"
COMBINED = "combined"
LOGS = "logs"
_SUBDIRS = (SCENARIOS, COMPLETION, COMBINED, LOGS)

# Pre-freeze work goes here so it can never be mistaken for a definitive run and
# can be deleted wholesale. Timing probes and smoke tests write under it.
PREFLIGHT = "_preflight"


class ResultLayout:
    """The directory tree for one manifest's results."""

    def __init__(self, root: Path | str, manifest_sha: str):
        if not _SHA_RE.match(str(manifest_sha)):
            raise ValueError(
                f"manifest_sha must be a hex digest, got {manifest_sha!r}. "
                "Results are addressed by configuration, never by name or date."
            )
        self.root = Path(root)
        self.manifest_sha = str(manifest_sha)
        self.base = self.root / self.manifest_sha

    def __repr__(self) -> str:
        return f"ResultLayout({self.base})"

    @property
    def scenarios(self) -> Path:
        return self.base / SCENARIOS

    @property
    def completion(self) -> Path:
        return self.base / COMPLETION

    def scenarios_for(self, grid: str) -> Path:
        """Scenario CSVs for ONE grid.

        Separated because the aggregator globs this directory and validates the
        result against the exact set the grid declares. The core and sensitivity
        grids write into the same manifest directory, so a shared directory would
        hand the core aggregator all 36 sensitivity files as UNEXPECTED -- and
        the natural "fix" of filtering to the expected set would destroy the only
        check that catches a genuinely unexpected scenario.
        """
        return self.base / SCENARIOS / grid

    def completion_for(self, grid: str) -> Path:
        return self.base / COMPLETION / grid

    @property
    def combined(self) -> Path:
        return self.base / COMBINED

    @property
    def logs(self) -> Path:
        return self.base / LOGS

    def create(self) -> ResultLayout:
        for sub in _SUBDIRS:
            (self.base / sub).mkdir(parents=True, exist_ok=True)
        return self

    def combined_csv(self, name: str = "benchmark_results.csv") -> Path:
        return self.combined / name

    def completion_marker(self, grid: str) -> Path:
        """The marker written by the aggregator for ONE grid, at the end.

        Per GRID, not per run. The core and sensitivity grids are aggregated by
        separate jobs into the same manifest directory, so a single shared marker
        would be written twice and the survivor would attest to whichever
        finished last -- the same last-writer-wins defect as the combined file
        itself, reintroduced one level up.
        """
        return self.combined / f"benchmark_complete_{grid}.json"

    def publication_marker(self) -> Path:
        """The WHOLE-BENCHMARK marker, written only by the finalizer.

        Grid-level markers say "core finished" or "sensitivity finished". Neither
        says "the manuscript benchmark is complete", and the asymmetric failure is
        real: core aggregates successfully, sensitivity never finishes, and a
        figure script finds a perfectly valid core marker and regenerates outputs
        from half the benchmark.

        This project has already met both other members of that class -- several
        jobs writing one combined file, and two grids sharing one scenario
        directory -- so the third is closed here rather than left to discipline.
        """
        return self.base / "publication_complete.json"

    def scenario_csv(self, scenario_id: str, grid: str | None = None) -> Path:
        base = self.scenarios_for(grid) if grid else self.scenarios
        return base / f"{scenario_id}.csv"

    def scenario_completion(self, scenario_id: str, grid: str | None = None) -> Path:
        base = self.completion_for(grid) if grid else self.completion
        return base / f"{scenario_id}.json"

    def completed_scenarios(self, grid: str | None = None) -> dict[str, dict]:
        """Scenarios with a completion record, keyed by scenario id.

        A CSV without a record is NOT complete: it is what a job killed part-way
        through an adaptive extension leaves behind.
        """
        out = {}
        base = self.completion_for(grid) if grid else self.completion
        if not base.exists():
            return out
        for path in sorted(base.glob("*.json")):
            try:
                out[path.stem] = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"corrupt completion record {path}: {exc}") from exc
        return out

    def orphan_scenarios(self, grid: str | None = None) -> list[str]:
        """Scenario CSVs with no completion record -- truncated or in flight."""
        base = self.scenarios_for(grid) if grid else self.scenarios
        if not base.exists():
            return []
        done = set(self.completed_scenarios(grid))
        return sorted(p.stem for p in base.glob("*.csv") if p.stem not in done)


def preflight_layout(root: Path | str, label: str) -> ResultLayout:
    """A clearly-separated tree for pre-freeze probes.

    Uses the same class so a probe exercises the real execution path, but under
    `_preflight/` so it is obvious that nothing here is a manuscript result and a
    single directory removal disposes of all of it.
    """
    layout = ResultLayout.__new__(ResultLayout)
    layout.root = Path(root) / PREFLIGHT
    layout.manifest_sha = str(label)
    layout.base = layout.root / str(label)
    return layout


def require_layout(root: Path | str, manifest_sha: str) -> ResultLayout:
    """Resolve a layout that must already contain a completed run.

    This is the loader entry point. It refuses to guess: no "latest", no newest
    CSV, no search across manifests.
    """
    root = Path(root)
    layout = ResultLayout(root, manifest_sha)
    if not layout.base.exists():
        available = (
            sorted(p.name for p in root.iterdir() if p.is_dir() and _SHA_RE.match(p.name))
            if root.exists() else []
        )
        raise FileNotFoundError(
            f"no results for manifest {manifest_sha} under {root}. "
            + (f"Available manifests: {available}" if available else "No manifests present.")
            + " Pass the manifest SHA explicitly; there is deliberately no 'latest'."
        )
    return layout
