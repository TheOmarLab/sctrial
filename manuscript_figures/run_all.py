#!/usr/bin/env python3
"""
Run all manuscript figure and table generation scripts.

Usage:
    python -m manuscript_figures.run_all              # everything
    python -m manuscript_figures.run_all --main       # main figures only
    python -m manuscript_figures.run_all --supp       # supplementary only
    python -m manuscript_figures.run_all --figure 2   # single main figure
    python -m manuscript_figures.run_all --supp-fig 7 # single supp figure
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from ._shared import apply_style, MAIN_OUTPUT, SUPP_OUTPUT


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MAIN_FIGURES = {
    1: ("Figure 1: Problem & Framework",       "main.figure1_problem_framework"),
    2: ("Figure 2: Immunotherapy DiD",          "main.figure2_immunotherapy_did"),
    3: ("Figure 3: Multi-Dataset Generalization","main.figure3_multi_dataset"),
    4: ("Figure 4: Statistical Robustness",     "main.figure4_statistical_robustness"),
    5: ("Figure 5: Biological Discovery",       "main.figure5_biological_discovery"),
    6: ("Figure 6: Scalability & Power",        "main.figure6_scalability_power"),
}

SUPP_TABLES = {
    "tables": ("Supplementary Tables 1–4", "supp.supp_tables"),
}

SUPP_FIGURES = {
    1: ("Supp Fig 1: QC Metrics",              "supp.supp_fig1_qc"),
    2: ("Supp Fig 2: Aggregation Sensitivity",  "supp.supp_fig2_aggregation"),
    3: ("Supp Fig 3: Clinical Trial Details",    "supp.supp_fig3_clinical"),
    4: ("Supp Fig 4: UMAP",                     "supp.supp_fig4_umap"),
    5: ("Supp Fig 5: Outcome Correlation",       "supp.supp_fig5_outcome"),
    6: ("Supp Fig 6: Robustness Details",        "supp.supp_fig6_robustness_details"),
    7: ("Supp Fig 7: Biological Details",        "supp.supp_fig7_bio_details"),
    8: ("Supp Fig 8: Individual Heterogeneity",  "supp.supp_fig8_heterogeneity"),
    9: ("Supp Fig 9: Temporal Dynamics",         "supp.supp_fig9_temporal"),
}


def _run(label: str, module_path: str) -> bool:
    """Import *module_path* and call its ``generate()`` function."""
    import importlib

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    t0 = time.time()
    try:
        mod = importlib.import_module(f".{module_path}", package="manuscript_figures")
        mod.generate()
        elapsed = time.time() - t0
        print(f"  ✓ Done ({elapsed:.1f}s)")
        return True
    except Exception:
        elapsed = time.time() - t0
        print(f"  ✗ FAILED ({elapsed:.1f}s)")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_main(figures: list[int] | None = None):
    """Generate main figures (all or a subset)."""
    targets = figures or sorted(MAIN_FIGURES)
    results = {}
    for num in targets:
        if num not in MAIN_FIGURES:
            print(f"  Unknown main figure: {num}")
            continue
        label, mod = MAIN_FIGURES[num]
        results[label] = _run(label, mod)
    return results


def run_supp(figures: list[int] | None = None):
    """Generate supplementary tables and figures (all or a subset)."""
    results = {}

    # Always run tables unless specific figures requested
    if figures is None:
        label, mod = SUPP_TABLES["tables"]
        results[label] = _run(label, mod)

    targets = figures or sorted(SUPP_FIGURES)
    for num in targets:
        if num not in SUPP_FIGURES:
            print(f"  Unknown supplementary figure: {num}")
            continue
        label, mod = SUPP_FIGURES[num]
        results[label] = _run(label, mod)
    return results


def run_all():
    """Generate everything."""
    apply_style()
    print("=" * 60)
    print("  MANUSCRIPT FIGURE GENERATION")
    print("=" * 60)
    print(f"  Main figures  → {MAIN_OUTPUT}")
    print(f"  Supplementary → {SUPP_OUTPUT}")
    print()

    t0 = time.time()
    results = {}
    results.update(run_main())
    results.update(run_supp())

    # Summary
    elapsed = time.time() - t0
    n_ok = sum(results.values())
    n_fail = len(results) - n_ok
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE  ({elapsed / 60:.1f} min)")
    print(f"  {n_ok} succeeded, {n_fail} failed")
    if n_fail:
        print("  Failed:")
        for label, ok in results.items():
            if not ok:
                print(f"    • {label}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate manuscript figures")
    parser.add_argument("--main", action="store_true", help="Main figures only")
    parser.add_argument("--supp", action="store_true", help="Supplementary only")
    parser.add_argument("--figure", type=int, help="Single main figure number")
    parser.add_argument("--supp-fig", type=int, help="Single supp figure number")
    args = parser.parse_args()

    apply_style()

    if args.figure:
        run_main([args.figure])
    elif args.supp_fig:
        run_supp([args.supp_fig])
    elif args.main:
        run_main()
    elif args.supp:
        run_supp()
    else:
        run_all()


if __name__ == "__main__":
    main()
