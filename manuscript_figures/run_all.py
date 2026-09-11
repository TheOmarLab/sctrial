#!/usr/bin/env python3
"""
Run all manuscript figure and table generation scripts.

Usage:
    python -m manuscript_figures.run_all              # everything
    python -m manuscript_figures.run_all --main       # main figures only
    python -m manuscript_figures.run_all --supp       # supplementary only
    python -m manuscript_figures.run_all --figure 2   # single main figure
    python -m manuscript_figures.run_all --supp-fig 3 # single supp figure
    python -m manuscript_figures.run_all --supp-tables # supplementary tables only
"""

from __future__ import annotations

import argparse
import time
import traceback

from ._shared import MAIN_OUTPUT, SUPP_OUTPUT, apply_style

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MAIN_FIGURES = {
    2: ("Figure 2: TNBC Immunotherapy Primary Analysis",                "main.figure2_tnbc_analysis"),
    3: ("Figure 3: Robustness & Benchmarking",                          "main.figure3_robustness_benchmarking"),
    4: ("Figure 4: Biological Discovery & Multi-Dataset Generalization", "main.figure4_biological_discovery_multi_dataset"),
    5: ("Figure 5: Validation & Dynamics",                              "main.figure5_validation_dynamics"),
}

SUPP_TABLES = {
    "tables": ("Supplementary Tables 1–6", "supp.supp_tables"),
}

SUPP_FIGURES = {
    1: ("Supp Fig 1: Data Quality and Cohort Characterisation",          "supp.supp_fig1_data_quality_cohort"),
    2: ("Supp Fig 2: Cell Annotation and Baseline Comparability",        "supp.supp_fig2_annotation_baseline"),
    3: ("Supp Fig 3: Pseudoreplication Bias & Melanoma/TNBC Analysis",   "supp.supp_fig3_melanoma_tnbc_analysis"),
    4: ("Supp Fig 4: Model Diagnostics and Assumption Checks",           "supp.supp_fig4_model_diagnostics"),
    5: ("Supp Fig 5: Sensitivity and Robustness",                        "supp.supp_fig5_sensitivity_robustness"),
    6: ("Supp Fig 6: Benchmark Calibration, Power and Robustness",        "supp.supp_fig6_benchmark_calibration"),
    7: ("Supp Fig 7: Cross-Dataset Biological Consistency",              "supp.supp_fig7_cross_dataset_biology"),
    8: ("Supp Fig 8: Heterogeneity and Temporal Dynamics",               "supp.supp_fig8_heterogeneity_temporal"),
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
    parser.add_argument("--supp-tables", action="store_true", help="Supplementary tables only")
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
    elif args.supp_tables:
        label, mod = SUPP_TABLES["tables"]
        _run(label, mod)
    else:
        run_all()


if __name__ == "__main__":
    main()
