#!/usr/bin/env python
"""
Generate all 10 alternative Panel B variants.

5 options × 2 datasets (Sade-Feldman v1 + CAR-T v2).

Run from the repo root:
    python -m manuscript_figures.main._generate_panel_b_alternatives
"""

from __future__ import annotations

import gc
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the package root is importable
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manuscript_figures._shared import apply_style, save_panel, MAIN_OUTPUT

# ---------- V1: Sade-Feldman (DiD) ----------
from manuscript_figures.main.figure5_biological_discovery import (
    _prepare_data as _prepare_v1,
    FIGURE_NAME as FIG5_NAME,
)
# ---------- V2: CAR-T (within-arm) ----------
from manuscript_figures.main.figure5v2_cart_biological_discovery import (
    _prepare_data as _prepare_v2,
    FIGURE_NAME as FIG5V2_NAME,
)
# ---------- Alternative panels ----------
from manuscript_figures.main._panel_b_alternatives import (
    panel_B1,
    panel_B2,
    panel_B3,
    panel_B4,
    panel_B5,
)

ALTERNATIVES = [
    ("B1_ridgeplot", panel_B1),
    ("B2_MA_plot", panel_B2),
    ("B3_running_ES", panel_B3),
    ("B4_pathway_network", panel_B4),
    ("B5_gene_waterfall", panel_B5),
]


def main():
    apply_style()

    # ── V1: Sade-Feldman ──────────────────────────────────────────────
    print("=" * 60)
    print("Preparing Sade-Feldman data (v1) ...")
    print("=" * 60)
    data_v1 = _prepare_v1()

    for panel_name, panel_func in ALTERNATIVES:
        print(f"  Generating {panel_name} (v1 — Sade-Feldman) ...")
        fig, ax = plt.subplots(figsize=(9, 6))
        try:
            panel_func(ax, data_v1, mode="did")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error: {exc}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="red")
        fig.tight_layout()
        save_panel(fig, f"panel_{panel_name}", FIG5_NAME, MAIN_OUTPUT)

    # Free V1 data
    del data_v1["adata"]
    del data_v1
    gc.collect()

    # ── V2: CAR-T ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Preparing CAR-T data (v2) ...")
    print("=" * 60)
    data_v2 = _prepare_v2()

    for panel_name, panel_func in ALTERNATIVES:
        print(f"  Generating {panel_name} (v2 — CAR-T) ...")
        fig, ax = plt.subplots(figsize=(9, 6))
        try:
            panel_func(ax, data_v2, mode="within")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()
            ax.text(0.5, 0.5, f"Error: {exc}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="red")
        fig.tight_layout()
        save_panel(fig, f"panel_{panel_name}", FIG5V2_NAME, MAIN_OUTPUT)

    # Free V2 data
    del data_v2["adata"]
    del data_v2
    gc.collect()

    print("\n✓ All 10 alternative Panel B variants generated.")


if __name__ == "__main__":
    main()
