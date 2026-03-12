"""
Shared utilities for manuscript figure generation.

Provides data loading (with caching), gene-signature scoring,
style configuration, and I/O helpers used by all figure scripts.
"""

from __future__ import annotations

import gc
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

# Suppress only non-critical warnings; preserve inference/statistics warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()            # manuscript_figures/
PROJECT_DIR = SCRIPT_DIR.parent.resolve()               # sc_trial_inference/
REPO_ROOT = PROJECT_DIR.parent.resolve()         # sc-trialdiff/  (up from sctrial/)

# Figures are saved to sc-trialdiff/manuscript/{main,supp}/
MANUSCRIPT_DIR = PROJECT_DIR / "manuscript"
MAIN_OUTPUT = MANUSCRIPT_DIR / "main"
SUPP_OUTPUT = MANUSCRIPT_DIR / "supp"
DATA_DIR = PROJECT_DIR / "data"

for _d in (MAIN_OUTPUT, SUPP_OUTPUT, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLORS = {
    "treated":   "#4C72B0",
    "control":   "#E1812C",
    "neutral":   "#8172B3",
    "highlight":  "#C44E52",
    "success":   "#55A868",
    "gray":      "#8C8C8C",
}

def apply_style():
    """Set publication-quality matplotlib defaults (call once per script)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def despine(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_figure(fig, name: str, output_dir: Path, *, close: bool = True):
    """Save figure as both PNG and PDF at 600 DPI."""
    for fmt in ("png", "pdf"):
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(str(path), format=fmt, dpi=600, bbox_inches="tight",
                    facecolor="white")
    print(f"  Saved {name}")
    if close:
        plt.close(fig)

def save_panel(fig, panel_name: str, figure_name: str, output_dir: Path,
               *, close: bool = True):
    """Save a standalone panel as PNG."""
    panel_dir = output_dir / f"{figure_name}_panels"
    panel_dir.mkdir(exist_ok=True)
    path = panel_dir / f"{panel_name}.png"
    fig.savefig(str(path), format="png", dpi=600, bbox_inches="tight",
                facecolor="white")
    print(f"    Saved panel: {panel_name}")
    if close:
        plt.close(fig)

# ---------------------------------------------------------------------------
# sctrial imports
# ---------------------------------------------------------------------------

try:
    from sctrial import (  # noqa: F401 — re-exported for figure scripts
        TrialDesign,
        add_log1p_cpm_layer,
        between_arm_comparison,
        did_table,
        harmonize_response,
        hedges_g,
        load_aml,
        load_cart,
        loo_cv_did,
        run_gsea_did,
        verify_paired_participants,
        within_arm_comparison,
    )
    SCTRIAL_AVAILABLE = True
except ImportError:
    SCTRIAL_AVAILABLE = False
    print("WARNING: sctrial not installed – some figures will be skipped.")

# ---------------------------------------------------------------------------
# Gene signatures & scoring
# ---------------------------------------------------------------------------

GENE_SIGNATURES: dict[str, list[str]] = {
    "Cytotoxic T Cell Activity": [
        "GZMB", "GZMA", "GZMH", "GZMK", "PRF1", "GNLY", "NKG7", "KLRK1",
        "KLRD1", "FASLG", "IFNG",
    ],
    "Immune Exhaustion": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "TOX2",
        "ENTPD1", "CD244", "CD160", "BTLA",
    ],
    "Interferon Response": [
        "ISG15", "IFI6", "IFIT1", "IFIT2", "IFIT3", "MX1", "MX2",
        "OAS1", "OAS2", "OAS3", "STAT1", "IRF7", "IRF9",
    ],
    "Memory T Cell": [
        "IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28",
        "BCL2", "EOMES", "ID3",
    ],
    "T Cell Activation": [
        "CD69", "CD44", "IL2RA", "ICOS", "TNFRSF4", "TNFRSF9",
        "CD40LG", "HLA-DRA", "HLA-DRB1",
    ],
    "Inflammatory Response": [
        "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4",
        "NFKB1", "NLRP3", "CASP1",
    ],
    "Antigen Presentation": [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2",
        "PSMB8", "PSMB9", "CD74",
    ],
    "Cell Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2",
        "MCM2", "MCM7", "TYMS",
    ],
    "Regulatory T Cell": [
        "FOXP3", "IL2RA", "CTLA4", "TNFRSF18", "IKZF2", "IKZF4",
        "IL10", "TGFB1", "ENTPD1",
    ],
    "NK Cell Activity": [
        "NCAM1", "FCGR3A", "NCR1", "NCR3", "KLRF1", "KLRC1",
        "KIR2DL1", "KIR2DL3", "KIR3DL1",
    ],
    "Apoptosis": [
        "BCL2", "BAX", "BAK1", "CASP3", "CASP8", "CASP9",
        "FAS", "FASLG", "BID", "PARP1",
    ],
    "Oxidative Stress Response": [
        "NFE2L2", "HMOX1", "NQO1", "GCLC", "GCLM", "GSR",
        "SOD1", "SOD2", "CAT", "GPX1",
    ],
}

CLINICAL_SIGNATURES: dict[str, list[str]] = {
    "Cytotoxic": [
        "GZMB", "GZMA", "GZMK", "GZMH", "PRF1", "NKG7", "GNLY", "IFNG",
    ],
    "Exhaustion": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "ENTPD1",
    ],
    "Memory_T": [
        "IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28",
    ],
    "IFN_response": [
        "ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "OAS1", "IRF7",
    ],
    "Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2",
    ],
}

SIGNATURE_DISPLAY_NAMES: dict[str, str] = {
    "Cytotoxic T Cell Activity": "Cytotoxic T Cells",
    "Immune Exhaustion": "T Cell Exhaustion",
    "Interferon Response": "IFN Response",
    "Memory T Cell": "Memory T Cells",
    "T Cell Activation": "T Cell Activation",
    "Inflammatory Response": "Inflammation",
    "Antigen Presentation": "Antigen Presentation",
    "Cell Proliferation": "Cell Proliferation",
    "Regulatory T Cell": "Regulatory T Cells",
    "NK Cell Activity": "NK Cell Activity",
    "Apoptosis": "Apoptosis",
    "Oxidative Stress Response": "Oxidative Stress",
    "Cytotoxic": "Cytotoxic T Cells",
    "Exhaustion": "T Cell Exhaustion",
    "Memory_T": "Memory T Cells",
    "IFN_response": "IFN Response",
    "Proliferation": "Cell Proliferation",
}


def sig_display(name: str) -> str:
    """Return the short display name for a signature column."""
    clean = name.replace("sig_", "")
    return SIGNATURE_DISPLAY_NAMES.get(clean, clean)


def score_signatures(adata, *, layer=None, min_genes=3):
    """Score all 12 GENE_SIGNATURES using scanpy.tl.score_genes."""
    import scanpy as sc

    if layer is None:
        for candidate in ("log1p_tpm", "log1p_cpm", "counts"):
            if candidate in adata.layers:
                layer = candidate
                break

    sig_cols: list[str] = []
    for name, genes in GENE_SIGNATURES.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) >= min_genes:
            col = f"sig_{name}"
            try:
                sc.tl.score_genes(
                    adata, available, score_name=col,
                    use_raw=False, layer=layer,
                )
                sig_cols.append(col)
            except Exception as exc:
                warnings.warn(f"score_genes failed for {name}: {exc}", stacklevel=2)
    return adata, sig_cols


def score_clinical_signatures(adata, *, layer=None, min_genes=3):
    """Score the 5 CLINICAL_SIGNATURES using scanpy.tl.score_genes."""
    import scanpy as sc

    sig_cols: list[str] = []
    for name, genes in CLINICAL_SIGNATURES.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) >= min_genes:
            col = f"sig_{name}"
            try:
                sc.tl.score_genes(
                    adata, available, score_name=col,
                    use_raw=False, layer=layer,
                )
                sig_cols.append(col)
            except Exception as exc:
                warnings.warn(f"score_genes failed for {name}: {exc}", stacklevel=2)
    return adata, sig_cols


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

_DATA_CACHE: dict = {}


def clear_cache():
    """Free all cached datasets."""
    _DATA_CACHE.clear()
    gc.collect()


def get_sade_feldman():
    """Sade-Feldman immunotherapy (full, ~13 K cells)."""
    if "sf" in _DATA_CACHE:
        return _DATA_CACHE["sf"]
    from sctrial.datasets import load_sade_feldman
    adata = load_sade_feldman(
        data_dir=DATA_DIR,
        allow_download=True,
        max_cells_per_participant_visit=None,
        processed_name="sade_feldman_processed_v6.h5ad",
        force_reprocess=False,
    )
    print(f"  Sade-Feldman: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["sf"] = adata
    return adata


def get_stephenson():
    """Stephenson COVID-19 (~205 K cells)."""
    if "steph" in _DATA_CACHE:
        return _DATA_CACHE["steph"]
    from sctrial.datasets import load_stephenson_data
    adata = load_stephenson_data(data_dir=DATA_DIR, allow_download=True, force_reprocess=False)
    print(f"  Stephenson: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["steph"] = adata
    return adata


def get_vaccine():
    """Vaccine GSE171964 (full, ~21 K cells)."""
    if "vax" in _DATA_CACHE:
        return _DATA_CACHE["vax"]
    from sctrial.datasets import load_vaccine_gse171964
    adata = load_vaccine_gse171964(
        data_dir=DATA_DIR,
        allow_download=True,
        max_participants=None,
        max_cells_per_group=None,
        processed_name="vaccine_gse171964_day0_day7_full.h5ad",
        force_reprocess=False,
    )
    if "pt_id" in adata.obs.columns and "participant_id" not in adata.obs.columns:
        adata.obs["participant_id"] = adata.obs["pt_id"]
    if "day" in adata.obs.columns and "visit" not in adata.obs.columns:
        adata.obs["visit"] = adata.obs["day"].map({0: "Pre", 7: "Post"})
    print(f"  Vaccine: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["vax"] = adata
    return adata


def get_aml():
    """AML chemotherapy dataset (GSE116256, ~16 K cells)."""
    if "aml" in _DATA_CACHE:
        return _DATA_CACHE["aml"]
    from sctrial.datasets import load_aml
    adata = load_aml()
    print(f"  AML: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["aml"] = adata
    return adata


def get_cart():
    """CAR-T therapy dataset (GSE290722, ~44 K cells)."""
    if "cart" in _DATA_CACHE:
        return _DATA_CACHE["cart"]
    from sctrial.datasets import load_cart
    adata = load_cart()
    print(f"  CAR-T: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["cart"] = adata
    return adata


def load_clinical_trial_dataset(name: str):
    """Load a clinical-trial dataset by short name (*aml*, *cart*).

    .. deprecated::
        Use :func:`get_aml` or :func:`get_cart` directly instead.
    """
    _loaders = {"aml": get_aml, "cart": get_cart}
    if name not in _loaders:
        raise ValueError(f"Unknown clinical dataset: {name!r}")
    return _loaders[name]()


if not SCTRIAL_AVAILABLE:
    # Inline fallback only when sctrial is not installed.
    def harmonize_response(adata, *, force: bool = False):  # type: ignore[misc]
        """Create ``response_harmonized`` column (fallback implementation)."""
        if force and "response_harmonized" in adata.obs.columns:
            del adata.obs["response_harmonized"]

        if "response_harmonized" in adata.obs.columns:
            return adata

        mapping = {
            "responder": "Responder", "Responder": "Responder", "R": "Responder",
            "non-responder": "Non-responder", "Non-responder": "Non-responder",
            "NR": "Non-responder", "nonresponder": "Non-responder",
        }
        for col in ("response", "Response", "clinical_response"):
            if col in adata.obs.columns:
                adata.obs["response_harmonized"] = (
                    adata.obs[col].astype(str).map(lambda x: mapping.get(x, x))
                )
                break
        return adata


# ---------------------------------------------------------------------------
# Cell-type harmonization across datasets
# ---------------------------------------------------------------------------

# Maps every dataset-specific cell-type label to a shared coarse vocabulary.
# The canonical categories are:
#   CD4+ T, CD8+ T, Treg, T other, NK, B cell, Plasma,
#   Monocyte, DC, Erythroid, Platelet, HSC/Prog, Other
_CELLTYPE_MAP: dict[str, str] = {
    # -- Sade-Feldman (cell_type from marker-based annotation) --
    "CD8 T cell": "CD8+ T",
    "CD4 T cell": "CD4+ T",
    "Treg": "Treg",
    "NK cell": "NK",
    "B cell": "B cell",
    "Plasma cell": "Plasma",
    "Monocyte/Macrophage": "Monocyte",
    "Dendritic cell": "DC",
    "Unassigned": "Other",
    # -- Stephenson (celltype from full_clustering) --
    "CD4.CM": "CD4+ T",
    "CD4.EM": "CD4+ T",
    "CD4.IL22": "CD4+ T",
    "CD4.Naive": "CD4+ T",
    "CD4.Prolif": "CD4+ T",
    "CD4.Tfh": "CD4+ T",
    "CD4.Th1": "CD4+ T",
    "CD4.Th2": "CD4+ T",
    "CD4.Th17": "CD4+ T",
    "CD8.EM": "CD8+ T",
    "CD8.Naive": "CD8+ T",
    "CD8.Prolif": "CD8+ T",
    "CD8.TE": "CD8+ T",
    "MAIT": "T other",
    "gdT": "T other",
    "NKT": "T other",
    "NK_16hi": "NK",
    "NK_56hi": "NK",
    "NK_prolif": "NK",
    "B_naive": "B cell",
    "B_immature": "B cell",
    "B_exhausted": "B cell",
    "B_non-switched_memory": "B cell",
    "B_switched_memory": "B cell",
    "Plasma_cell_IgA": "Plasma",
    "Plasma_cell_IgG": "Plasma",
    "Plasma_cell_IgM": "Plasma",
    "Plasmablast": "Plasma",
    "CD14_mono": "Monocyte",
    "CD16_mono": "Monocyte",
    "CD83_CD14_mono": "Monocyte",
    "C1_CD16_mono": "Monocyte",
    "Mono_prolif": "Monocyte",
    "ASDC": "DC",
    "DC1": "DC",
    "DC2": "DC",
    "DC3": "DC",
    "DC_prolif": "DC",
    "pDC": "DC",
    "ILC1_3": "Other",
    "ILC2": "Other",
    "HSC_CD38neg": "HSC/Prog",
    "HSC_CD38pos": "HSC/Prog",
    "HSC_MK": "HSC/Prog",
    "HSC_erythroid": "Erythroid",
    "HSC_myeloid": "HSC/Prog",
    "HSC_prolif": "HSC/Prog",
    "Platelets": "Platelet",
    "RBC": "Erythroid",
    # -- Vaccine (clustnm: "C0_CD4 T", "C1_NK", ...) --
    "C0_CD4 T": "CD4+ T",
    "C6_CD8 T": "CD8+ T",
    "C10_Naive CD8 T": "CD8+ T",
    "C12_Tregs": "Treg",
    "C1_NK": "NK",
    "C16_NK T": "T other",
    "C5_B": "B cell",
    "C17_Naive B": "B cell",
    "C14_Plasmablasts": "Plasma",
    "C3_CD14+ monocytes": "Monocyte",
    "C4_CD16+ monocytes": "Monocyte",
    "C8_CD14+BDCA1+PD-L1+ cells": "Monocyte",
    "C7_cDC2": "DC",
    "C11_pDC": "DC",
    "C13_cDC1": "DC",
    "C9_Platelets": "Platelet",
    "C15_HPCs": "HSC/Prog",
    "C2": "Other",
    # -- AML (cell_type from van Galen CellType / PredictionRefined) --
    "HSC": "HSC/Prog",
    "Prog": "HSC/Prog",
    "GMP": "HSC/Prog",
    "GMP-like": "HSC/Prog",
    "ProMono": "HSC/Prog",
    "ProMono-like": "HSC/Prog",
    "HSC-like": "HSC/Prog",
    "Prog-like": "HSC/Prog",
    "CLP": "HSC/Prog",
    "MEP": "HSC/Prog",
    "T": "T other",
    "T_cell": "T other",
    "CTL": "CD8+ T",
    "B": "B cell",
    "B_cell": "B cell",
    "ProB": "B cell",
    "NK": "NK",
    "Mono": "Monocyte",
    "Monocyte": "Monocyte",
    "Macrophage": "Monocyte",
    "Myeloid": "Monocyte",
    "Neutrophil": "Monocyte",
    "cDC": "DC",
    "cDC-like": "DC",
    "Dendritic": "DC",
    "Plasma": "Plasma",
    "lateEry": "Erythroid",
    "earlyEry": "Erythroid",
    "Erythroid": "Erythroid",
    "Platelet": "Platelet",
    "Normal": "Other",
    "Malignant_AML": "Other",
    "Unknown": "Other",
    # -- CAR-T (cell_type from marker-based annotation) --
    "CD4+ T naive": "CD4+ T",
    "CD4+ T memory": "CD4+ T",
    "CD4+ T effector": "CD4+ T",
    "CD8+ T naive": "CD8+ T",
    "CD8+ T memory": "CD8+ T",
    "CD8+ T effector": "CD8+ T",
    "CD8+ T exhausted": "CD8+ T",
    "T proliferating": "T other",
    "gdT cell": "T other",
    "NK CD56bright": "NK",
    "NK CD56dim": "NK",
    "ILC": "Other",
    "Monocyte classical": "Monocyte",
    "Monocyte intermediate": "Monocyte",
    "Monocyte non-classical": "Monocyte",
    "cDC1": "DC",
    "cDC2": "DC",
    "B naive": "B cell",
    "B memory": "B cell",
    "HSC/MPP": "HSC/Prog",
}

# Canonical display order for harmonized cell types.
HARMONIZED_CELLTYPE_ORDER = [
    "CD4+ T", "CD8+ T", "Treg", "T other", "NK", "B cell", "Plasma",
    "Monocyte", "DC", "Erythroid", "Platelet", "HSC/Prog", "Other",
]


def harmonize_celltype(label: str) -> str:
    """Map a dataset-specific cell-type label to the shared vocabulary."""
    return _CELLTYPE_MAP.get(label, "Other")


def dfo_sort_key(label: str) -> tuple[int, int]:
    """Sort key for COVID-19 days-from-onset bins (``DFO_0-7``, ``DFO_15+``)."""
    s = str(label)
    m = re.match(r"DFO_(\d+)-(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"DFO_(\d+)\+", s)
    if m:
        return (int(m.group(1)), int(m.group(1)) + 1000)
    return (10_000, 10_000)
