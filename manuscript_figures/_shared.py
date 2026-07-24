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
REPO_ROOT = PROJECT_DIR.parent.parent.resolve()         # sc-trialdiff/  (up from sctrial/)

# Figures are saved to sc-trialdiff/manuscript/{main,supp}/
# GSEA results cached at sc-trialdiff/manuscript/GSEA/{dataset}/{library}/
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
MAIN_OUTPUT = MANUSCRIPT_DIR / "main"
SUPP_OUTPUT = MANUSCRIPT_DIR / "supp"
GSEA_OUTPUT = MANUSCRIPT_DIR / "GSEA"

for _d in (MAIN_OUTPUT, SUPP_OUTPUT, GSEA_OUTPUT):
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
        did_fit,
        did_table,
        get_did_aggregated_df,
        get_within_arm_aggregated_df,
        harmonize_response,
        hedges_g,
        load_aml,
        load_cart,
        loo_cv_did,
        run_gsea_cross_sectional,
        run_gsea_did,
        run_gsea_within_arm,
        verify_paired_participants,
        within_arm_comparison,
        within_arm_fit_beta,
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


# Canonical display names for datasets in figures/tables
DATASET_DISPLAY_NAMES: dict[str, str] = {
    "Sade-Feldman": "Melanoma",
    "Stephenson": "COVID-19",
}


def dataset_display(name: str) -> str:
    """Return the manuscript display name for a dataset."""
    return DATASET_DISPLAY_NAMES.get(name, name)


# Normalised expression layers, in preference order, for signature scoring.
# "counts" is deliberately ABSENT: scoring raw UMI counts differences two count
# means that both scale with sequencing depth, so a technical depth shift is
# reported as biological change. Falling back to it silently produced exactly
# that artifact for every dataset whose normalised layer is `log1p_norm`
# (vaccine, AML, CAR-T, TNBC), which was missing from this list entirely.
_SCORING_LAYERS = ("log1p_tpm", "log1p_cpm", "log1p_norm")


def _resolve_scoring_layer(adata, layer):
    """Pick a normalised layer, failing loudly rather than degrading to counts."""
    if layer is not None:
        if layer == "counts":
            raise ValueError(
                "Refusing to score signatures on the raw 'counts' layer: scores "
                "would track sequencing depth rather than biology. Pass a "
                f"normalised layer (one of {_SCORING_LAYERS}) or layer=None to "
                "auto-select."
            )
        return layer
    for candidate in _SCORING_LAYERS:
        if candidate in adata.layers:
            return candidate
    if adata.layers:
        raise KeyError(
            f"No normalised expression layer found. Looked for {_SCORING_LAYERS}; "
            f"available layers: {sorted(adata.layers)}. Normalise before scoring "
            "(e.g. add_log1p_cpm_layer) instead of scoring raw counts."
        )
    return None  # no layers at all -> adata.X, which loaders leave normalised


def score_signatures(adata, *, layer=None, min_genes=3):
    """Score all 12 GENE_SIGNATURES using scanpy.tl.score_genes."""
    import scanpy as sc

    layer = _resolve_scoring_layer(adata, layer)

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

    layer = _resolve_scoring_layer(adata, layer)

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
        max_cells_per_participant_visit=None,
        processed_name="sade_feldman_processed_v6.h5ad",
        force_reprocess=False,
    )
    print(f"  Sade-Feldman: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["sf"] = adata
    return adata

def get_tnbc_zhang():
    """Zhang TNBC chemotherapy dataset (GSE161529, ~22 K cells)."""
    if "tnbc" in _DATA_CACHE:
        return _DATA_CACHE["tnbc"]
    from sctrial.datasets import load_tnbc_zhang
    adata = load_tnbc_zhang(
        max_cells_per_participant_visit=None,
        processed_name="tnbc_zhang_processed.h5ad",
        force_reprocess=False,
        allow_download=True,
    )
    print(f"  TNBC (Zhang): {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["tnbc"] = adata
    return adata


def get_stephenson():
    """Stephenson COVID-19 (~205 K cells)."""
    if "steph" in _DATA_CACHE:
        return _DATA_CACHE["steph"]
    from sctrial.datasets import load_stephenson_data
    adata = load_stephenson_data(force_reprocess=False)
    print(f"  Stephenson: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["steph"] = adata
    return adata


def get_vaccine():
    """Vaccine GSE171964 (full, ~21 K cells)."""
    if "vax" in _DATA_CACHE:
        return _DATA_CACHE["vax"]
    from sctrial.datasets import load_vaccine_gse171964
    adata = load_vaccine_gse171964()
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
    "Mono-like": "Monocyte",
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
    # -- Sade-Feldman published G1-G11 (Cell 2018, Table S1 sheet
    #    "Gene marker-Fig1B-C" column headers, verbatim). The authors leave
    #    G5/G8/G10/G11 unresolved between CD4 and CD8, so they map to
    #    "T other" rather than being forced onto a lineage.
    "B cells": "B cell",
    "Plasma cells": "Plasma",
    "Monocytes/Macrophages": "Monocyte",
    "Dendritic cells": "DC",
    "Lymphocytes": "T other",
    "Exhausted CD8 T cells": "CD8+ T",
    "Regulatory T cells": "Treg",
    "Cytotoxicity (Lymphocytes)": "T other",
    "Exhausted/HS CD8 T cells": "CD8+ T",
    "Memory T cells": "T other",
    "Lymphocytes exhausted/cell-cycle": "T other",
    # -- TNBC lineage derived from Zhang 2021 97-cluster names --
    # ("CD4 T cell"/"CD8 T cell"/"ILC" are already mapped above.)
    "T cell (unresolved)": "T other",
    "Mast cell": "Other",
    # -- CAR-T published Major_Alias (Cheloni 2025, GSE290722 metadata) --
    "CD4 T Cells": "CD4+ T",
    "CD8 T Cells": "CD8+ T",
    "T regs": "Treg",
    "NK T Cells": "T other",
    "NK Cells": "NK",
    "NK Cells/ILC": "NK",
    "B Cells": "B cell",
    "Monocytes": "Monocyte",
    "pDCs": "DC",
    "pDCs/HSC/MPP": "HSC/Prog",
    "Erythrocytes": "Erythroid",
    "Mast Cells": "Other",
    "Doublets": "Other",
}

# Canonical display order for harmonized cell types.
HARMONIZED_CELLTYPE_ORDER = [
    "CD4+ T", "CD8+ T", "Treg", "T other", "NK", "B cell", "Plasma",
    "Monocyte", "DC", "Erythroid", "Platelet", "HSC/Prog", "Other",
]


_UNMAPPED_CELLTYPES: set[str] = set()


def harmonize_celltype(label: str) -> str:
    """Map a dataset-specific cell-type label to the shared vocabulary.

    An unmapped label silently becoming "Other" is how a whole published
    population disappears from a cross-dataset panel (this is exactly how AML's
    monocytic blasts, "Mono-like", were being dropped). Unmapped labels are
    therefore warned about once each, so a vocabulary gap surfaces immediately
    instead of quietly shrinking a lineage.
    """
    mapped = _CELLTYPE_MAP.get(label)
    if mapped is None:
        key = str(label)
        if key not in _UNMAPPED_CELLTYPES:
            _UNMAPPED_CELLTYPES.add(key)
            warnings.warn(
                f"Cell-type label {key!r} is not in _CELLTYPE_MAP and will be "
                "collapsed to 'Other'. Add it to the map if it is a real "
                "population.",
                stacklevel=2,
            )
        return "Other"
    return mapped


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


# ---------------------------------------------------------------------------
# GSEA helpers — load cached results or run fresh
# ---------------------------------------------------------------------------

GSEA_LIBRARIES: list[tuple[str, str]] = [
    ("MSigDB_Hallmark_2020", "Hallmark"),
    ("KEGG_2021_Human", "KEGG"),
    ("Reactome_2022", "Reactome"),
    ("GO_Biological_Process_2023", "GO_BP"),
    ("WikiPathways_2024_Human", "WikiPathways"),
]


def _gsea_cache_path(
    dataset: str, library_short: str, method: str = "did",
) -> Path:
    """Return ``manuscript/GSEA/{dataset}/{method}/{library}/results.csv``.

    Including *method* (``"did"``, ``"within_arm"``, ``"cross_sectional"``)
    prevents silently serving cached results from a different analysis type.
    """
    return GSEA_OUTPUT / dataset / method / library_short / "results.csv"


def load_or_run_gsea_did(
    adata,
    design,
    visits: tuple[str, str],
    layer: str | None,
    dataset_name: str,
    *,
    force: bool = False,
) -> object:
    """Load cached GSEA results for *dataset_name* or run fresh.

    Results are saved per-library under ``manuscript/GSEA/{dataset}/{lib}/``.
    If cached CSV files exist and *force* is False, they are loaded directly.
    Otherwise ``run_gsea_did`` is called for each library and results are saved.

    Returns a combined DataFrame (all libraries) sorted by |NES|, or None.
    """
    import pandas as pd

    try:
        import gseapy as gp  # noqa: F401

        from sctrial import run_gsea_did as _run_gsea
    except ImportError:
        print(f"    {dataset_name}: sctrial/gseapy not available — skipping GSEA")
        return None

    frames: list[pd.DataFrame] = []
    for lib_name, short_name in GSEA_LIBRARIES:
        cache_csv = _gsea_cache_path(dataset_name, short_name, method="did")

        # Try loading from cache
        if not force and cache_csv.exists():
            df = pd.read_csv(cache_csv)
            if len(df) > 0:
                df["Library"] = short_name
                frames.append(df)
                print(f"    {dataset_name}/{short_name}: {len(df)} pathways (cached)")
                continue

        # Run fresh
        outdir = str(cache_csv.parent)
        os.makedirs(outdir, exist_ok=True)
        try:
            res = _run_gsea(
                adata, gene_sets=lib_name, design=design, visits=visits,
                layer=layer, rank_by="tstat",
                min_size=10, max_size=500, permutation_num=1000,
                outdir=outdir, no_plot=True,
            )
            if isinstance(res, pd.DataFrame) and len(res) > 0:
                res["Library"] = short_name
                res.to_csv(cache_csv, index=False)
                frames.append(res)
                print(f"    {dataset_name}/{short_name}: {len(res)} pathways")
            else:
                print(f"    {dataset_name}/{short_name}: no results")
        except Exception as exc:
            print(f"    {dataset_name}/{short_name}: {exc}")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "NES" in combined.columns:
        combined = combined.sort_values("NES", key=abs, ascending=False)
    return combined


def load_or_run_gsea_cross_sectional(
    adata,
    design,
    visit: str,
    layer: str | None,
    dataset_name: str,
    *,
    force: bool = False,
) -> object:
    """Load cached GSEA results for cross-sectional design or run fresh.

    Uses run_gsea_cross_sectional (between_arm_comparison) for single-visit
    designs. Results are saved per-library under manuscript/GSEA/{dataset}/{lib}/.

    Returns a combined DataFrame (all libraries) sorted by |NES|, or None.
    """
    import pandas as pd

    try:
        import gseapy as gp  # noqa: F401

        from sctrial import run_gsea_cross_sectional as _run_gsea
    except ImportError:
        print(f"    {dataset_name}: sctrial/gseapy not available — skipping GSEA")
        return None

    frames: list[pd.DataFrame] = []
    for lib_name, short_name in GSEA_LIBRARIES:
        cache_csv = _gsea_cache_path(dataset_name, short_name, method="cross_sectional")

        if not force and cache_csv.exists():
            df = pd.read_csv(cache_csv)
            if len(df) > 0:
                df["Library"] = short_name
                frames.append(df)
                print(f"    {dataset_name}/{short_name}: {len(df)} pathways (cached)")
                continue

        outdir = str(cache_csv.parent)
        os.makedirs(outdir, exist_ok=True)
        try:
            res = _run_gsea(
                adata, gene_sets=lib_name, design=design, visit=visit,
                layer=layer, rank_by="tstat",
                min_size=10, max_size=500, permutation_num=1000,
                outdir=outdir, no_plot=True,
            )
            if isinstance(res, pd.DataFrame) and len(res) > 0:
                res["Library"] = short_name
                res.to_csv(cache_csv, index=False)
                frames.append(res)
                print(f"    {dataset_name}/{short_name}: {len(res)} pathways")
            else:
                print(f"    {dataset_name}/{short_name}: no results")
        except Exception as exc:
            print(f"    {dataset_name}/{short_name}: {exc}")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "NES" in combined.columns:
        combined = combined.sort_values("NES", key=abs, ascending=False)
    return combined


def load_or_run_gsea_within_arm(
    adata,
    design,
    arm: str,
    visits: tuple[str, str],
    layer: str | None,
    dataset_name: str,
    *,
    force: bool = False,
) -> object:
    """Load cached GSEA results for within-arm design or run fresh.

    Uses run_gsea_within_arm (within_arm_comparison) for longitudinal
    single-arm designs. Results are saved per-library under
    manuscript/GSEA/{dataset}/{lib}/.

    Returns a combined DataFrame (all libraries) sorted by |NES|, or None.
    """
    import pandas as pd

    try:
        import gseapy as gp  # noqa: F401

        from sctrial import run_gsea_within_arm as _run_gsea
    except ImportError:
        print(f"    {dataset_name}: sctrial/gseapy not available — skipping GSEA")
        return None

    frames: list[pd.DataFrame] = []
    for lib_name, short_name in GSEA_LIBRARIES:
        cache_csv = _gsea_cache_path(dataset_name, short_name, method="within_arm")

        if not force and cache_csv.exists():
            df = pd.read_csv(cache_csv)
            if len(df) > 0:
                df["Library"] = short_name
                frames.append(df)
                print(f"    {dataset_name}/{short_name}: {len(df)} pathways (cached)")
                continue

        outdir = str(cache_csv.parent)
        os.makedirs(outdir, exist_ok=True)
        try:
            res = _run_gsea(
                adata, gene_sets=lib_name, design=design, arm=arm, visits=visits,
                layer=layer, rank_by="tstat",
                min_size=10, max_size=500, permutation_num=1000,
                outdir=outdir, no_plot=True,
            )
            if isinstance(res, pd.DataFrame) and len(res) > 0:
                res["Library"] = short_name
                res.to_csv(cache_csv, index=False)
                frames.append(res)
                print(f"    {dataset_name}/{short_name}: {len(res)} pathways")
            else:
                print(f"    {dataset_name}/{short_name}: no results")
        except Exception as exc:
            print(f"    {dataset_name}/{short_name}: {exc}")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "NES" in combined.columns:
        combined = combined.sort_values("NES", key=abs, ascending=False)
    return combined


def load_or_run_gsea_prerank(
    ranking,
    dataset_name: str,
    *,
    force: bool = False,
) -> object:
    """Load cached GSEA results or run ``gseapy.prerank`` from a ranking.

    Used for cross-sectional designs (e.g. Stephenson) where ``run_gsea_did``
    is not applicable.  *ranking* is a pre-computed gene-level statistic
    (e.g. Welch t-stat) indexed by gene name.
    """
    import pandas as pd

    try:
        import gseapy as gp
    except ImportError:
        print(f"    {dataset_name}: gseapy not available — skipping GSEA")
        return None

    frames: list[pd.DataFrame] = []
    for lib_name, short_name in GSEA_LIBRARIES:
        cache_csv = _gsea_cache_path(dataset_name, short_name, method="prerank")

        if not force and cache_csv.exists():
            df = pd.read_csv(cache_csv)
            if len(df) > 0:
                df["Library"] = short_name
                frames.append(df)
                print(f"    {dataset_name}/{short_name}: {len(df)} pathways (cached)")
                continue

        outdir = str(cache_csv.parent)
        os.makedirs(outdir, exist_ok=True)
        try:
            pre_res = gp.prerank(
                rnk=ranking, gene_sets=lib_name,
                min_size=10, max_size=500, permutation_num=1000,
                outdir=outdir, no_plot=True,
            )
            res_df = pre_res.res2d if hasattr(pre_res, "res2d") else pre_res
            if isinstance(res_df, pd.DataFrame) and len(res_df) > 0:
                res_df["Library"] = short_name
                res_df.to_csv(cache_csv, index=False)
                frames.append(res_df)
                print(f"    {dataset_name}/{short_name}: {len(res_df)} pathways")
            else:
                print(f"    {dataset_name}/{short_name}: no results")
        except Exception as exc:
            print(f"    {dataset_name}/{short_name}: {exc}")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "NES" in combined.columns:
        combined = combined.sort_values("NES", key=abs, ascending=False)
    return combined
