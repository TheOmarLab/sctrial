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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
MAIN_OUTPUT = MANUSCRIPT_DIR / "main"
SUPP_OUTPUT = MANUSCRIPT_DIR / "supp"

for _d in (MAIN_OUTPUT, SUPP_OUTPUT):
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
    from sctrial import (
        TrialDesign,
        did_table,
        hedges_g,
        run_gsea_did,
        verify_paired_participants,
        loo_cv_did,
        within_arm_comparison,
        between_arm_comparison,
        add_log1p_cpm_layer,
    )
    SCTRIAL_AVAILABLE = True
except ImportError:
    SCTRIAL_AVAILABLE = False
    print("WARNING: sctrial not installed – some figures will be skipped.")

# ---------------------------------------------------------------------------
# Gene signatures & scoring
# ---------------------------------------------------------------------------

GENE_SIGNATURES = {
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

SIGNATURE_DISPLAY_NAMES = {
    # Full signature set (GENE_SIGNATURES keys)
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
    # Clinical signature set (CLINICAL_SIGNATURES keys) — mapped to
    # the same display names as their full-set counterparts so that
    # heatmaps and cross-dataset comparisons merge correctly.
    "Cytotoxic": "Cytotoxic T Cells",
    "Exhaustion": "T Cell Exhaustion",
    "Memory_T": "Memory T Cells",
    "IFN_response": "IFN Response",
    "Proliferation": "Cell Proliferation",
}

# Smaller signature set used by clinical-trial datasets
CLINICAL_SIGNATURES = {
    "Cytotoxic":      ["GZMB", "GZMA", "GZMK", "GZMH", "PRF1", "NKG7", "GNLY", "IFNG"],
    "Exhaustion":     ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "ENTPD1"],
    "Memory_T":       ["IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28"],
    "IFN_response":   ["ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "OAS1", "IRF7"],
    "Proliferation":  ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2"],
}


def sig_display(name: str) -> str:
    """Short display name for a signature column (``sig_*`` prefix stripped)."""
    clean = name.replace("sig_", "")
    return SIGNATURE_DISPLAY_NAMES.get(clean, clean)


def score_signatures(adata, *, layer=None, min_genes=3):
    """Score all 12 gene signatures, returning *(adata, sig_cols)*."""
    import scanpy as sc

    if layer is None:
        for candidate in ("log1p_tpm", "log1p_cpm", "counts"):
            if candidate in adata.layers:
                layer = candidate
                break

    print(f"  Scoring {len(GENE_SIGNATURES)} signatures (layer={layer})")
    sig_cols: list[str] = []
    for name, genes in GENE_SIGNATURES.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) >= min_genes:
            col = f"sig_{name}"
            try:
                sc.tl.score_genes(adata, available, score_name=col,
                                  use_raw=False, layer=layer)
                sig_cols.append(col)
            except Exception as exc:
                print(f"    Warning: could not score {name}: {exc}")
    return adata, sig_cols


def score_clinical_signatures(adata, *, layer=None, min_genes=3):
    """Score the smaller clinical-trial signature set."""
    import scanpy as sc

    sig_cols: list[str] = []
    for name, genes in CLINICAL_SIGNATURES.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) >= min_genes:
            col = f"sig_{name}"
            try:
                sc.tl.score_genes(adata, available, score_name=col,
                                  use_raw=False, layer=layer)
                sig_cols.append(col)
            except Exception as exc:
                print(f"    Warning: could not score {name}: {exc}")
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
        processed_name="sade_feldman_tpm_v5_full.h5ad",
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
    adata = load_stephenson_data(force_reprocess=False)
    print(f"  Stephenson: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["steph"] = adata
    return adata


def get_vaccine():
    """Vaccine GSE171964 (full, ~21 K cells)."""
    if "vax" in _DATA_CACHE:
        return _DATA_CACHE["vax"]
    from sctrial.datasets import load_vaccine_gse171964
    adata = load_vaccine_gse171964(
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


def _resolve_clinical_datasets_dir() -> str:
    """Locate clinical-trial dataset root."""
    env_dir = os.environ.get("SCTRIAL_CLINICAL_DATASETS_DIR")
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend([
        REPO_ROOT / "manuscript" / "datasets",          # sc-trialdiff/manuscript/datasets/
        SCRIPT_DIR / "datasets",
        PROJECT_DIR / "manuscript_figures" / "datasets",
        PROJECT_DIR.parent / "manuscript_figures" / "datasets",
    ])
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


CLINICAL_DATASETS_DIR = _resolve_clinical_datasets_dir()


def load_clinical_trial_dataset(name: str):
    """Load a clinical-trial dataset by short name (*aml*, *cart*, *melanoma*)."""
    import anndata as ad

    paths = {
        "aml":  os.path.join(CLINICAL_DATASETS_DIR, "GSE116256_AML",
                             "processed", "gse116256_aml_processed.h5ad"),
        "cart": os.path.join(CLINICAL_DATASETS_DIR, "GSE290722_CAR-T",
                             "processed", "gse290722_cart_processed.h5ad"),
        "melanoma": os.path.join(CLINICAL_DATASETS_DIR, "GSE115978_Melanoma",
                                 "processed", "gse115978_melanoma_processed.h5ad"),
    }
    if name not in paths:
        raise ValueError(f"Unknown clinical dataset: {name!r}")
    path = paths[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found at {path}")
    adata = ad.read_h5ad(path)
    print(f"  {name}: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    return adata


def harmonize_response(adata, *, force: bool = False):
    """Create ``response_harmonized`` column with consistent labels."""
    if force and "response_harmonized" in adata.obs.columns:
        del adata.obs["response_harmonized"]

    if "response_harmonized" in adata.obs.columns:
        if "participant_id" in adata.obs.columns:
            n_per = adata.obs.groupby("participant_id")["response_harmonized"].nunique()
            if (n_per > 1).any():
                pid_resp = adata.obs.groupby("participant_id")["response_harmonized"].agg(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
                )
                adata.obs["response_harmonized"] = adata.obs["participant_id"].map(pid_resp)
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

    if "response_harmonized" in adata.obs.columns and "participant_id" in adata.obs.columns:
        pid_resp = adata.obs.groupby("participant_id")["response_harmonized"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        )
        adata.obs["response_harmonized"] = adata.obs["participant_id"].map(pid_resp)

    return adata


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
