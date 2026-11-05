"""
Shared utilities for manuscript figure generation.

Provides data loading (with caching), gene-signature scoring,
style configuration, and I/O helpers used by all figure scripts.
"""

from __future__ import annotations

import gc
import hashlib
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
#
# The default layout assumes the local checkout (sc-trialdiff/sctrial/
# sc_trial_inference). On the HPC the repo is checked out one level shallower,
# so REPO_ROOT resolves wrong; SCTRIAL_MANUSCRIPT_DIR overrides the output root
# without touching the local default. Falls back to <project>/manuscript when
# the computed default does not exist but a sibling one does.
_env_manuscript = os.environ.get("SCTRIAL_MANUSCRIPT_DIR")
if _env_manuscript:
    MANUSCRIPT_DIR = Path(_env_manuscript).resolve()
elif (REPO_ROOT / "manuscript").exists() or not (PROJECT_DIR / "manuscript").exists():
    MANUSCRIPT_DIR = REPO_ROOT / "manuscript"
else:
    MANUSCRIPT_DIR = PROJECT_DIR / "manuscript"
MAIN_OUTPUT = MANUSCRIPT_DIR / "main"
SUPP_OUTPUT = MANUSCRIPT_DIR / "supp"
GSEA_OUTPUT = MANUSCRIPT_DIR / "GSEA"

for _d in (MAIN_OUTPUT, SUPP_OUTPUT, GSEA_OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)


def _stats_fingerprint() -> str:
    """Short hash of the package source that determines every DiD estimate.

    Fold this into the key of any cache whose content comes from ``did_fit``,
    ``did_table`` or DiD-ranked GSEA. A figure's own ``_CODE_VERSION`` or source
    hash cannot see a change to the package estimator, so without this a cache
    keeps serving results from the previous estimator, on every machine that
    holds one.
    """
    import sctrial

    pkg = Path(sctrial.__file__).resolve().parent
    h = hashlib.md5()  # noqa: S324 - cache tag, not security
    for rel in ("stats/did.py", "utils.py"):
        h.update((pkg / rel).read_bytes())
    return h.hexdigest()[:8]


STATS_FINGERPRINT = _stats_fingerprint()

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
        between_arm_fit_beta,
        did_fit,
        did_table,
        get_between_arm_aggregated_df,
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

# Literature-grounded, adversarially-verified module gene sets (2026-07-31).
# Each list was curated against MSigDB (Hallmark/Reactome/KEGG) and the dataset
# source papers, then skeptically verified (unsupported/dropout/cross-lineage
# genes cut). See manuscript/revised_module_signatures_2026-07-31.md for the
# per-gene rationale and citations. Scored with scanpy score_genes (see
# score_signatures); signatures may share a few effector genes by design and the
# overlaps are documented (e.g. NK/cytotoxic share KLRD1/GNLY/NKG7).
GENE_SIGNATURES: dict[str, list[str]] = {
    "Cytotoxic T Cell Activity": [
        "GZMB", "GZMA", "GZMH", "GZMK", "GZMM", "PRF1", "GNLY", "NKG7",
        "KLRK1", "KLRD1", "KLRG1", "FASLG", "IFNG", "CST7", "CCL5",
        "CX3CR1", "FGFBP2",
    ],
    "T Cell Exhaustion": [  # terminal exhaustion program
        "TOX", "TOX2", "PDCD1", "HAVCR2", "LAG3", "TIGIT", "ENTPD1",
        "CXCL13", "LAYN", "CD38", "PRDM1", "BATF", "NR4A2", "NR4A3",
        "CD160", "CD244",
    ],
    "Checkpoint Receptors": [  # acute inhibitory-receptor panel
        "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "BTLA", "CD160",
        "CD244", "CD96",
    ],
    "Type I Interferon": [
        "ISG15", "IFI27", "IFI44", "IFI44L", "IFI35", "MX1", "OAS1",
        "OASL", "IFIT2", "IFIT3", "IFITM3", "RSAD2", "LY6E", "USP18",
        "IRF7", "STAT2", "ISG20", "BST2", "HERC6", "EPSTI1", "SAMD9L",
        "PARP9", "RTP4", "LGALS3BP", "CMPK2",
    ],
    "Type II Interferon": [  # IFN-gamma response
        "STAT1", "IRF1", "GBP1", "GBP2", "GBP4", "GBP5", "CXCL9",
        "CXCL10", "CXCL11", "IDO1", "SOCS1", "SOCS3", "JAK2", "UBE2L6",
        "WARS", "IL18BP", "VCAM1", "SERPING1", "FGL2", "IFI30",
    ],
    "Memory T Cell": [
        "IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28", "ID3",
        "REL", "FOXP1", "BACH2", "MYB", "FOXO1", "KLF2", "S1PR1",
    ],
    "T Cell Activation": [
        "CD69", "CD44", "IL2RA", "ICOS", "TNFRSF4", "TNFRSF9", "CD40LG",
        "EGR1", "EGR2", "EGR3", "NR4A1", "MYC", "IL2RB", "IRF4",
        "TNFRSF18",
    ],
    "Inflammatory Response": [
        "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4", "NFKB1",
        "NLRP3", "CASP1", "CXCL1", "CXCL2", "CXCL3", "CCL20", "PTGS2",
        "IL1A", "TNFAIP3", "NFKBIA", "IL18", "S100A8", "S100A9",
    ],
    "Antigen Presentation": [
        "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP",
        "NLRC5", "PSMB8", "PSMB9", "PSMB10", "CIITA", "CD74", "HLA-DRA",
        "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DMA",
        "HLA-DMB", "CTSS",
    ],
    "Cell Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2", "MCM2",
        "MCM7", "TYMS", "BIRC5", "UBE2C", "CENPF", "CENPE", "CCNB2",
        "RRM2", "HMGB2", "TPX2", "NUSAP1", "CKS2", "AURKA", "CKS1B",
        "STMN1",
    ],
    "Regulatory T Cell": [
        "FOXP3", "IKZF2", "IKZF4", "IL2RA", "CTLA4", "TNFRSF18",
        "TNFRSF4", "TNFRSF9", "TIGIT", "ENTPD1", "LRRC32", "RTKN2",
        "CCR8", "LAYN", "IL1R2",
    ],
    "NK Cell Activity": [
        "FCGR3A", "NCR1", "NCR3", "NCR2", "KLRF1", "KLRC1", "KLRC2",
        "KLRD1", "KLRB1", "IL2RB", "TBX21", "GZMM", "SH2D1B", "GNLY",
        "NKG7",
    ],
    "Apoptosis": [
        "BAX", "BAK1", "BID", "BCL2L11", "BBC3", "PMAIP1", "CASP3",
        "CASP7", "CASP8", "CASP9", "FAS", "TNFRSF10B", "DIABLO", "APAF1",
        "CDKN1A", "GADD45A", "GADD45B", "DDIT3", "FDXR", "BTG2",
    ],
    "Oxidative Stress Response": [
        "NFE2L2", "HMOX1", "NQO1", "GCLC", "GCLM", "GSR", "SOD1", "SOD2",
        "CAT", "GPX1", "TXN", "TXNRD1", "PRDX1", "PRDX2", "SRXN1",
        "G6PD", "GPX4", "GLRX", "SLC7A11",
    ],
    "Humoral Plasma Cell": [
        "MS4A1", "CD79A", "CD79B", "CD19", "CD37", "BANK1", "TCL1A",
        "FCRL5", "POU2AF1", "MZB1", "JCHAIN", "XBP1", "PRDM1", "DERL3",
        "SDC1", "TNFRSF17", "SEC11C", "HERPUD1", "FKBP11", "SLAMF7",
        "IGHM", "IGHG1", "IGKC", "IGLC2", "FDCSP",
    ],
    "Monocyte Macrophage": [
        "CD14", "CD68", "LYZ", "CSF1R", "AIF1", "TYROBP", "FCER1G",
        "CTSS", "FCGR3A", "ITGAM", "MNDA", "CD163", "MRC1", "MSR1",
        "MARCO", "APOE", "C1QA", "C1QB", "C1QC", "TREM2", "SELENOP",
        "LGALS3", "FCN1", "VCAN", "S100A9",
    ],
    "Tissue-Resident Memory": [
        "ITGAE", "ITGA1", "CXCR6", "ZNF683", "CD101", "RGS1", "RGS2",
        "DUSP6", "RBPJ", "CD69", "ID2", "PRDM1", "NR4A1", "DUSP1",
        "EGR1",
    ],
    "T Follicular Helper": [
        "CXCL13", "CXCR5", "PDCD1", "ICOS", "BCL6", "TOX", "TOX2",
        "IL21", "BTLA", "CD200", "MAF", "ASCL2", "SH2D1A", "CD40LG",
        "TNFRSF4", "BATF",
    ],
    "Hypoxia": [
        "VEGFA", "SLC2A1", "SLC2A3", "HK1", "HK2", "PGK1", "LDHA",
        "ALDOA", "ENO1", "PDK1", "BNIP3L", "NDRG1", "ADM", "ANGPTL4",
        "DDIT4", "PPP1R15A", "ERO1A", "P4HA1", "CA12", "IGFBP3", "PLIN2",
    ],
}

CLINICAL_SIGNATURES: dict[str, list[str]] = {
    "Cytotoxic": [
        "GZMB", "GZMA", "GZMK", "GZMH", "PRF1", "NKG7", "GNLY", "IFNG",
    ],
    "Exhaustion": [
        "TOX", "PDCD1", "LAG3", "HAVCR2", "TIGIT", "ENTPD1", "CXCL13",
    ],
    "Memory_T": [
        "IL7R", "TCF7", "LEF1", "CCR7", "SELL", "CD27", "CD28",
    ],
    "IFN_response": [
        "ISG15", "IFI27", "IFIT3", "MX1", "OAS1", "IRF7", "ISG20",
    ],
    "Proliferation": [
        "MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNA2",
    ],
}

SIGNATURE_DISPLAY_NAMES: dict[str, str] = {
    "Cytotoxic T Cell Activity": "Cytotoxic T Cells",
    "T Cell Exhaustion": "T Cell Exhaustion",
    "Checkpoint Receptors": "Checkpoint Receptors",
    "Type I Interferon": "Type I IFN",
    "Type II Interferon": "Type II IFN",
    "Memory T Cell": "Memory T Cells",
    "T Cell Activation": "T Cell Activation",
    "Inflammatory Response": "Inflammation",
    "Antigen Presentation": "Antigen Presentation",
    "Cell Proliferation": "Cell Proliferation",
    "Regulatory T Cell": "Regulatory T Cells",
    "NK Cell Activity": "NK Cell Activity",
    "Apoptosis": "Apoptosis",
    "Oxidative Stress Response": "Oxidative Stress",
    "Humoral Plasma Cell": "Humoral/Plasma",
    "Monocyte Macrophage": "Monocyte/Macrophage",
    "Tissue-Resident Memory": "Tissue-Res. Memory",
    "T Follicular Helper": "Tfh/Tph",
    "Hypoxia": "Hypoxia",
    "Cytotoxic": "Cytotoxic T Cells",
    "Exhaustion": "T Cell Exhaustion",
    "Memory_T": "Memory T Cells",
    "IFN_response": "Type I IFN",
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


# ---------------------------------------------------------------------------
# Text labels for scatter plots
# ---------------------------------------------------------------------------

# adjustText settings, gentlest first. Every candidate is laid out and scored on
# its defects (labels touching a label or a marker, leader lines crossing), with
# mean leader length breaking ties; the lowest score wins. Several candidates are
# needed because convergence depends on the font metrics of the rendering
# machine. Crossing removal is off in the stronger settings: it swaps label pairs
# every step and the layout oscillates instead of converging.
_LABEL_LAYOUTS = (
    dict(expand=(1.2, 1.4), force_text=(0.3, 0.5), force_static=(0.5, 0.7)),
    dict(expand=(1.2, 1.4), force_text=(0.6, 0.9), force_static=(0.8, 1.0),
         prevent_crossings=False, max_move=(20, 20)),
    dict(expand=(1.2, 1.4), force_text=(0.6, 0.9), force_static=(1.0, 1.2),
         force_pull=(0.0, 0.0), prevent_crossings=False, max_move=(20, 20)),
    dict(expand=(1.1, 1.3), force_text=(0.8, 1.0), force_static=(1.0, 1.2),
         prevent_crossings=False, max_move=(40, 40)),
    dict(expand=(1.5, 1.8), force_text=(0.5, 0.8), force_static=(0.8, 1.0),
         prevent_crossings=False, max_move=(20, 20)),
)


def _segments_cross(p1, p2, q1, q2) -> bool:
    """True when segments p1-p2 and q1-q2 properly intersect."""
    import numpy as np

    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return bool(orient(p1, p2, q1) * orient(p1, p2, q2) < 0
                and orient(q1, q2, p1) * orient(q1, q2, p2) < 0)


def _leaders(boxes, target_px, target_r_px):
    """Per label: (label centre x, y, target x, y, gap); centre is None when attached."""
    import numpy as np

    out = []
    for bb, (mx, my), r in zip(boxes, target_px, target_r_px):
        gap = float(np.hypot(np.clip(mx, bb.x0, bb.x1) - mx, np.clip(my, bb.y0, bb.y1) - my))
        out.append(((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2, mx, my, gap)
                   if gap > r + 2 else (None, None, mx, my, gap))
    return out


def _crossing_pairs(leaders):
    return [(i, j) for i, a in enumerate(leaders) for j, b in enumerate(leaders)
            if i < j and a[0] is not None and b[0] is not None
            and _segments_cross(a[:2], a[2:4], b[:2], b[2:4])]


def label_layout_score(texts, target_px, target_r_px, obstacle_px, obstacle_r_px,
                       renderer, blocked_px=(), own=None,
                       own_slack_px: float = 0.0) -> tuple[int, float]:
    """(defects, mean label-to-target gap in px) for a finished label layout.

    Defects are label-label overlaps, labels touching any obstacle disc or any
    blocked display-space box (e.g. a reference line), and crossing leader lines
    (label centre to its target, counted when the label is detached). ``own``
    gives each label's own marker as an index into the obstacles; that disc is
    shrunk by ``own_slack_px`` so a label may sit right beside its own point.
    """
    import numpy as np

    boxes = [t.get_window_extent(renderer) for t in texts]
    defects = sum(boxes[i].overlaps(boxes[j])
                  for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    defects += sum(bb.overlaps(blk) for bb in boxes for blk in blocked_px)
    for k, bb in enumerate(boxes):
        radii = np.array(obstacle_r_px, dtype=float, copy=True)
        if own is not None:
            radii[own[k]] -= own_slack_px
        cx = np.clip(obstacle_px[:, 0], bb.x0, bb.x1)
        cy = np.clip(obstacle_px[:, 1], bb.y0, bb.y1)
        defects += int(np.any((cx - obstacle_px[:, 0]) ** 2 + (cy - obstacle_px[:, 1]) ** 2
                              < radii ** 2))
    leaders = _leaders(boxes, target_px, target_r_px)
    defects += len(_crossing_pairs(leaders))
    return defects, float(np.mean([seg[4] for seg in leaders]))


def label_points(ax, names, x, y, s, *, fontsize: float, obstacles=None,
                 hlines=(), vlines=(), priority=None, counts=None,
                 max_mean_leader_pt: float | None = None, fontweight: str = "bold",
                 color: str = "black") -> list[str]:
    """Label scatter points clear of each other and of markers, with leader lines.

    ``names``, ``x``, ``y`` are the points to label (data coordinates) and ``s`` their
    marker area in pt^2 (scalar or one per point), as passed to ``scatter``.
    ``obstacles`` is an optional ``(xy, s)`` pair of further markers no label may
    touch; the labelled points are always obstacles. ``hlines``/``vlines`` are
    data-coordinate reference lines (e.g. a significance threshold) no label may
    sit on. ``priority`` (indices into ``names``, most important first) and
    ``counts`` (label counts to try, largest first) make the function label the top
    ``n`` points for the first ``n`` whose layout has no defect and, when
    ``max_mean_leader_pt`` is set, a mean label-to-point gap no longer than that:
    the right behaviour for a cell too small to name every point. Returns the
    names that were labelled.

    Each candidate setting in ``_LABEL_LAYOUTS`` is laid out by adjustText against
    data-space boxes around every marker (marker radius plus 1 pt), with a fixed
    iteration budget and numpy's global RNG seeded (adjustText jitters coincident
    labels with it), then crossing leaders are swapped while that lowers the
    defect score; the best layout is kept. The canvas is drawn first so marker
    sizes are measured at the figure dpi. Axis limits, the axes position and the
    label font size must be final before calling: adjustText works in display
    space, so a later resize moves markers under labels.
    """
    import numpy as np
    from adjustText import adjust_text
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.transforms import Bbox

    names = list(names)
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    size = np.broadcast_to(np.asarray(s, dtype=float), (len(names),))
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = fig.dpi / 72

    def _discs(points, areas):
        px = ax.transData.transform(points)
        return px, (np.sqrt(areas) / 2 + 1.0) * to_px

    if obstacles is not None:
        obs_xy = np.asarray(obstacles[0], dtype=float).reshape(-1, 2)
        obs_s = np.broadcast_to(np.asarray(obstacles[1], dtype=float), (len(obs_xy),))
        # A labelled point passed again as an obstacle would block its own label.
        dup = np.array([np.any(np.all(np.isclose(o, xy), axis=1)) for o in obs_xy],
                       dtype=bool)
        all_xy = np.vstack([xy, obs_xy[~dup]])
        all_s = np.concatenate([size, obs_s[~dup]])
    else:
        all_xy, all_s = xy, size
    obstacle_px, obstacle_r = _discs(all_xy, all_s)
    # adjustText repels from each marker's own extent; the 1 pt clearance applies
    # only when scoring (padding the solver's boxes stalls it in dense clusters).
    marker_r = np.sqrt(all_s) / 2 * to_px
    inv = ax.transData.inverted()
    lo = inv.transform(obstacle_px - marker_r[:, None])
    hi = inv.transform(obstacle_px + marker_r[:, None])
    boxes = [Bbox([np.minimum(a, b), np.maximum(a, b)]) for a, b in zip(lo, hi)]
    # Reference lines become thin display-space strips (1 pt either side), and the
    # four frame edges become strips just inside the axes, so no label sits on a
    # line it does not describe or touches a spine.
    frame = ax.get_window_extent(renderer)
    half = 1.0 * to_px
    edge = 1.5 * to_px
    blocked_px = [
        Bbox([[frame.x0, frame.y0], [frame.x0 + edge, frame.y1]]),
        Bbox([[frame.x1 - edge, frame.y0], [frame.x1, frame.y1]]),
        Bbox([[frame.x0, frame.y0], [frame.x1, frame.y0 + edge]]),
        Bbox([[frame.x0, frame.y1 - edge], [frame.x1, frame.y1]]),
    ]
    for yv in hlines:
        py = ax.transData.transform((0.0, yv))[1]
        blocked_px.append(Bbox([[frame.x0, py - half], [frame.x1, py + half]]))
    for xv in vlines:
        px_ = ax.transData.transform((xv, 0.0))[0]
        blocked_px.append(Bbox([[px_ - half, frame.y0], [px_ + half, frame.y1]]))
    # Each setting runs twice, with and without the frame strips among the solver's
    # obstacles (ensure_inside_axes alone lets labels rest on a spine, but the strips
    # also perturb adjustText's initial explode step); scoring always includes them.
    line_boxes = [Bbox(inv.transform(b.get_points())) for b in blocked_px[4:]]
    frame_boxes = [Bbox(inv.transform(b.get_points())) for b in blocked_px[:4]]

    order = list(priority) if priority is not None else list(range(len(names)))
    tries = list(counts) if counts is not None else [len(names)]
    tries = [n for n in tries if n <= len(names)] or [len(names)]

    def _place(keep):
        target_px, target_r = _discs(xy[keep], size[keep])
        best = None
        for settings, with_frame in ((s_, f_) for s_ in _LABEL_LAYOUTS for f_ in (True, False)):
            texts = [ax.text(xy[i, 0], xy[i, 1], names[i], fontsize=fontsize,
                             fontweight=fontweight, color=color, zorder=6) for i in keep]
            objects = boxes + line_boxes + (frame_boxes if with_frame else [])
            rng_state = np.random.get_state()
            np.random.seed(0)
            try:
                adjust_text(texts, objects=objects, ax=ax, iter_lim=3000,
                            ensure_inside_axes=True, **settings)
            finally:
                np.random.set_state(rng_state)
            args = (target_px, target_r, obstacle_px, obstacle_r, renderer, blocked_px,
                    keep, 1.0 * to_px)
            score = label_layout_score(texts, *args)
            for _ in range(len(texts)):
                improved = False
                cur = [t.get_window_extent(renderer) for t in texts]
                for i, j in _crossing_pairs(_leaders(cur, target_px, target_r)):
                    pi, pj = texts[i].get_position(), texts[j].get_position()
                    texts[i].set_position(pj)
                    texts[j].set_position(pi)
                    new = label_layout_score(texts, *args)
                    if new < score:
                        score, improved = new, True
                        break
                    texts[i].set_position(pi)
                    texts[j].set_position(pj)
                if not improved:
                    break
            if best is None or score < best[0]:
                if best is not None:
                    for t in best[1]:
                        t.remove()
                best = (score, texts)
            else:
                for t in texts:
                    t.remove()
        return best

    # Accept the largest count whose layout is defect-free with short leaders; if
    # none is, fall back to the fewest defects (then more labels, shorter leaders).
    max_gap_px = None if max_mean_leader_pt is None else max_mean_leader_pt * to_px
    attempts, chosen = [], None
    for n in tries:
        keep = sorted(order[:n])
        (defects, mean_gap), texts = _place(keep)
        attempts.append(((defects, -n, mean_gap), keep, [t.get_position() for t in texts]))
        for t in texts:
            t.remove()
        if defects == 0 and (max_gap_px is None or mean_gap <= max_gap_px):
            chosen = attempts[-1]
            break
    _, keep, positions = chosen if chosen is not None else min(attempts, key=lambda a: a[0])
    texts = [ax.text(px_, py_, names[i], fontsize=fontsize, fontweight=fontweight,
                     color=color, ha="center", va="center", zorder=6)
             for i, (px_, py_) in zip(keep, positions)]

    target_px, target_r = _discs(xy[keep], size[keep])
    for txt, i, (mx, my), r in zip(texts, keep, target_px, target_r):
        bb = txt.get_window_extent(renderer)
        gap = np.hypot(np.clip(mx, bb.x0, bb.x1) - mx, np.clip(my, bb.y0, bb.y1) - my)
        if gap > r + 2:
            ax.add_patch(FancyArrowPatch(
                posA=txt.get_position(), posB=tuple(xy[i]), patchA=txt, arrowstyle="-",
                color="gray", lw=0.4, alpha=0.7, shrinkA=0.5,
                shrinkB=float(np.sqrt(size[i]) / 2), zorder=2, transform=ax.transData))
    return [names[i] for i in keep]


def key_below_axes(ax, *, ncol: int, fontsize: float | None = None, gap_pt: float = 2.0):
    """Move the axis legend just below the axis' own text (x tick labels and x label).

    The offset is measured after a draw, in points, so the key clears the axis text at any
    panel size and never sits on data. Call it once the panel's fonts are final. Returns the
    new legend, or None when the axis has no legend.
    """
    import matplotlib.transforms as mtransforms

    leg = ax.get_legend()
    if leg is None:
        return None
    handles = leg.legend_handles
    labels = [t.get_text() for t in leg.get_texts()]
    if fontsize is None:
        fontsize = leg.get_texts()[0].get_fontsize() if labels else plt.rcParams["legend.fontsize"]
    leg.remove()
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bottom = ax.get_window_extent(renderer).y0
    text_bottom = min(axes_bottom, ax.xaxis.get_tightbbox(renderer).y0)
    offset_pt = (axes_bottom - text_bottom) * 72 / fig.dpi + gap_pt
    return ax.legend(
        handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
        bbox_transform=mtransforms.offset_copy(ax.transAxes, fig=fig, y=-offset_pt,
                                               units="points"),
        ncol=ncol, fontsize=fontsize, frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
        borderpad=0.3, columnspacing=0.8, handlelength=0.8, handletextpad=0.3,
        labelspacing=0.2,
    )


def letter_at_panel_corner(ax, letter: str, *, fontsize: float) -> None:
    """Put a bold panel letter at the left edge of the panel's own labels, level with its title.

    For panels whose tick labels or title reach well left of the axes (heatmaps with long row
    names, narrow panels with long titles), where a fixed axes-fraction offset lands on a label.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lefts = [t.get_window_extent(renderer).x0 for t in ax.get_yticklabels() if t.get_text()]
    lefts += [ax.get_window_extent(renderer).x0]
    if ax.get_title():
        lefts.append(ax.title.get_window_extent(renderer).x0)
        top = ax.title.get_window_extent(renderer).y1
    else:
        top = ax.get_window_extent(renderer).y1
    x, y = fig.transFigure.inverted().transform((min(lefts), top))
    fig.text(x, y, letter, fontsize=fontsize, fontweight="bold", va="top", ha="left")


def group_axis_labels(ax, groups, *, fontsize: float, gap_pt: float = 1.5,
                      axis: str = "x") -> None:
    """Write one label per group of ticks, outside the tick labels.

    ``groups`` is a list of (label, start, end) in data coordinates along ``axis``. Used where
    each tick carries only its own short label (Pre/Post, a visit) and the shared part (dataset,
    arm) would otherwise repeat on every tick. On the x axis the label goes under the tick labels;
    on the y axis it goes to their left, centred on the group.
    """
    import matplotlib.transforms as mtransforms

    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', not {axis!r}")
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = ax.get_window_extent(renderer)
    if axis == "x":
        edge = min([box.y0] + [t.get_window_extent(renderer).y0
                               for t in ax.get_xticklabels() if t.get_text()])
        offset_pt = (box.y0 - edge) * 72 / fig.dpi + gap_pt
        trans = mtransforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=-offset_pt,
                                        units="points")
        for label, a, b in groups:
            ax.text((a + b) / 2, 0.0, label, transform=trans, ha="center", va="top",
                    fontsize=fontsize, clip_on=False)
    else:
        edge = min([box.x0] + [t.get_window_extent(renderer).x0
                               for t in ax.get_yticklabels() if t.get_text()])
        offset_pt = (box.x0 - edge) * 72 / fig.dpi + gap_pt
        trans = mtransforms.offset_copy(ax.get_yaxis_transform(), fig=fig, x=-offset_pt,
                                        units="points")
        for label, a, b in groups:
            ax.text(0.0, (a + b) / 2, label, transform=trans, ha="right", va="center_baseline",
                    fontsize=fontsize, clip_on=False)


def plot_cell_vs_participant_beta(ax, res_cell, res_part, *, title: str,
                                  composite: bool = False,
                                  label_fs: float | None = None) -> None:
    """Scatter of cell-level against participant-level beta_DiD, one point per signature.

    Shared by Figure 2B (TNBC) and Supp Figure 3B (melanoma). ``res_cell`` and
    ``res_part`` are ``did_table`` outputs; ``composite`` sizes everything for the
    small multi-panel artboard. ``label_fs`` overrides the signature-label size of
    the standalone panel (by default the active tick-label size).
    """
    import matplotlib.patches as mpatches
    from adjustText import adjust_text
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from scipy import stats

    rc = res_cell.set_index("feature")
    rp = res_part.set_index("feature")
    common = rc.index.intersection(rp.index)

    beta_cell = rc.loc[common, "beta_DiD"].values
    beta_part = rp.loc[common, "beta_DiD"].values
    colors = [COLORS["treated"] if b > 0 else COLORS["control"] for b in beta_part]

    ax.scatter(beta_cell, beta_part, c=colors, s=15 if composite else 30,
               edgecolors="white", linewidths=0.5, zorder=3)

    lim_lo = min(beta_cell.min(), beta_part.min()) * 1.2
    lim_hi = max(beta_cell.max(), beta_part.max()) * 1.2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--",
            color=COLORS["gray"], lw=1, zorder=1)
    ax.axhline(0, color=COLORS["gray"], lw=0.5, ls=":", zorder=0)
    ax.axvline(0, color=COLORS["gray"], lw=0.5, ls=":", zorder=0)

    # Explicit limits guarantee every marker is inside the view; an earlier version
    # clipped a marker below the axis while its label floated in the corner. The
    # standalone panel shares one range so the identity line runs at 45 degrees. The
    # composite cell is a ~3.5:1 strip where shared limits squeeze the narrow
    # cell-level spread into the middle third, so each axis spans its own data there.
    if composite:
        for set_lim, v in ((ax.set_xlim, beta_cell), (ax.set_ylim, beta_part)):
            pad = 0.15 * (v.max() - v.min())
            set_lim(v.min() - pad, v.max() + pad)
    else:
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)

    ax.tick_params(axis="y", labelsize=5)
    ax.set_xlabel(r'$\beta_{DID}$ (cell-level)', fontsize=9)
    ax.set_ylabel(r'$\beta_{DID}$ (participant-level)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Final text sizes, fixed BEFORE label layout: the standalone export raises
    # text to a minimum size afterwards and the composite shrinks it, and any
    # resize after adjust_text has run makes the labels collide.
    if composite:
        label_fs = 4.0
    elif label_fs is None:
        label_fs = max(
            5.0, FontProperties(size=plt.rcParams["xtick.labelsize"]).get_size_in_points()
        )

    # Spearman rho, not Pearson r. The key and the rho statistic sit in one row
    # below the axes, so nothing inside the plot competes with the labels
    # (adjustText 1.3.0 cannot avoid artists: its ``objects`` path maps
    # display-space boxes through the data transform a second time).
    rho, p = stats.spearmanr(beta_cell, beta_part)
    handles = [
        mpatches.Patch(facecolor=COLORS["treated"], label="Positive effect"),
        mpatches.Patch(facecolor=COLORS["control"], label="Negative effect"),
        Line2D([], [], linestyle="none", label=f"ρ = {rho:.2f}, p = {p:.1e}"),
    ]
    ax.legend(handles=handles, fontsize=5 if composite else label_fs, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.5 if composite else -0.12),
              frameon=True, framealpha=0.9, handlelength=0.8, handletextpad=0.3,
              columnspacing=1.2, borderpad=0.25)

    # Label every signature and let adjustText resolve overlaps, drawing a thin
    # leader line from each label back to its marker. x/y are the markers as static
    # obstacles (not label positions), so no label sits on another signature's
    # point; a fixed iteration budget keeps the layout reproducible (the default
    # one-second time limit depends on the machine).
    texts = [ax.text(xv, yv, sig_display(feat), fontsize=label_fs, alpha=0.9)
             for feat, xv, yv in zip(common, beta_cell, beta_part)]
    adjust_text(
        texts, x=beta_cell, y=beta_part, ax=ax, expand=(1.5, 1.8),
        force_text=(0.3, 0.5), iter_lim=1000,
        ensure_inside_axes=True,
    )
    for txt, xv, yv in zip(texts, beta_cell, beta_part):
        tx, ty = txt.get_position()
        ax.annotate("", xy=(xv, yv), xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.4, alpha=0.6))
    despine(ax)


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
    """Score all GENE_SIGNATURES modules using scanpy.tl.score_genes."""
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
        allow_download=True
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
    adata = load_stephenson_data(force_reprocess=False, allow_download=True)
    print(f"  Stephenson: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["steph"] = adata
    return adata


def get_vaccine():
    """Vaccine GSE171964 (full, ~21 K cells)."""
    if "vax" in _DATA_CACHE:
        return _DATA_CACHE["vax"]
    from sctrial.datasets import load_vaccine_gse171964
    adata = load_vaccine_gse171964(allow_download=True)
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
    adata = load_aml(allow_download=True)
    print(f"  AML: {adata.n_obs:,} cells, {adata.n_vars:,} genes")
    _DATA_CACHE["aml"] = adata
    return adata


def get_cart():
    """CAR-T therapy dataset (GSE290722, ~44 K cells)."""
    if "cart" in _DATA_CACHE:
        return _DATA_CACHE["cart"]
    from sctrial.datasets import load_cart
    adata = load_cart(allow_download=True)
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


def _gsea_fingerprint_path(cache_csv: Path) -> Path:
    return cache_csv.with_name("stats_fingerprint.txt")


def _gsea_did_cache_fresh(cache_csv: Path) -> bool:
    """A DiD-ranked GSEA cache is usable only if it was built by the current estimator.

    The ranking comes from ``did_table``, so results built by a different estimator
    are stale even though the file path is unchanged. A missing or mismatched
    fingerprint is treated as a cache miss.
    """
    stamp = _gsea_fingerprint_path(cache_csv)
    return (
        cache_csv.exists()
        and stamp.exists()
        and stamp.read_text().strip() == STATS_FINGERPRINT
    )


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
        if not force and _gsea_did_cache_fresh(cache_csv):
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
                _gsea_fingerprint_path(cache_csv).write_text(STATS_FINGERPRINT)
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
