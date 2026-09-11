"""
Sanity check for TNBC preprocessing (GSE169246 / Zhang et al. 2021).

Two sections run automatically if the relevant files exist:

  SECTION A — Raw MTX funnel
    Loads the raw MTX files from scratch and replays every filtering step
    (tissue filter → Pre/Post → min-cell-per-visit → pairing → QC thresholds),
    recording exact cell counts at each stage.

  SECTION B — Processed h5ad checks
    Loads the final h5ad and plots QC distributions, UMAPs, cell type
    composition, and runs data integrity checks.

Run from the repo root:
    python scripts/tnbc_sanity_check.py

Override paths:
    python scripts/tnbc_sanity_check.py --raw-dir datasets/tnbc_zhang/raw \
                                        --h5ad   datasets/tnbc_zhang/processed/tnbc_zhang_processed.h5ad
"""

import sys
import gzip
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy.io import mmread

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--raw-dir", default="datasets/tnbc_zhang/raw")
parser.add_argument("--h5ad",    default="datasets/tnbc_zhang/processed/tnbc_zhang_processed.h5ad")
args, _ = parser.parse_known_args()

RAW_DIR  = Path(args.raw_dir)
H5AD     = Path(args.h5ad)
OUT_DIR  = Path("scripts/tnbc_sanity_check_figs")
OUT_DIR.mkdir(exist_ok=True)

# QC thresholds — must match build_tnbc_anndata_v2.py
MIN_GENES        = 200
MAX_GENES        = 6000
MAX_MT_PCT       = 20
MIN_CELLS_GENE   = 10   # sc.pp.filter_genes min_cells
MIN_CELLS_VISIT  = 50   # per-patient per-visit minimum

# Clinical response table — mirrors build_tnbc_anndata_v2.py _CLINICAL
_CLINICAL = pd.DataFrame({
    "participant_id": [
        "P019", "P012", "P017", "P002", "P005", "P016",
        "P022", "P020", "P013", "P025", "P018", "P023",
    ],
    "response": [
        "R", "R", "R", "NR", "NR", "NR",
        "R", "R", "R", "R",  "R",  "NR",
    ],
    "tumor_size_change": [
        -0.67, -0.46, -0.22,  0.00,  0.09,  0.17,
        -0.85, -0.55, -0.30, -0.23, -0.09,  0.03,
    ],
}).set_index("participant_id")


def _hbar(ax, steps, title):
    """Draw a horizontal funnel bar chart on ax."""
    labels = [s[0] for s in steps]
    counts = [s[1] for s in steps]
    bars = ax.barh(labels[::-1], counts[::-1], color="steelblue", edgecolor="white")
    for bar, n in zip(bars, counts[::-1]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{n:,}", va="center", fontsize=8)
    ax.set_xlabel("Number of cells")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Raw MTX funnel
# ═══════════════════════════════════════════════════════════════════════════════
raw_files = [
    RAW_DIR / "GSE169246_TNBC_RNA.counts.mtx.gz",
    RAW_DIR / "GSE169246_TNBC_RNA.barcode.tsv.gz",
    RAW_DIR / "GSE169246_TNBC_RNA.feature.tsv.gz",
    RAW_DIR / "geo_metadata.csv",
]

if not all(f.exists() for f in raw_files):
    missing = [f for f in raw_files if not f.exists()]
    print(f"SECTION A skipped — raw files not found:\n  " + "\n  ".join(str(m) for m in missing))
else:
    print("=" * 70)
    print("SECTION A — Raw MTX funnel")
    print("=" * 70)

    funnel: list[tuple[str, int]] = []

    # ── A1. Load raw matrix ──────────────────────────────────────────────────
    print("\nA1. Loading raw MTX...")
    mat = mmread(str(RAW_DIR / "GSE169246_TNBC_RNA.counts.mtx.gz")).T.tocsc()

    with gzip.open(str(RAW_DIR / "GSE169246_TNBC_RNA.barcode.tsv.gz"), "rt") as f:
        barcodes = [l.strip() for l in f]
    with gzip.open(str(RAW_DIR / "GSE169246_TNBC_RNA.feature.tsv.gz"), "rt") as f:
        features_raw = [l.strip().split("\t") for l in f]
    gene_ids   = [r[0] for r in features_raw]
    gene_names = [r[1] if len(r) > 1 else r[0] for r in features_raw]

    raw_adata = ad.AnnData(
        X=mat,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )
    raw_adata.var["gene_ids"] = gene_ids
    raw_adata.var_names_make_unique()

    n = raw_adata.n_obs
    funnel.append(("Raw MTX (all cells)", n))
    print(f"  {n:,} cells × {raw_adata.n_vars:,} genes")

    # ── A2. Parse metadata from barcodes ────────────────────────────────────
    print("\nA2. Parsing barcode metadata...")
    obs = raw_adata.obs.copy()
    obs["barcode_full"]  = obs.index
    obs["sample_id"]     = obs["barcode_full"].str.split(".").str[1]
    obs["timepoint_raw"] = obs["sample_id"].str.split("_").str[0]
    obs["patient_id"]    = obs["sample_id"].str.extract(r"(P\d+)")
    obs["tissue_type"]   = obs["sample_id"].str.split("_").str[-1]

    geo_meta = pd.read_csv(RAW_DIR / "geo_metadata.csv")
    geo_meta["treatment"]  = geo_meta["treatment"].replace({"Anti-PD-L1+Chemo": "anti-PDL1+Chemo"})
    geo_meta["patient_id"] = geo_meta["title"].str.extract(r"_?(P\d+)_")
    patient_arm = geo_meta.drop_duplicates("patient_id").set_index("patient_id")["treatment"]
    obs["arm"]              = obs["patient_id"].map(patient_arm)
    obs["response"]         = obs["patient_id"].map(_CLINICAL["response"])
    obs["tumor_size_change"]= obs["patient_id"].map(_CLINICAL["tumor_size_change"])
    raw_adata.obs = obs

    tissue_counts   = obs["tissue_type"].value_counts()
    timepoint_counts = obs["timepoint_raw"].value_counts()
    print(f"  Tissue types:  {dict(tissue_counts)}")
    print(f"  Timepoints:    {dict(timepoint_counts)}")

    # snapshots: (stage_label, obs_df, patient_col, timepoint_col, removed_patients_set)
    # collected at each filter step; used to draw the multi-stage panel figure.
    snapshots: list[tuple] = []

    # ── A3. Filter: tumor biopsies only ─────────────────────────────────────
    # Snapshot BEFORE this filter so the panel shows blood cells too
    snapshots.append((
        "1. All cells (blood + tumor, all timepoints)",
        raw_adata.obs.copy(), "patient_id", "timepoint_raw",
        set(),  # nothing removed yet
    ))

    raw_adata = raw_adata[raw_adata.obs["tissue_type"] == "t"].copy()
    funnel.append(("Tumor biopsies (tissue='t')", raw_adata.n_obs))
    print(f"\nA3. Tumor biopsies: {raw_adata.n_obs:,}")
    snapshots.append((
        "2. Tumor biopsies only (tissue='t')",
        raw_adata.obs.copy(), "patient_id", "timepoint_raw",
        set(),
    ))

    # ── A4. Filter: Pre + Post only ──────────────────────────────────────────
    # Snapshot BEFORE this filter to show Prog cells about to be dropped
    # (mark patients that only have Prog as "removed")
    prog_only = {
        pid for pid in raw_adata.obs["patient_id"].unique()
        if set(raw_adata.obs.loc[raw_adata.obs["patient_id"] == pid, "timepoint_raw"].unique())
           <= {"Prog"}
    }
    snapshots.append((
        "3. Tumor biopsies (Pre + Post + Prog)",
        raw_adata.obs.copy(), "patient_id", "timepoint_raw",
        prog_only,  # these patients vanish after the Pre/Post filter
    ))

    raw_adata = raw_adata[raw_adata.obs["timepoint_raw"].isin(["Pre", "Post"])].copy()
    raw_adata.obs["visit"]          = raw_adata.obs["timepoint_raw"].astype(str)
    raw_adata.obs["participant_id"] = raw_adata.obs["patient_id"].astype(str)
    funnel.append(("Pre + Post timepoints", raw_adata.n_obs))
    print(f"A4. Pre+Post only: {raw_adata.n_obs:,}")

    # ── A5. Remove patients with <50 cells in any visit ──────────────────────
    print(f"\nA5. Per-patient per-visit cell counts (before 50-cell filter):")
    pids_all = sorted(raw_adata.obs["participant_id"].unique())
    print(f"  {'Patient':<8} {'Pre':>8} {'Post':>8}  flag")
    remove_pids: set[str] = set()
    for pid in pids_all:
        n_pre  = ((raw_adata.obs["participant_id"] == pid) & (raw_adata.obs["visit"] == "Pre")).sum()
        n_post = ((raw_adata.obs["participant_id"] == pid) & (raw_adata.obs["visit"] == "Post")).sum()
        flags = []
        if 0 < n_pre  < MIN_CELLS_VISIT: flags.append(f"Pre<{MIN_CELLS_VISIT}")
        if 0 < n_post < MIN_CELLS_VISIT: flags.append(f"Post<{MIN_CELLS_VISIT}")
        if flags:
            remove_pids.add(pid)
        print(f"  {pid:<8} {n_pre:>8,} {n_post:>8,}  {'REMOVE: ' + ', '.join(flags) if flags else 'OK'}")

    # Snapshot BEFORE the 50-cell filter (grey = about to be removed)
    snapshots.append((
        f"4. Pre + Post only (before min-{MIN_CELLS_VISIT}-cells/visit filter)",
        raw_adata.obs.copy(), "participant_id", "visit",
        remove_pids,
    ))

    if remove_pids:
        raw_adata = raw_adata[~raw_adata.obs["participant_id"].isin(remove_pids)].copy()
        print(f"  Removed {len(remove_pids)} patient(s): {sorted(remove_pids)}")
    funnel.append((f"Min {MIN_CELLS_VISIT} cells/visit filter", raw_adata.n_obs))

    # ── A6. Keep only paired patients ────────────────────────────────────────
    paired: set[str] = set()
    for pid in raw_adata.obs["participant_id"].unique():
        visits = set(raw_adata.obs.loc[raw_adata.obs["participant_id"] == pid, "visit"].unique())
        if {"Pre", "Post"}.issubset(visits):
            paired.add(pid)
    unpaired = set(raw_adata.obs["participant_id"].unique()) - paired
    if unpaired:
        print(f"\nA6. Unpaired patients removed: {sorted(unpaired)}")

    # Snapshot BEFORE pairing filter
    snapshots.append((
        "5. After min-cells/visit filter (before pairing filter)",
        raw_adata.obs.copy(), "participant_id", "visit",
        unpaired,
    ))

    raw_adata = raw_adata[raw_adata.obs["participant_id"].isin(paired)].copy()
    funnel.append(("Paired patients only", raw_adata.n_obs))
    print(f"A6. Paired patients: {len(paired)}  →  {raw_adata.n_obs:,} cells")

    # ── A7. QC metrics ───────────────────────────────────────────────────────
    print(f"\nA7. QC filtering...")
    raw_adata.var["mt"] = raw_adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(raw_adata, qc_vars=["mt"], percent_top=None, inplace=True)

    n_pre_qc = raw_adata.n_obs
    mask_genes = (
        (raw_adata.obs["n_genes_by_counts"] >= MIN_GENES) &
        (raw_adata.obs["n_genes_by_counts"] <= MAX_GENES)
    )
    mask_mt = raw_adata.obs["pct_counts_mt"] < MAX_MT_PCT

    funnel.append((f"min_genes ≥ {MIN_GENES}", mask_genes.sum()))
    funnel.append((f"max_genes ≤ {MAX_GENES}", mask_genes.sum()))
    funnel.append((f"pct_mt < {MAX_MT_PCT}%",  (mask_genes & mask_mt).sum()))

    raw_qc = raw_adata[mask_genes & mask_mt].copy()
    n_genes_before = raw_qc.n_vars
    sc.pp.filter_genes(raw_qc, min_cells=MIN_CELLS_GENE)
    funnel.append((f"min_cells per gene ≥ {MIN_CELLS_GENE}", raw_qc.n_obs))

    print(f"  Before QC: {n_pre_qc:,} cells, {n_genes_before:,} genes")
    print(f"  After QC:  {raw_qc.n_obs:,} cells, {raw_qc.n_vars:,} genes")
    print(f"  Removed:   {n_pre_qc - raw_qc.n_obs:,} cells, {n_genes_before - raw_qc.n_vars:,} genes")

    # Snapshot AFTER QC — final raw state (nothing removed at patient level)
    snapshots.append((
        f"6. After QC (min_genes={MIN_GENES}, max_genes={MAX_GENES}, pct_mt<{MAX_MT_PCT}%)",
        raw_qc.obs.copy(), "participant_id", "visit",
        set(),
    ))

    # ── A8. Funnel bar chart ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    _hbar(ax, funnel, "Cell-count funnel — raw MTX → QC")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "A1_raw_funnel.png", dpi=150)
    plt.close(fig)
    print(f"\n  → saved A1_raw_funnel.png")

    # ── A9. QC metric distributions ──────────────────────────────────────────
    qc_metrics = [
        ("n_genes_by_counts", "Genes per cell",         [(MIN_GENES, "r", "--"), (MAX_GENES, "r", "--")]),
        ("total_counts",       "UMI counts per cell",    []),
        ("pct_counts_mt",      "% mitochondrial counts", [(MAX_MT_PCT, "r", "--")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, title, thresholds) in zip(axes, qc_metrics):
        vals = raw_adata.obs[col].values
        ax.violinplot(vals, positions=[0], showmedians=True)
        ax.set_xticks([])
        ax.set_ylabel(title)
        ax.set_title(title)
        for thresh, color, ls in thresholds:
            ax.axhline(thresh, color=color, linestyle=ls, linewidth=1.2,
                       label=f"threshold={thresh}")
            ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.05, 0.97, f"median={np.median(vals):.0f}", transform=ax.transAxes,
                va="top", fontsize=8, color="gray")
    plt.suptitle("SECTION A — QC distributions (paired tumor biopsies, pre-QC-filter)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "A2_raw_qc_violins.png", dpi=150)
    plt.close(fig)
    print(f"  → saved A2_raw_qc_violins.png")

    # ── A10. Counts vs genes scatter ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sc_plot = ax.scatter(
        raw_adata.obs["total_counts"],
        raw_adata.obs["n_genes_by_counts"],
        c=raw_adata.obs["pct_counts_mt"],
        s=0.3, alpha=0.3, cmap="viridis", rasterized=True,
    )
    plt.colorbar(sc_plot, ax=ax, label="% MT")
    ax.axhline(MIN_GENES, color="r", linestyle="--", linewidth=0.8, label=f"min_genes={MIN_GENES}")
    ax.axhline(MAX_GENES, color="r", linestyle="--", linewidth=0.8, label=f"max_genes={MAX_GENES}")
    ax.set_xlabel("Total UMI counts")
    ax.set_ylabel("Genes detected")
    ax.set_title("SECTION A — Counts vs Genes (coloured by % MT)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "A3_raw_qc_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  → saved A3_raw_qc_scatter.png")

    # ── A11. Multi-stage per-patient cell-count panels ────────────────────────
    # One subplot per preprocessing stage. Bars coloured by timepoint/visit;
    # patients that are removed in THIS step are shown in grey.
    TIMEPOINT_COLORS = {
        "Pre":  "#4c72b0",
        "Post": "#dd8452",
        "Prog": "#55a868",
        "b":    "#c44e52",
        "t":    "#8172b2",
    }
    # Gather every patient that ever appears, sorted, for a consistent x-axis
    all_patients = sorted(
        set(pid for _, obs_snap, pat_col, _, _ in snapshots
            for pid in obs_snap[pat_col].dropna().unique())
    )
    # Response map (from any snapshot that has the response column)
    resp_map: dict[str, str] = {}
    for _, obs_snap, pat_col, _, _ in snapshots:
        if "response" in obs_snap.columns:
            resp_map.update(
                obs_snap.groupby(pat_col, observed=True)["response"]
                .first().dropna().to_dict()
            )

    n_stages = len(snapshots)
    fig, axes = plt.subplots(n_stages, 1, figsize=(max(14, len(all_patients) * 0.9 + 3),
                                                    n_stages * 3.5))

    for ax, (stage_label, obs_snap, pat_col, tp_col, removed_pids) in zip(axes, snapshots):
        # Build count table: patients × timepoints
        pt_tp = (
            obs_snap.groupby([pat_col, tp_col], observed=True)
            .size().unstack(fill_value=0)
        )
        # Reindex to full patient list (missing patients get 0)
        pt_tp = pt_tp.reindex(all_patients, fill_value=0)

        timepoints = pt_tp.columns.tolist()
        n_tp = len(timepoints)
        width = 0.7 / max(n_tp, 1)
        x = np.arange(len(all_patients))

        for i, tp in enumerate(timepoints):
            offset = (i - n_tp / 2 + 0.5) * width
            base_color = TIMEPOINT_COLORS.get(tp, f"C{i}")
            bar_colors = [
                "#cccccc" if pid in removed_pids else base_color
                for pid in all_patients
            ]
            ax.bar(x + offset, pt_tp[tp].values, width,
                   label=tp, color=bar_colors, edgecolor="none")

        # x-axis labels: patient + response, mark removed with strikethrough-style
        x_labels = []
        for pid in all_patients:
            r = resp_map.get(pid, "")
            suffix = f"\n({r})" if r else ""
            x_labels.append(f"{pid}{suffix}")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11, rotation=45, ha="right")
        ax.set_ylabel("Cells", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(stage_label, fontsize=13, loc="left", pad=3)
        ax.spines[["top", "right"]].set_visible(False)

        if MIN_CELLS_VISIT and tp_col == "visit":
            ax.axhline(MIN_CELLS_VISIT, color="gray", linestyle=":", linewidth=0.8)

        # Legend only on first panel
        if ax is axes[0]:
            ax.legend(title="Timepoint", fontsize=11, title_fontsize=12, loc="upper right")

        # Annotate removed patients with a grey label
        for j, pid in enumerate(all_patients):
            if pid in removed_pids:
                ax.text(x[j], ax.get_ylim()[1] * 0.02, "✕",
                        ha="center", va="bottom", fontsize=7, color="#888888")

    plt.suptitle("SECTION A — Cells per patient at each preprocessing stage\n"
                 "(grey bars + ✕ = patients removed at this step)",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "A4_stages_per_patient.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved A4_stages_per_patient.png")

    # Print full funnel table
    print(f"\n  ── Full funnel ──────────────────────────────────────────────")
    for label, n in funnel:
        print(f"  {label:<40} {n:>8,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Processed h5ad checks
# ═══════════════════════════════════════════════════════════════════════════════
if not H5AD.exists():
    print(f"\nSECTION B skipped — {H5AD} not found")
    sys.exit(0)

print("\n" + "=" * 70)
print("SECTION B — Processed h5ad checks")
print("=" * 70)
print(f"\nLoading {H5AD} ...")
adata = ad.read_h5ad(H5AD)
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  Layers: {list(adata.layers.keys())}")
print(f"  obs columns: {list(adata.obs.columns)}")

# Reconstruct QC metrics from stored raw counts for plots
proc_raw = adata.copy()
if "counts" in adata.layers:
    proc_raw.X = adata.layers["counts"].copy()
    print("  Using 'counts' layer for QC metric reconstruction")
else:
    print("  WARNING: no 'counts' layer — using .X (may be log-normalised)")
proc_raw.var["mt"] = proc_raw.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(proc_raw, qc_vars=["mt"], percent_top=None, inplace=True)

# ── B1. QC metric distributions (post-filter, from stored counts) ────────────
print("\nB1. QC distributions (stored counts layer)...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = [
    ("n_genes_by_counts", "Genes per cell",         [(MIN_GENES, "r", "--"), (MAX_GENES, "r", "--")]),
    ("total_counts",       "UMI counts per cell",    []),
    ("pct_counts_mt",      "% mitochondrial counts", [(MAX_MT_PCT, "r", "--")]),
]
for ax, (col, title, thresholds) in zip(axes, metrics):
    vals = proc_raw.obs[col].values
    ax.violinplot(vals, positions=[0], showmedians=True)
    ax.set_xticks([])
    ax.set_ylabel(title)
    ax.set_title(title)
    for thresh, color, ls in thresholds:
        ax.axhline(thresh, color=color, linestyle=ls, linewidth=1.2, label=f"threshold={thresh}")
        ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    med = np.median(vals)
    ax.text(0.05, 0.97, f"median={med:.0f}", transform=ax.transAxes,
            va="top", fontsize=8, color="gray")
plt.suptitle("SECTION B — QC metrics (post-filter, from counts layer)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "B1_proc_qc_violins.png", dpi=150)
plt.close(fig)
print("  → saved B1_proc_qc_violins.png")

fig, ax = plt.subplots(figsize=(7, 5))
sc_plot = ax.scatter(
    proc_raw.obs["total_counts"], proc_raw.obs["n_genes_by_counts"],
    c=proc_raw.obs["pct_counts_mt"],
    s=0.5, alpha=0.4, cmap="viridis", rasterized=True,
)
plt.colorbar(sc_plot, ax=ax, label="% MT")
ax.axhline(MIN_GENES, color="r", linestyle="--", linewidth=0.8, label=f"min_genes={MIN_GENES}")
ax.axhline(MAX_GENES, color="r", linestyle="--", linewidth=0.8, label=f"max_genes={MAX_GENES}")
ax.set_xlabel("Total UMI counts")
ax.set_ylabel("Genes detected")
ax.set_title("SECTION B — Counts vs Genes (coloured by % MT)")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "B2_proc_qc_scatter.png", dpi=150)
plt.close(fig)
print("  → saved B2_proc_qc_scatter.png")

# ── B2. Per-patient per-visit cell counts ────────────────────────────────────
print("\nB2. Per-patient / per-visit cell counts...")
if "participant_id" in adata.obs and "visit" in adata.obs:
    pt_visit = (
        adata.obs.groupby(["participant_id", "visit"], observed=True)
        .size().unstack(fill_value=0).sort_index()
    )
    resp_map = {}
    if "response" in adata.obs.columns:
        resp_map = adata.obs.groupby("participant_id", observed=True)["response"].first().to_dict()
    row_labels = [f"{pid}\n({resp_map.get(pid, '?')})" for pid in pt_visit.index]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pt_visit) * 0.5 + 2)))
    x = np.arange(len(pt_visit))
    width = 0.35
    colors = {"Pre": "#4c72b0", "Post": "#dd8452"}
    for i, v in enumerate(pt_visit.columns.tolist()):
        offset = (i - len(pt_visit.columns) / 2 + 0.5) * width
        ax.bar(x + offset, pt_visit[v], width, label=v,
               color=colors.get(v, f"C{i}"), edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels, fontsize=8)
    ax.set_ylabel("Cell count")
    ax.set_title("SECTION B — Cells per patient per visit (final processed)")
    ax.axhline(MIN_CELLS_VISIT, color="gray", linestyle=":", linewidth=0.8)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "B3_proc_cells_per_patient_visit.png", dpi=150)
    plt.close(fig)
    print("  → saved B3_proc_cells_per_patient_visit.png")
    print(pt_visit.to_string())

# ── B3. UMAP panels ───────────────────────────────────────────────────────────
print("\nB3. UMAP panels...")
if "X_umap" not in adata.obsm:
    print("  WARNING: no UMAP — skipping")
else:
    umap_colors = [c for c in ["cell_type", "leiden", "visit", "response", "participant_id", "arm"]
                   if c in adata.obs.columns]
    ncols = 3
    nrows = (len(umap_colors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = axes.flatten()
    for ax, col in zip(axes_flat, umap_colors):
        legend_loc = "on data" if adata.obs[col].nunique() <= 20 else "right margin"
        sc.pl.umap(adata, color=col, ax=ax, show=False,
                   legend_loc=legend_loc, legend_fontsize=7, title=col)
    for ax in axes_flat[len(umap_colors):]:
        ax.set_visible(False)
    plt.suptitle("SECTION B — UMAP embeddings", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "B4_umap_panels.png", dpi=150)
    plt.close(fig)
    print("  → saved B4_umap_panels.png")

    qc_umap_cols = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
                    if c in adata.obs.columns]
    if qc_umap_cols:
        fig, axes = plt.subplots(1, len(qc_umap_cols), figsize=(len(qc_umap_cols) * 5, 4.5))
        if len(qc_umap_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, qc_umap_cols):
            sc.pl.umap(adata, color=col, ax=ax, show=False, color_map="viridis", title=col)
        plt.suptitle("SECTION B — QC metrics on UMAP", fontsize=12)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "B5_umap_qc.png", dpi=150)
        plt.close(fig)
        print("  → saved B5_umap_qc.png")

# ── B4. Cell type composition ─────────────────────────────────────────────────
print("\nB4. Cell type composition...")
if "cell_type" in adata.obs.columns:
    cell_types = adata.obs["cell_type"].value_counts()
    print(f"  Cell types: {len(cell_types)}")
    for ct, n in cell_types.items():
        print(f"    {ct}: {n:,} ({n/adata.n_obs*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        cell_types.values, labels=cell_types.index,
        autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
        startangle=90, pctdistance=0.8,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("SECTION B — Overall cell type composition")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "B6_celltype_pie.png", dpi=150)
    plt.close(fig)
    print("  → saved B6_celltype_pie.png")

    if "visit" in adata.obs.columns:
        for visit in sorted(adata.obs["visit"].unique()):
            sub = adata[adata.obs["visit"] == visit]
            ct_counts = (
                sub.obs.groupby(["participant_id", "cell_type"], observed=True)
                .size().unstack(fill_value=0)
            )
            ct_frac = ct_counts.div(ct_counts.sum(axis=1), axis=0)
            fig, ax = plt.subplots(figsize=(max(6, len(ct_frac) * 0.7 + 2), 5))
            ct_frac.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="none")
            ax.set_ylabel("Fraction of cells")
            ax.set_title(f"SECTION B — Cell type composition per patient — {visit}")
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
            ax.tick_params(axis="x", rotation=45)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            fig.savefig(OUT_DIR / f"B7_celltype_per_patient_{visit}.png", dpi=150)
            plt.close(fig)
            print(f"  → saved B7_celltype_per_patient_{visit}.png")

    if "response" in adata.obs.columns:
        ct_resp = (
            adata.obs.groupby(["response", "cell_type"], observed=True)
            .size().unstack(fill_value=0)
        )
        ct_resp_frac = ct_resp.div(ct_resp.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ct_resp_frac.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="none")
        ax.set_ylabel("Fraction of cells")
        ax.set_title("SECTION B — Cell type composition R vs NR")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.tick_params(axis="x", rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "B8_celltype_by_response.png", dpi=150)
        plt.close(fig)
        print("  → saved B8_celltype_by_response.png")

# ── B5. Integrity checks ──────────────────────────────────────────────────────
print("\nB5. Data integrity checks...")
checks_passed = True

for col in ["participant_id", "visit", "response", "cell_type", "arm"]:
    if col not in adata.obs.columns:
        print(f"  MISSING column: {col}")
        checks_passed = False
        continue
    n_nan = adata.obs[col].isna().sum()
    print(f"  {col}: {'OK' if n_nan == 0 else f'WARNING {n_nan:,} NaN'}")
    if n_nan > 0:
        checks_passed = False

if "participant_id" in adata.obs and "visit" in adata.obs:
    unpaired = [
        (pid, set(adata.obs.loc[adata.obs["participant_id"] == pid, "visit"].unique()))
        for pid in adata.obs["participant_id"].unique()
        if not {"Pre", "Post"}.issubset(
            set(adata.obs.loc[adata.obs["participant_id"] == pid, "visit"].unique())
        )
    ]
    if unpaired:
        print(f"  WARNING: {len(unpaired)} patients lack both Pre+Post: {[p[0] for p in unpaired]}")
        checks_passed = False
    else:
        print(f"  All {adata.obs['participant_id'].nunique()} patients have Pre+Post: OK")

n_dup = adata.obs.index.duplicated().sum()
print(f"  Barcodes unique: {'OK' if n_dup == 0 else f'WARNING {n_dup:,} duplicates'}")
if n_dup > 0:
    checks_passed = False

if "highly_variable" in adata.var.columns:
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  HVGs: {n_hvg} (expected ~2000)")

for key, expected_dim in [("X_pca", 50), ("X_umap", 2)]:
    if key in adata.obsm:
        dim = adata.obsm[key].shape[1]
        ok = dim == expected_dim
        print(f"  {key}: shape {adata.obsm[key].shape} — {'OK' if ok else f'WARNING: got {dim}, expected {expected_dim}'}")
        if not ok:
            checks_passed = False
    else:
        print(f"  MISSING: {key}")
        checks_passed = False

print(f"\n  {'All checks passed.' if checks_passed else 'Some checks FAILED — review warnings above.'}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n── Final summary ────────────────────────────────────────────────────────")
print(f"  Cells:    {adata.n_obs:,}")
print(f"  Genes:    {adata.n_vars:,}")
if "participant_id" in adata.obs:
    print(f"  Patients: {adata.obs['participant_id'].nunique()}")
if "arm" in adata.obs:
    print(f"  Arms:     {dict(adata.obs['arm'].value_counts())}")
if "response" in adata.obs:
    print(f"  Response: {dict(adata.obs['response'].value_counts())}")
if "leiden" in adata.obs:
    print(f"  Leiden clusters: {adata.obs['leiden'].nunique()}")
if "cell_type" in adata.obs:
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")

print(f"\nFigures saved to: {OUT_DIR.resolve()}/")
print("  SECTION A (raw MTX):")
print("    A1_raw_funnel.png              — full cell-count funnel from MTX to QC")
print("    A2_raw_qc_violins.png          — QC distributions before QC filter")
print("    A3_raw_qc_scatter.png          — counts vs genes before QC filter")
print("    A4_raw_cells_per_patient_visit.png — per-patient counts before QC")
print("  SECTION B (processed h5ad):")
print("    B1_proc_qc_violins.png         — QC distributions after all filters")
print("    B2_proc_qc_scatter.png         — counts vs genes after all filters")
print("    B3_proc_cells_per_patient_visit.png — per-patient counts after all filters")
print("    B4_umap_panels.png             — UMAP coloured by cell_type, visit, response, …")
print("    B5_umap_qc.png                 — UMAP coloured by QC metrics")
print("    B6_celltype_pie.png            — overall cell type pie")
print("    B7_celltype_per_patient_*.png  — composition per patient per visit")
print("    B8_celltype_by_response.png    — composition R vs NR")
