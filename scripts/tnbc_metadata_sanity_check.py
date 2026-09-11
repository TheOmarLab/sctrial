"""
Sanity check: participant IDs, treatment arms, and response status in the
processed TNBC h5ad, cross-referenced against three independent sources:

  Source 1 — _CLINICAL table (hardcoded in build_tnbc_anndata_v2.py)
  Source 2 — geo_metadata.csv (GEO submission metadata)
  Source 3 — mmc3.xlsx        (Zhang et al. 2021 supplementary table)

Run from the repo root:
    python scripts/tnbc_metadata_sanity_check.py
"""

import sys
from pathlib import Path

import pandas as pd
import anndata as ad

H5AD     = Path("datasets/tnbc_zhang/processed/tnbc_zhang_processed.h5ad")
GEO_CSV  = Path("datasets/tnbc_zhang/raw/geo_metadata.csv")
MMC3     = Path("datasets/tnbc_zhang/raw/mmc3.xlsx")

# ── Ground truth: _CLINICAL (mirrors build_tnbc_anndata_v2.py exactly) ────────
CLINICAL = pd.DataFrame({
    "participant_id": [
        "P019", "P012", "P017", "P002", "P005", "P016",   # PTX + ATZ
        "P022", "P020", "P013", "P025", "P018", "P023",   # PTX
    ],
    "arm_clinical": [
        "anti-PDL1+Chemo", "anti-PDL1+Chemo", "anti-PDL1+Chemo",
        "anti-PDL1+Chemo", "anti-PDL1+Chemo", "anti-PDL1+Chemo",
        "Chemo", "Chemo", "Chemo", "Chemo", "Chemo", "Chemo",
    ],
    "response": ["R","R","R","NR","NR","NR","R","R","R","R","R","NR"],
    "tumor_size_change": [
        -0.67, -0.46, -0.22,  0.00,  0.09,  0.17,
        -0.85, -0.55, -0.30, -0.23, -0.09,  0.03,
    ],
}).set_index("participant_id")

# ── Ground truth: geo_metadata.csv ────────────────────────────────────────────
geo = pd.read_csv(GEO_CSV)
geo["patient_id"] = geo["title"].str.extract(r"_?(P\d+)_")
# Normalise capitalisation to match the build script replacement
geo["treatment_norm"] = (
    geo["treatment"]
    .str.replace("Anti-PD-L1\\+Chemo", "anti-PDL1+Chemo", regex=True)
)
# Check per-patient consistency within geo_metadata
geo_consistency = (
    geo.groupby("patient_id")["treatment_norm"]
    .nunique()
    .rename("n_unique_arm")
)
geo_arm = (
    geo.drop_duplicates("patient_id")
    .set_index("patient_id")["treatment_norm"]
    .rename("arm_geo")
)

# ── Ground truth: mmc3.xlsx ───────────────────────────────────────────────────
mmc3 = pd.read_excel(MMC3, sheet_name="Single cell clustering", header=1)
mmc3_norm = mmc3.copy()
mmc3_norm["Treatment_norm"] = (
    mmc3_norm["Treatment"]
    .str.replace("Anti-PD-L1\\+Chemo", "anti-PDL1+Chemo", regex=True)
)
mmc3_per_pat = (
    mmc3_norm.drop_duplicates("Patient")
    .set_index("Patient")[["Treatment_norm", "Efficacy"]]
    .rename(columns={"Treatment_norm": "arm_mmc3", "Efficacy": "efficacy_mmc3"})
)

# ── Load processed h5ad ───────────────────────────────────────────────────────
print(f"Loading {H5AD} ...")
adata = ad.read_h5ad(H5AD)
obs = adata.obs.copy()
for col in obs.select_dtypes("category").columns:
    obs[col] = obs[col].astype(str)

print(f"  {adata.n_obs:,} cells, {obs['participant_id'].nunique()} patients\n")

# ── Per-patient summary from h5ad ────────────────────────────────────────────
h5ad_per_pat = (
    obs.groupby("participant_id", observed=True)
    .agg(
        n_cells       = ("participant_id", "size"),
        arm_h5ad      = ("arm",            "first"),
        response_h5ad = ("response",       "first"),
        tsc_h5ad      = ("tumor_size_change", "first"),
        n_arm_vals    = ("arm",            "nunique"),
        n_resp_vals   = ("response",       "nunique"),
    )
)

# ── Build comparison table ────────────────────────────────────────────────────
patients = sorted(h5ad_per_pat.index)

rows = []
for pid in patients:
    row = {"participant_id": pid}
    row["n_cells"]       = h5ad_per_pat.loc[pid, "n_cells"]
    row["arm_h5ad"]      = h5ad_per_pat.loc[pid, "arm_h5ad"]
    row["arm_clinical"]  = CLINICAL.loc[pid, "arm_clinical"]   if pid in CLINICAL.index else "MISSING"
    row["arm_geo"]       = geo_arm.get(pid, "MISSING")
    row["arm_mmc3"]      = mmc3_per_pat.loc[pid, "arm_mmc3"]   if pid in mmc3_per_pat.index else "MISSING"
    row["response_h5ad"] = h5ad_per_pat.loc[pid, "response_h5ad"]
    row["response_clinical"] = CLINICAL.loc[pid, "response"]   if pid in CLINICAL.index else "MISSING"
    row["tsc_h5ad"]      = h5ad_per_pat.loc[pid, "tsc_h5ad"]
    row["tsc_clinical"]  = CLINICAL.loc[pid, "tumor_size_change"] if pid in CLINICAL.index else float("nan")
    row["efficacy_mmc3"] = mmc3_per_pat.loc[pid, "efficacy_mmc3"] if pid in mmc3_per_pat.index else "MISSING"
    row["arm_consistent_in_h5ad"]  = h5ad_per_pat.loc[pid, "n_arm_vals"]  == 1
    row["resp_consistent_in_h5ad"] = h5ad_per_pat.loc[pid, "n_resp_vals"] == 1
    row["geo_arm_consistent"]      = geo_consistency.get(pid, 1) == 1
    rows.append(row)

df = pd.DataFrame(rows).set_index("participant_id")

# ── Print per-patient comparison ──────────────────────────────────────────────
print("=" * 80)
print("PER-PATIENT METADATA VERIFICATION")
print("=" * 80)

issues = []

for pid in patients:
    r = df.loc[pid]
    print(f"\n{pid}  ({r['n_cells']:,} cells)")

    # Arm
    arm_match = (r["arm_h5ad"] == r["arm_clinical"] == r["arm_geo"] == r["arm_mmc3"])
    arm_flag  = "" if arm_match else "  ← MISMATCH"
    print(f"  ARM")
    print(f"    h5ad:     {r['arm_h5ad']}")
    print(f"    _CLINICAL:{r['arm_clinical']}{arm_flag}")
    print(f"    geo_meta: {r['arm_geo']}{'' if r['arm_geo'] == r['arm_h5ad'] else '  ← MISMATCH'}")
    print(f"    mmc3:     {r['arm_mmc3']}{'' if r['arm_mmc3'] == r['arm_h5ad'] else '  ← MISMATCH'}")
    if not r["arm_consistent_in_h5ad"]:
        print(f"    WARNING: cells for {pid} have mixed arm values in h5ad!")
        issues.append(f"{pid}: mixed arm values in h5ad")
    if not r["geo_arm_consistent"]:
        print(f"    WARNING: {pid} has inconsistent arm values across geo_metadata rows!")
        issues.append(f"{pid}: inconsistent arm in geo_metadata")

    # Response
    resp_match = (r["response_h5ad"] == r["response_clinical"])
    tsc_match  = abs(float(r["tsc_h5ad"]) - float(r["tsc_clinical"])) < 1e-6
    resp_flag  = "" if resp_match else "  ← MISMATCH"
    tsc_flag   = "" if tsc_match  else "  ← MISMATCH"
    print(f"  RESPONSE")
    print(f"    h5ad:           {r['response_h5ad']}")
    print(f"    _CLINICAL:      {r['response_clinical']}{resp_flag}")
    print(f"    tumor_sz_change h5ad:     {float(r['tsc_h5ad']):+.2f}")
    print(f"    tumor_sz_change _CLINICAL:{float(r['tsc_clinical']):+.2f}{tsc_flag}")
    print(f"    mmc3 efficacy:  {r['efficacy_mmc3']}  (PR/SD→roughly R; PD→NR)")
    if not resp_match:
        issues.append(f"{pid}: response mismatch between h5ad and _CLINICAL")
    if not tsc_match:
        issues.append(f"{pid}: tumor_size_change mismatch between h5ad and _CLINICAL")
    if not r["resp_consistent_in_h5ad"]:
        print(f"    WARNING: cells for {pid} have mixed response values in h5ad!")
        issues.append(f"{pid}: mixed response values in h5ad")

# ── Check participant_id parsing (barcode → patient_id round-trip) ────────────
print("\n" + "=" * 80)
print("PARTICIPANT ID PARSING CHECK")
print("=" * 80)
# Re-derive participant_id from barcode_full and compare to stored participant_id
if "barcode_full" in obs.columns:
    derived_pid = obs["barcode_full"].str.split(".").str[1].str.extract(r"(P\d+)")[0]
    mismatch_pid = obs["participant_id"] != derived_pid
    n_mismatch = mismatch_pid.sum()
    if n_mismatch == 0:
        print(f"  All {adata.n_obs:,} barcodes → participant_id parsing: OK")
    else:
        print(f"  WARNING: {n_mismatch:,} cells have mismatched participant_id vs barcode-derived ID")
        issues.append(f"{n_mismatch} cells: participant_id != barcode-derived patient_id")
        print(obs[mismatch_pid][["barcode_full", "participant_id"]].head(10).to_string())
else:
    print("  SKIPPED: 'barcode_full' not in obs")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
summary = df[[
    "n_cells", "arm_h5ad", "arm_clinical", "arm_geo", "arm_mmc3",
    "response_h5ad", "response_clinical", "tsc_h5ad", "tsc_clinical", "efficacy_mmc3",
]].copy()
summary["arm_ok"]  = (
    (df["arm_h5ad"] == df["arm_clinical"]) &
    (df["arm_h5ad"] == df["arm_geo"]) &
    (df["arm_h5ad"] == df["arm_mmc3"])
)
summary["resp_ok"] = (df["response_h5ad"] == df["response_clinical"])
summary["tsc_ok"]  = (df["tsc_h5ad"].astype(float) - df["tsc_clinical"].astype(float)).abs() < 1e-6

print(summary[[
    "n_cells",
    "arm_h5ad", "arm_ok",
    "response_h5ad", "resp_ok",
    "tsc_h5ad", "tsc_ok",
    "efficacy_mmc3",
]].to_string())

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
if not issues:
    print("ALL CHECKS PASSED — participant IDs, arms, and response labels look correct.")
else:
    print(f"ISSUES FOUND ({len(issues)}):")
    for issue in issues:
        print(f"  ✗ {issue}")
print("=" * 80)
