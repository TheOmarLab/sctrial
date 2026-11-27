"""Build the full GSE169246 longitudinal tissue atlas from sctrial's v5 cache.

Requires numpy, scipy and h5py. Pass the file made by load_tnbc_zhang without
a participant-visit cell cap and an official GEO series matrix (GSE169246).
Gene membership is frozen in src/sctrial/interactive/atlas/data/reactome-2022.gmt, extracted by stable
identifier from the Enrichr Reactome_2022 library. Every retained cell is used.
"""

import argparse
import csv
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

ATLAS_DATA = Path(__file__).resolve().parents[2] / "src/sctrial/interactive/atlas/data"
GMT = ATLAS_DATA / "reactome-2022.gmt"
GEO = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169246"
MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE169nnn/GSE169246/matrix/GSE169246_series_matrix.txt.gz"
LIBRARY = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=Reactome_2022"
GENES = ("GZMB", "NKG7", "IFIT1", "ISG15", "CD79A", "MS4A1")
SETS = (
    ("ifn_ab", "Interferon alpha/beta signaling", "R-HSA-909733"),
    ("tcr", "TCR signaling", "R-HSA-202403"),
    ("mhc_ii", "MHC class II antigen presentation", "R-HSA-2132295"),
)
POPULATIONS = (
    ("cd8_t", "CD8 T cell"),
    ("cd4_t", "CD4 T cell"),
    ("b_cell", "B cell"),
    ("t_unresolved", "T cell (unresolved)"),
    ("myeloid", "Monocyte/Macrophage"),
    ("plasma", "Plasma cell"),
    ("treg", "Treg"),
    ("nk", "NK cell"),
    ("unassigned", "Unassigned"),
    ("ilc", "ILC"),
    ("dendritic", "Dendritic cell"),
    ("mast", "Mast cell"),
)
ARMS = (("all", "Both arms"), ("chemo", "Paclitaxel"), ("combo", "Atezolizumab + paclitaxel"))
VISITS = (("pre", "Pre-treatment"), ("post", "Follow-up"))
N_BINS = 32


def labels(pairs):
    return [dict(id=i, label=label) for i, label in pairs]


def category(handle, name):
    column = handle["obs"][name]
    codes = column["codes"][:]
    if np.any(codes < 0):
        raise ValueError(f"Missing {name}")
    return column["categories"].asstr()[:][codes]


def checked_index(values, order):
    if set(values) != set(order):
        raise ValueError(f"Unexpected categories: {set(values) ^ set(order)}")
    return np.array([order.index(name) for name in values], dtype=np.uint8)


def validate_visits(handle, series_matrix):
    with gzip.open(series_matrix, "rt") as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r and r[0].startswith("!Sample_")]
    names = next(r[1:] for r in rows if r[0] == "!Sample_title")
    public = {}
    for key in ("treatment", "group", "tissue"):
        row = next(
            r for r in rows if r[0] == "!Sample_characteristics_ch1" and r[1].startswith(key + ": ")
        )
        public[key] = dict(zip(names, [v.removeprefix(key + ": ") for v in row[1:]], strict=True))
    samples, arms, visits, sites = (
        category(handle, n) for n in ("sample_id", "arm", "visit", "biopsy_site")
    )
    mapped = set(zip(samples, arms, visits, sites, strict=True))
    for sample, arm, visit, site in mapped:
        if not (
            public["treatment"][sample].lower() == arm.lower()
            and public["group"][sample].lower().startswith(visit.lower())
            and public["tissue"][sample] == site
        ):
            raise ValueError(f"GEO sample metadata disagrees with the cached object for {sample}")
    return len(mapped)


def feature_metadata(var_names, gmt):
    members_by_id = {}
    for row in csv.reader(gmt.read_text().splitlines(), delimiter="\t"):
        members_by_id[row[0].split()[-1]] = list(dict.fromkeys(g for g in row[2:] if g))
    features = [dict(id=g, label=g, kind="gene", unit="log1p(CP10K)") for g in GENES]
    present = set(var_names)
    for ident, label, reactome in SETS:
        source = members_by_id[reactome]
        measured = [g for g in source if g in present]
        if not measured:
            raise ValueError(f"No measured genes for {reactome}")
        features.append(
            dict(
                id=ident,
                label=label,
                kind="pathway",
                unit="Mean member log1p(CP10K)",
                members=measured,
                memberCount=len(measured),
                sourceMembers=source,
                sourceMemberCount=len(source),
                accession=reactome,
                source="Reactome_2022 (Enrichr)",
                url="https://reactome.org/content/detail/" + reactome,
            )
        )
    return features


def extract_panel(layer, var_names, names):
    if layer.attrs["encoding-type"] != "csr_matrix":
        raise ValueError("Expected cached CSR log1p_norm expression layer")
    n, p = layer.attrs["shape"]
    columns = [var_names.index(g) for g in names]
    panel = np.empty((n, len(columns)), dtype=np.float32)
    pointers = layer["indptr"][:]
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        a, b = pointers[start], pointers[stop]
        block = csr_matrix(
            (layer["data"][a:b], layer["indices"][a:b], pointers[start : stop + 1] - a),
            shape=(stop - start, p),
        )
        panel[start:stop] = block[:, columns].toarray()
    if not np.all(np.isfinite(panel)) or np.any(panel < 0):
        raise ValueError("Expression matrix contains missing or negative values")
    return panel


def build(args, temp):
    with h5py.File(args.h5ad) as h:
        if h["uns/dataset"][()] != b"GSE169246" or h["uns/processing_params/version"][()] != b"v5":
            raise ValueError("Expected GSE169246 processed cache v5")
        if not isinstance(
            h["uns/processing_params/max_cells_per_participant_visit"][()], h5py.Empty
        ):
            raise ValueError("Expected an uncapped cell count")
        if set(category(h, "tissue_type")) != {"t"} or set(category(h, "annotation_source")) != {
            "publication"
        }:
            raise ValueError("Unexpected tissue or annotation source")
        verified_samples = validate_visits(h, args.series_matrix)
        pops = checked_index(category(h, "cell_type"), [p for _, p in POPULATIONS])
        arms = checked_index(category(h, "arm"), ["Chemo", "anti-PDL1+Chemo"]) + 1
        visits = checked_index(category(h, "visit"), ["Pre", "Post"])
        _, people = np.unique(category(h, "participant_id"), return_inverse=True)
        xy = h["obsm/X_umap"][:].astype(np.float64)
        if not np.all(np.isfinite(xy)) or xy.shape != (len(people), 2):
            raise ValueError("Incomplete cell coordinates")
        var_names = list(h["var/_index"].asstr()[:])
        if len(set(var_names)) != len(var_names):
            raise ValueError("Gene symbols must be unique")
        features = feature_metadata(var_names, GMT)
        panel_names = list(dict.fromkeys(g for f in features for g in f.get("members", [f["id"]])))
        panel = extract_panel(h["layers/log1p_norm"], var_names, panel_names)
    member_columns = {
        f["id"]: [panel_names.index(g) for g in f["members"]] for f in features if "members" in f
    }
    signals = np.column_stack(
        [
            panel[:, member_columns[f["id"]]].mean(axis=1, dtype=np.float64)
            if "members" in f
            else panel[:, panel_names.index(f["id"])]
            for f in features
        ]
    )
    for j, f in enumerate(features):
        f["domain"] = [0, float(signals[:, j].max())]

    minima = xy.min(axis=0)
    maxima = xy.max(axis=0)
    coordinates = np.rint((xy - minima) / (maxima - minima) * 65535).astype("<u2")
    stride = 7 + len(features)
    records = np.zeros((len(xy), stride), dtype=np.uint8)
    records[:, :4] = coordinates.view(np.uint8).reshape(-1, 4)
    records[:, 4:7] = np.column_stack([pops, arms, visits])
    records[:, 7:] = np.rint(255 * signals / signals.max(axis=0)).astype(np.uint8)
    # Sort by public display attributes rather than preserving sample/barcode order.
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0], pops, visits, arms))
    records[order].tofile(temp / "map.bin")

    groups = []
    pairs_by_arm = {}
    for arm, (arm_id, _) in enumerate(ARMS):
        arm_mask = (arms == arm) if arm else np.ones(len(arms), dtype=bool)
        in_arm = np.unique(people[arm_mask])
        pairs_by_arm[arm_id] = len(in_arm)
        captures = {
            (int(person), visit): np.flatnonzero((people == person) & (visits == visit))
            for person in in_arm
            for visit in range(2)
        }
        if any(not len(cells) for cells in captures.values()):
            raise ValueError("Complete paired visits required")
        denominators = [int((arm_mask & (visits == v)).sum()) for v in range(2)]
        for pop, (pop_id, _) in enumerate(POPULATIONS):
            paired = []
            for person in in_arm:
                pair = {}
                for visit, (visit_id, _) in enumerate(VISITS):
                    capture = captures[int(person), visit]
                    within = capture[pops[capture] == pop]
                    means = signals[within].mean(axis=0) if len(within) else [None] * len(features)
                    pair[visit_id] = dict(
                        cells=len(within),
                        totalCells=len(capture),
                        fraction=len(within) / len(capture),
                        signals={f["id"]: value for f, value in zip(features, means, strict=True)},
                    )
                paired.append(pair)
            for visit, (visit_id, _) in enumerate(VISITS):
                ix = np.flatnonzero(arm_mask & (visits == visit) & (pops == pop))
                n = len(ix)
                individual = [r[visit_id] for r in paired if r[visit_id]["cells"]]
                stats = {}
                gene_means = panel[ix].mean(axis=0, dtype=np.float64) if n else None
                for j, f in enumerate(features):
                    values = signals[ix, j]
                    counts, _ = np.histogram(values, bins=N_BINS, range=tuple(f["domain"]))
                    record = dict(
                        mean=float(values.mean()) if n else None,
                        participantMean=float(np.mean([r["signals"][f["id"]] for r in individual]))
                        if individual
                        else None,
                        bins=counts.tolist(),
                        zeros=int((values == 0).sum()),
                    )
                    if f["id"] in member_columns:
                        record["memberMeans"] = (
                            gene_means[member_columns[f["id"]]].tolist()
                            if n
                            else [None] * f["memberCount"]
                        )
                        if n:
                            np.testing.assert_allclose(
                                np.mean(record["memberMeans"]), record["mean"], rtol=1e-10
                            )
                    if sum(record["bins"]) != n:
                        raise ValueError("Cell histogram lost observations")
                    stats[f["id"]] = record
                groups.append(
                    dict(
                        arm=arm_id,
                        visit=visit_id,
                        population=pop_id,
                        cells=n,
                        totalCells=denominators[visit],
                        participants=len(individual),
                        totalParticipants=len(in_arm),
                        fraction=n / denominators[visit],
                        participantPairs=paired,
                        signals=stats,
                    )
                )
        for visit, denominator in zip(VISITS, denominators, strict=True):
            assert (
                sum(g["cells"] for g in groups if g["arm"] == arm_id and g["visit"] == visit[0])
                == denominator
            )

    assert (
        len(xy) == 141553
        and verified_samples == 24
        and pairs_by_arm == {"all": 12, "chemo": 6, "combo": 6}
    )
    study = dict(
        title="Triple-negative breast cancer",
        accession="GSE169246",
        citation="Zhang et al., Cancer Cell, 2021",
        doi="https://doi.org/10.1016/j.ccell.2021.09.010",
        url=GEO,
        cells=len(xy),
        participants=len(np.unique(people)),
        samples=verified_samples,
        tissue="CD45+ cells from paired tumor biopsies",
        scope="All cells retained in the 12-participant paired tumor-biopsy cohort processed with sctrial. The original study also profiled blood and additional participants.",
        annotation="sctrial grouped the authors’ published Table S2 RNA clusters by lineage. Unassigned and unresolved cells contribute to each capture denominator.",
        embedding="Joint UMAP of normalized RNA expression across the displayed cohort, computed by sctrial using 2,000 highly variable genes, 30 PCs and 15 neighbors.",
        expression="The arithmetic mean includes zeros. Each cell's RNA profile was normalized to 10,000 counts, then natural-log transformed with log1p.",
        pathway="The per-cell score averages normalized log1p expression of Reactome member genes retained in the expression matrix with equal gene weight.",
        histogram="32 equal-width bins on each feature’s full scale across both visits and arms. The first bin includes zeros. The last includes the upper endpoint.",
        library=LIBRARY,
        membershipFile="reactome-2022.gmt",
        gmtSha256=hashlib.sha256(GMT.read_bytes()).hexdigest(),
        geoSampleMetadata=MATRIX,
        reactomeLicense="https://reactome.org/license",
        processing="https://github.com/TheOmarLab/sctrial/blob/c4a9da8/src/sctrial/datasets.py",
        inputCacheVersion="v5",
        histogramBins=N_BINS,
    )
    map_info = dict(
        rows=len(xy),
        stride=stride,
        featureIds=[f["id"] for f in features],
        xDomain=[minima[0], maxima[0]],
        yDomain=[minima[1], maxima[1]],
        sha256=hashlib.sha256((temp / "map.bin").read_bytes()).hexdigest(),
    )
    with (temp / "study.json").open("w", encoding="utf-8") as out:
        json.dump(
            dict(
                study=study,
                arms=labels(ARMS),
                visits=labels(VISITS),
                populations=labels(POPULATIONS),
                features=features,
                groups=groups,
                map=map_info,
            ),
            out,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    print(
        f"{len(xy):,} cells, {len(groups)} groups, {len(features)} features, {verified_samples} verified biopsies"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("h5ad", type=Path)
    cli.add_argument("series_matrix", type=Path, help=f"Official GEO metadata from {MATRIX}")
    cli.add_argument("--output", type=Path, default=ATLAS_DATA)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as directory:
        stage = Path(directory)
        build(args, stage)
        for name in ("map.bin", "study.json"):
            (stage / name).replace(args.output / name)
