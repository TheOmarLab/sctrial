"""Gene types for the Figure 4 / Supp Fig 7 gene panels come from NCBI Gene, not from name patterns."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gt():
    name = "manuscript_figures._gene_types_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "manuscript_figures/_gene_types.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[name]


def test_reference_records_its_provenance_and_parses_completely(gt):
    header, table = gt.read_reference_table()
    text = "\n".join(header)
    assert "org.Hs.eg.db" in text and "NCBI Gene source date" in text
    assert "scripts/build_gene_type_reference.R" in text
    stated = header[-1].split("rows:")[1].split()
    assert len(table) == int(stated[0])
    assert (table["name_kind"] == "symbol").sum() == int(stated[1].lstrip("("))
    assert set(table["name_kind"]) == {"symbol", "alias"}
    assert (table[["name", "symbol", "gene_type"]] != "").all().all()
    # aliases that contain the header character or start with a quote survive parsing
    assert {"TEX#", "RDH#2", "'C-K-RAS"} <= set(table["name"])


# Names that reached "protein-coding" panels or text through the old name heuristic.
@pytest.mark.parametrize("name, gene_type", [
    ("CDKN2AIPNLP3", "pseudo"),
    ("PPP1R1AP2", "pseudo"),
    ("TAAR4P", "pseudo"),
    ("DDX3YP2", "pseudo"),  # resolves through its alias to DDX3P2
    ("IGLC6", "pseudo"),
    ("MIRLET7G", "ncRNA"),
    ("SCARNA22", "ncRNA"),
    ("DKFZP434A062", "ncRNA"),
])
def test_noncoding_genes_are_not_protein_coding(gt, name, gene_type):
    assert gt.gene_type(name) == gene_type
    assert not gt.is_protein_coding(name)
    assert not gt.is_display_gene(name)


@pytest.mark.parametrize("name", [
    "TP53", "CENPF", "EEF1A1", "KRT13", "SMIM22", "CCL3L1",
    "RPS6KA1",  # ribosomal protein S6 kinase, not a ribosomal protein
    "TRADD",  # caught by the old TCR segment pattern TRA[VDJ]
    "PIP4P2",  # caught by the old pseudogene suffix pattern
    "IGHA1", "IGHE", "IGKC", "IGLC2", "TRAC",  # functional constant regions, NCBI type "other"
    "TEC",  # protein-coding gene that shares its symbol with an NCBI "unknown" locus record
])
def test_protein_coding_genes_are_kept(gt, name):
    assert gt.is_protein_coding(name)
    assert gt.is_display_gene(name)


@pytest.mark.parametrize("name, symbol", [
    ("C10orf54", "VSIR"),
    ("FAM46C", "TENT5C"),
    ("KIAA0101", "PCLAF"),
    ("SEPT7", "SEPTIN7"),
])
def test_previous_symbols_resolve_to_current_genes(gt, name, symbol):
    res = gt.default_reference().resolve(name)
    assert (res.symbol, res.gene_type, res.via) == (symbol, "protein-coding", "alias")
    assert gt.is_display_gene(name)


@pytest.mark.parametrize("name", [
    "IGHV1-2", "TRBV7-4",  # V segments
    "AF178030.1", "RP11-543A18.1", "MGC4771", "",  # do not resolve
])
def test_segments_and_unresolved_names_are_excluded(gt, name):
    assert not gt.is_protein_coding(name)
    assert not gt.is_display_gene(name)


@pytest.mark.parametrize("name", [
    "MT-CO1", "RPL13A", "RPS4X", "RPLP0", "RPL36AL", "HLA-DRA", "HLA-A", "OR2A1", "OR2I1P",
    "VN1R1", "SPRR1A",
])
def test_technical_families_are_left_out_of_display(gt, name):
    assert not gt.is_display_gene(name)


def test_resolution_rules_on_a_toy_table(gt):
    table = pd.DataFrame([
        ("PCG", "PCG", "protein-coding", "symbol"),
        ("DUAL", "DUAL", "protein-coding", "symbol"),
        ("DUAL", "DUAL", "unknown", "symbol"),
        ("CLASH", "CLASH", "protein-coding", "symbol"),
        ("CLASH", "CLASH", "ncRNA", "symbol"),
        ("OLDNAME", "PCG", "protein-coding", "alias"),
        ("SHARED", "PCG", "protein-coding", "alias"),
        ("SHARED", "PSEUDO1", "pseudo", "alias"),
        ("PCG", "OTHER", "pseudo", "alias"),
    ], columns=["name", "symbol", "gene_type", "name_kind"])
    ref = gt.GeneTypeReference(table)
    assert ref.resolve("PCG").via == "symbol"  # a current symbol wins over an alias
    assert ref.gene_type("DUAL") == "protein-coding"  # placeholder record ignored
    assert ref.resolve("CLASH").via == "ambiguous" and not ref.is_protein_coding("CLASH")
    assert ref.resolve("OLDNAME") == gt.Resolution("OLDNAME", "PCG", "protein-coding", "alias")
    assert ref.resolve("SHARED").via == "ambiguous" and not ref.is_display_gene("SHARED")
    assert ref.resolve("NOPE").via == "unresolved"
    assert ref.summarize(["PCG", "OLDNAME", "CLASH", "NOPE"]) == {
        "symbol": 1, "alias": 1, "ambiguous": 1, "unresolved": 1,
    }


def test_reference_rejects_a_malformed_table(gt):
    with pytest.raises(ValueError, match="lacks columns"):
        gt.GeneTypeReference(pd.DataFrame({"name": ["A"], "gene_type": ["protein-coding"]}))


def test_describe_resolution_counts_and_kept(gt):
    line = gt.describe_resolution(["KRT13", "C10orf54", "AF178030.1", "CDKN2AIPNLP3"], "toy")
    assert "4 genes, 2 by NCBI symbol, 1 by alias, 0 ambiguous, 1 unresolved; 2 kept" in line
