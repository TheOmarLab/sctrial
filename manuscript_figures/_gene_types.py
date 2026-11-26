"""Gene types for the gene-level figure panels, looked up in NCBI Gene.

Figure 4 and Supplementary Figure 7 label, rank and display "protein-coding" genes. The reference
is ``reference/ncbi_gene_types.tsv.xz``, built from the Bioconductor package org.Hs.eg.db by
``scripts/build_gene_type_reference.R`` (version and NCBI source date are in its header). A name
resolves through its current NCBI symbol first and then through an alias or previous symbol, so
datasets annotated against older gene models (for example C10orf54, now VSIR) still resolve.

A gene counts as protein-coding if NCBI types it ``protein-coding``, or if it is a functional
immunoglobulin or T cell receptor constant-region gene (IGHA1, IGKC, IGLC2, TRAC, ...), which
encode protein but which NCBI types ``other`` together with the V, D and J segments. Pseudogenes,
non-coding RNAs, V/D/J segments and names that do not resolve are not protein-coding.

Display panels additionally leave out protein-coding families that dominate effect-size rankings
for technical or polymorphism reasons rather than biology: mitochondrially encoded genes,
cytosolic ribosomal proteins, HLA genes, olfactory and vomeronasal receptors, and small
proline-rich proteins.
"""

from __future__ import annotations

import csv
import lzma
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from sctrial.preprocessing import is_ribosomal_gene

REFERENCE_PATH = Path(__file__).resolve().parent / "reference" / "ncbi_gene_types.tsv.xz"

PROTEIN_CODING = "protein-coding"
_PLACEHOLDER = "unknown"

_IG_TR_CONSTANT = re.compile(r"^(IGH(A[12]|D|E|G[1-4]|M)|IGKC|IGLC\d+|TR(AC|BC[12]|DC|GC[12]))$")

_DISPLAY_EXCLUDED = re.compile(
    r"^(MT-.+"  # mitochondrially encoded
    r"|RPL36AL"  # ribosomal protein not covered by sctrial's ribosomal pattern
    r"|HLA-.+"  # HLA, highly polymorphic
    r"|OR\d+[A-Z]+\d+P?"  # olfactory receptors, including segregating "P" genes typed protein-coding
    r"|VN1R\d+"  # vomeronasal receptors
    r"|SPRR\d[A-Z]?)$"  # small proline-rich proteins
)


@dataclass(frozen=True)
class Resolution:
    """How a gene name resolved: ``via`` is symbol, alias, ambiguous or unresolved."""

    name: str
    symbol: str | None
    gene_type: str | None
    via: str


class GeneTypeReference:
    """Name-to-gene-type lookup built from a reference table.

    The table has columns ``name``, ``symbol``, ``gene_type`` and ``name_kind`` (``symbol`` for a
    current NCBI symbol, ``alias`` for an alias or previous symbol). A name that maps to records
    of different types is resolved by ignoring NCBI ``unknown`` placeholder records when a real
    type exists; if real types still disagree the name is ambiguous and resolves to no type.
    """

    def __init__(self, table: pd.DataFrame):
        need = {"name", "symbol", "gene_type", "name_kind"}
        missing = need - set(table.columns)
        if missing:
            raise ValueError(f"gene type reference lacks columns {sorted(missing)}")
        self._by_kind = {
            kind: {
                name: tuple(zip(g["symbol"], g["gene_type"]))
                for name, g in table[table["name_kind"] == kind].groupby("name", sort=False)
            }
            for kind in ("symbol", "alias")
        }
        self._cache: dict[str, Resolution] = {}

    def resolve(self, name: str) -> Resolution:
        name = str(name)
        hit = self._cache.get(name)
        if hit is None:
            hit = self._resolve(name)
            self._cache[name] = hit
        return hit

    def _resolve(self, name: str) -> Resolution:
        for kind in ("symbol", "alias"):
            records = self._by_kind[kind].get(name)
            if records is None:
                continue
            real = [r for r in records if r[1] != _PLACEHOLDER] or list(records)
            types = {t for _, t in real}
            if len(types) > 1:
                return Resolution(name, None, None, "ambiguous")
            symbols = sorted({s for s, _ in real})
            symbol = symbols[0] if len(symbols) == 1 else None
            return Resolution(name, symbol, types.pop(), kind)
        return Resolution(name, None, None, "unresolved")

    def gene_type(self, name: str) -> str | None:
        return self.resolve(name).gene_type

    def is_protein_coding(self, name: str) -> bool:
        res = self.resolve(name)
        if res.gene_type == PROTEIN_CODING:
            return True
        return res.gene_type == "other" and bool(_IG_TR_CONSTANT.match(res.symbol or ""))

    def is_display_gene(self, name: str) -> bool:
        name = str(name)
        if _DISPLAY_EXCLUDED.match(name) or is_ribosomal_gene(name):
            return False
        res = self.resolve(name)
        symbol = res.symbol or name
        if symbol != name and (_DISPLAY_EXCLUDED.match(symbol) or is_ribosomal_gene(symbol)):
            return False
        return self.is_protein_coding(name)

    def summarize(self, names: Iterable[str]) -> dict[str, int]:
        """Count how the given names resolved, for logging what was excluded and why."""
        counts = {"symbol": 0, "alias": 0, "ambiguous": 0, "unresolved": 0}
        for n in names:
            counts[self.resolve(n).via] += 1
        return counts


def read_reference_table(path: Path = REFERENCE_PATH) -> tuple[list[str], pd.DataFrame]:
    """Header lines and table of the reference file.

    Only the leading ``#`` lines are header: some NCBI aliases contain ``#`` (TEX#, RDH#2), so the
    table must not be read with a comment character, and some start with an apostrophe, so quote
    handling is off.
    """
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        header = []
        for line in fh:
            if not line.startswith("#"):
                break
            header.append(line.rstrip("\n"))
    table = pd.read_csv(path, sep="\t", skiprows=len(header), dtype=str, keep_default_na=False,
                        quoting=csv.QUOTE_NONE)
    return header, table


def load_reference(path: Path = REFERENCE_PATH) -> GeneTypeReference:
    return GeneTypeReference(read_reference_table(path)[1])


@lru_cache(maxsize=1)
def default_reference() -> GeneTypeReference:
    return load_reference()


def gene_type(name: str) -> str | None:
    """NCBI gene type of *name*, or None when it does not resolve unambiguously."""
    return default_reference().gene_type(name)


def is_protein_coding(name: str) -> bool:
    """True for NCBI protein-coding genes and functional IG/TR constant-region genes."""
    return default_reference().is_protein_coding(name)


def is_display_gene(name: str) -> bool:
    """True for protein-coding genes outside the technical families left out of display panels."""
    return default_reference().is_display_gene(name)


def describe_resolution(names: Iterable[str], context: str) -> str:
    """One log line saying how many names resolved, through which route, and how many were kept."""
    names = [str(n) for n in names]
    ref = default_reference()
    c = ref.summarize(names)
    kept = sum(ref.is_display_gene(n) for n in names)
    return (
        f"  {context}: {len(names)} genes, {c['symbol']} by NCBI symbol, {c['alias']} by alias, "
        f"{c['ambiguous']} ambiguous, {c['unresolved']} unresolved; {kept} kept for display"
    )
