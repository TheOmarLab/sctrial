# Build the gene-type reference used by manuscript_figures/_gene_types.py.
#
# Source: NCBI Gene, through the Bioconductor annotation package org.Hs.eg.db.
# Output: manuscript_figures/reference/ncbi_gene_types.tsv.xz, one row per (name, gene) pair, where
# a name is either a gene's current NCBI symbol or one of its aliases (previous symbols included).
# Rows for NCBI "biological-region" records are dropped: they are genomic features, not genes.
# An alias that is also some gene's current symbol is dropped, because lookup resolves current
# symbols first and such an alias could never be used. A few current symbols belong to more than
# one NCBI record (for example TEC: a protein-coding gene and an "unknown" locus record); all
# records are kept and the Python resolver decides.
#
# Usage (from sctrial/sc_trial_inference):
#   Rscript scripts/build_gene_type_reference.R

suppressPackageStartupMessages({
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

out <- file.path("manuscript_figures", "reference", "ncbi_gene_types.tsv.xz")
dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)

ids <- keys(org.Hs.eg.db, keytype = "ENTREZID")
genes <- AnnotationDbi::select(
  org.Hs.eg.db, keys = ids, columns = c("SYMBOL", "GENETYPE"), keytype = "ENTREZID"
)
stopifnot(!anyNA(genes$GENETYPE))
genes <- genes[!is.na(genes$SYMBOL) & genes$GENETYPE != "biological-region", ]

aliases <- AnnotationDbi::select(
  org.Hs.eg.db, keys = genes$ENTREZID, columns = "ALIAS", keytype = "ENTREZID"
)
aliases <- aliases[!is.na(aliases$ALIAS), ]
aliases <- merge(aliases, genes, by = "ENTREZID")
aliases <- aliases[!(aliases$ALIAS %in% genes$SYMBOL), ]

rows <- rbind(
  data.frame(name = genes$SYMBOL, symbol = genes$SYMBOL,
             gene_type = genes$GENETYPE, name_kind = "symbol"),
  data.frame(name = aliases$ALIAS, symbol = aliases$SYMBOL,
             gene_type = aliases$GENETYPE, name_kind = "alias")
)
rows <- unique(rows)
rows <- rows[order(rows$name, rows$name_kind, rows$symbol, method = "radix"), ]

meta <- metadata(org.Hs.eg.db)
src_date <- meta$value[meta$name == "EGSOURCEDATE"]
header <- c(
  sprintf("# source: NCBI Gene via Bioconductor org.Hs.eg.db %s", packageVersion("org.Hs.eg.db")),
  sprintf("# NCBI Gene source date: %s", src_date),
  sprintf("# built with %s, AnnotationDbi %s", R.version.string, packageVersion("AnnotationDbi")),
  "# generator: scripts/build_gene_type_reference.R",
  sprintf("# rows: %d (%d current symbols, %d alias rows)", nrow(rows),
          sum(rows$name_kind == "symbol"), sum(rows$name_kind == "alias"))
)
con <- xzfile(out, "w", compression = 9)
writeLines(header, con)
write.table(rows, con, sep = "\t", quote = FALSE, row.names = FALSE)
close(con)
cat(header, sep = "\n")
cat("wrote", out, "\n")
