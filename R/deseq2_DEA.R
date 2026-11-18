# Optional step: differential expression analysis with DESeq2
#
# This step is optional. The pipeline (python main.py) starts from the DESeq2 results,
# and data/example/Data/ already contains them. Run this script only to see how those
# results are produced, or to recompute them from raw counts.
#
# Output (same file names as the pipeline inputs):
#   GMCSFvUT_DEGs.csv, TNFvUT_DEGs.csv, GMTvUT_DEGs.csv  (DESeq2 results, each cytokine vs untreated)
#   Normalised_counts_GMT.csv                            (size-factor-normalized counts, all 14 samples)
#
# Model: one DESeq2 fit on all 14 samples with design ~ Donor + Cytokine
# (variation between donors is accounted for as a blocking factor), default settings,
# then one results() call per contrast against untreated (UT).
#
# Usage (from the repository root):
#   Rscript R/deseq2_DEA.R                                   # simulated example data
#   Rscript R/deseq2_DEA.R <raw_counts.csv> <samples.csv> <output_dir>
#
# By default the script uses data/example/DEA/Simulated_raw_counts.csv, simulated read counts
# for 2,000 genes with the same 14-sample design. Results from the simulated data are only a
# check that the script runs, and the output is written to data/example/DEA/output/ so that
# the example inputs of the pipeline are not overwritten.
#
# Inputs:
#   raw counts : genes in rows, first column = gene ID, one column per sample
#                (real data: GEO GSE288312 and GSE320016)
#   sample sheet: one row per sample with columns Sample, Cytokine, Donor
#                (Cytokine is one of UT, GMCSF, TNF, GM_TNF; sample names must match the
#                column names of the count table). data/example/DEA/samples.csv gives the
#                sample-to-donor assignment of the study, with the same library names and
#                donor codes as the GEO sample records.
#
# With the real data the script gives (padj <= 0.05, |log2FC| >= 1):
#   GM-CSF 429 up / 129 down, TNF-a 374 up / 67 down, GM+TNF 1,108 up / 380 down

args <- commandArgs(trailingOnly = TRUE)
counts_file  <- if (length(args) >= 1) args[1] else "data/example/DEA/Simulated_raw_counts.csv"
samples_file <- if (length(args) >= 2) args[2] else "data/example/DEA/samples.csv"
out_dir      <- if (length(args) >= 3) args[3] else "data/example/DEA/output"

suppressPackageStartupMessages(library(DESeq2))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

counts  <- read.csv(counts_file, row.names = 1, check.names = FALSE)
samples <- read.csv(samples_file, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")

stopifnot(all(c("Sample", "Donor", "Cytokine") %in% names(samples)),
          nrow(samples) == 14,
          all(samples$Sample %in% colnames(counts)))

counts <- as.matrix(counts[, samples$Sample])          # columns in the same order as the sample sheet
rownames(samples) <- samples$Sample
samples$Donor    <- factor(samples$Donor)
samples$Cytokine <- factor(samples$Cytokine, levels = c("UT", "GMCSF", "TNF", "GM_TNF"))

dds <- DESeqDataSetFromMatrix(countData = counts, colData = samples,
                              design = ~ Donor + Cytokine)
dds <- DESeq(dds)

contrasts <- list(GMCSFvUT = "GMCSF", TNFvUT = "TNF", GMTvUT = "GM_TNF")
for (name in names(contrasts)) {
  res <- results(dds, contrast = c("Cytokine", contrasts[[name]], "UT"))
  res <- res[order(res$padj), ]
  write.csv(as.data.frame(res), file.path(out_dir, paste0(name, "_DEGs.csv")))
  cat(sprintf("%-9s padj <= 0.05 and |log2FC| >= 1: up %d, down %d\n", name,
              sum(res$padj <= 0.05 & res$log2FoldChange >= 1, na.rm = TRUE),
              sum(res$padj <= 0.05 & res$log2FoldChange <= -1, na.rm = TRUE)))
}

write.csv(counts(dds, normalized = TRUE), file.path(out_dir, "Normalised_counts_GMT.csv"))

sessionInfo()
