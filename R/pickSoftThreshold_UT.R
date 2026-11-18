# Scale-free topology fit for the untreated (UT) samples (documentation of the choice of beta)
#
# The pipeline does not use R for any computation. This optional script reproduces the
# pickSoftThreshold result reported in the Methods: no power between 1 and 20 reached a
# scale-free fit index of 0.8 (maximum signed R^2 = 0.189 at power 6), so beta = 15 was fixed
# and beta = 12 and 18 were used as sensitivity analyses.
#
# Usage (from the repository root): Rscript R/pickSoftThreshold_UT.R
# Requires: WGCNA (the manuscript used WGCNA 1.74)

norm_file <- "data/example/Data/Normalised_counts_GMT.csv"   # use the full data file to reproduce the manuscript
ut_cols   <- c("UT_1", "UT_2", "UT_3", "UT_5", "UT_6")

suppressPackageStartupMessages(library(WGCNA))
norm <- read.csv(norm_file, row.names = 1, check.names = FALSE)
norm <- norm[rowSums(norm > 0) > 0, ]                         # same gene filter as the pipeline
datExpr <- t(log2(norm[, ut_cols] + 1))                       # samples in rows, genes in columns

sft <- pickSoftThreshold(datExpr, powerVector = 1:20, networkType = "signed", verbose = 0)
print(round(sft$fitIndices, 3))
cat("powerEstimate =", sft$powerEstimate, "\n")
cat("WGCNA version:", as.character(packageVersion("WGCNA")), "\n")
