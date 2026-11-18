# =============================================================
# src/label_preparation.py — Step 1
# Filter genes and assign Up / Down labels from the DESeq2 results.
# Up   = padj <= 0.05 and log2FC >= 1.0
# Down = padj <= 0.05 and log2FC <= -1.0
# All other genes are unlabeled (non-DEGs) and are never used for training.
# =============================================================

import numpy as np
import pandas as pd

import config as C
from src.utils.io_utils import load_df, save_df


def load_expression():
    """Normalized counts, genes with a zero count in every sample removed, log2(x + 1)."""
    norm = load_df(f"{C.DATA}/Normalised_counts_GMT.csv", index_col=0)
    expected_cols = sum(C.CONFIG["samples"].values(), [])
    if sorted(norm.columns) != sorted(expected_cols):
        raise ValueError(f"Columns in file: {list(norm.columns)}\nExpected: {expected_cols}")
    keep = (norm > 0).any(axis=1)
    norm_filtered = norm[keep]
    master_genes = norm_filtered.index
    log_norm = np.log2(norm_filtered + 1)
    return norm, master_genes, log_norm


def load_deg_tables(master_genes):
    deg = {}
    for cond, fname in C.CONFIG["deg_files"].items():
        d = load_df(f"{C.DATA}/{fname}", index_col=0)
        deg[cond] = d.loc[d.index.isin(master_genes)]
    return deg


def prepare_up_down(deg_df, padj_max, lfc_min):
    df = deg_df.copy()
    valid = df["padj"].notna()
    is_up   = valid & (df["padj"] <= padj_max) & (df["log2FoldChange"] >= lfc_min)
    is_down = valid & (df["padj"] <= padj_max) & (df["log2FoldChange"] <= -lfc_min)
    df["label"] = np.nan
    df.loc[is_up, "label"] = 1
    df.loc[is_down, "label"] = 0
    return df.dropna(subset=["label"])


def run():
    norm, master_genes, _ = load_expression()
    print(f"Raw: {norm.shape[0]:,} genes × {norm.shape[1]} samples")
    for grp, cols in C.CONFIG["samples"].items():
        print(f"  {grp:<7} n = {len(cols)}: {', '.join(cols)}")
    print(f"After filter: {len(master_genes):,} genes")

    deg = load_deg_tables(master_genes)
    pmax, lmin = C.CONFIG["labels"]["padj_max"], C.CONFIG["labels"]["lfc_min"]
    for cond in C.CONFIG["conditions"]:
        lab = prepare_up_down(deg[cond], pmax, lmin)
        save_df(lab, f"{C.FEAT_DIR}/labels_UpvDown_{cond}.csv", index=True)
        n_up, n_down = int((lab["label"] == 1).sum()), int((lab["label"] == 0).sum())
        print(f"  {cond:<7} DEG table: {len(deg[cond]):,} genes | NA padj: {deg[cond]['padj'].isna().sum():,} | "
              f"Up={n_up:,}, Down={n_down:,}, Ratio={n_up / max(n_down, 1):.2f}:1")
