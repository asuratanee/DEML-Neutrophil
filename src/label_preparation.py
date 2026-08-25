# =============================================================
# src/label_preparation.py
# Loads normalized counts and DESeq2 results, then assigns
# gene labels for downstream classification.
# =============================================================

import pandas as pd
import numpy as np
from config import (DATA_PATH, FEATURES_PATH, PADJ_THRESHOLD,
                    FC_THRESHOLD, STABLE_PADJ_MIN, DATA_BACKEND)


def load_data(data_path):
    """Load and filter normalized counts."""
    norm = pd.read_csv(f"{data_path}/Normalised_counts_GMT.csv", index_col=0)
    keep = (norm > 0).any(axis=1)
    norm_filtered = norm[keep]
    print(f"Genes after zero-count filter: {norm_filtered.shape[0]:,}")
    return norm_filtered


def load_deg_results(data_path, master_genes):
    """Load DESeq2 results for all three conditions."""
    files = {
        'GMCSF' : 'GMCSFvUT_DEGs.csv',
        'TNF'   : 'TNFvUT_DEGs.csv',
        'GM_TNF': 'GMTvUT_DEGs.csv',
    }
    deg = {}
    for cond, fname in files.items():
        df = pd.read_csv(f"{data_path}/{fname}", index_col=0)
        df = df.loc[df.index.isin(master_genes)]
        deg[cond] = df
        print(f"DEG {cond}: {len(df):,} genes")
    return deg


def prepare_deg_stable(deg_df,
                        padj_threshold=PADJ_THRESHOLD,
                        fc_threshold=FC_THRESHOLD,
                        stable_padj_min=STABLE_PADJ_MIN):
    """
    Assign binary labels for DEG vs Stable classification.
      label = 1 : DEG   (padj <= 0.05 and |log2FC| >= 1.0)
      label = 0 : Stable (padj > 0.5)
    """
    df = deg_df.copy()
    is_deg    = (df['padj'] <= padj_threshold) & (df['log2FoldChange'].abs() >= fc_threshold)
    is_stable = df['padj'] > stable_padj_min
    df['label'] = np.nan
    df.loc[is_deg,    'label'] = 1
    df.loc[is_stable, 'label'] = 0
    return df.dropna(subset=['label'])


def prepare_up_down(deg_df,
                    padj_threshold=PADJ_THRESHOLD,
                    fc_threshold=FC_THRESHOLD):
    """
    Assign binary labels for Up vs Down classification (DEGs only).
      label = 1 : Up-regulated   (padj <= 0.05 and log2FC >= 1.0)
      label = 0 : Down-regulated (padj <= 0.05 and log2FC <= -1.0)
    """
    df = deg_df.copy()
    is_up   = (df['padj'] <= padj_threshold) & (df['log2FoldChange'] >= fc_threshold)
    is_down = (df['padj'] <= padj_threshold) & (df['log2FoldChange'] <= -fc_threshold)
    df['label'] = np.nan
    df.loc[is_up,   'label'] = 1
    df.loc[is_down, 'label'] = 0
    return df.dropna(subset=['label'])


def run(data_path=DATA_PATH, features_path=FEATURES_PATH):
    print("=" * 55)
    print("  Step 1: Label Preparation")
    print("=" * 55)

    norm_filtered = load_data(data_path)
    master_genes  = norm_filtered.index
    deg           = load_deg_results(data_path, master_genes)

    print("\nDEG vs Stable:")
    for cond, df_deg in deg.items():
        ds = prepare_deg_stable(df_deg)
        n_deg    = (ds['label'] == 1).sum()
        n_stable = (ds['label'] == 0).sum()
        print(f"  {cond}: DEG={n_deg:,}, Stable={n_stable:,}")
        ds.to_csv(f"{features_path}/labels_DEGvStable_{cond}.csv")

    print("\nUp vs Down:")
    for cond, df_deg in deg.items():
        ud = prepare_up_down(df_deg)
        n_up   = (ud['label'] == 1).sum()
        n_down = (ud['label'] == 0).sum()
        print(f"  {cond}: Up={n_up:,}, Down={n_down:,}")
        ud.to_csv(f"{features_path}/labels_UpvDown_{cond}.csv")

    print("\n✅ Labels saved.")
    return norm_filtered, deg


if __name__ == "__main__":
    run()
