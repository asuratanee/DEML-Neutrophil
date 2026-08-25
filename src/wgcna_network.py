# =============================================================
# src/wgcna_network.py
# Constructs weighted gene co-expression networks using WGCNA.
# Requires R with WGCNA package installed.
# Estimated runtime: ~2 hours per condition on a standard CPU.
# =============================================================

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
from config import (DATA_PATH, NETWORKS_PATH, UT_COLS, CONDITIONS,
                    WGCNA_BETA, WGCNA_WEIGHT_THRESHOLD, WGCNA_NETWORK_TYPE)

pandas2ri.activate()


def load_r_wgcna():
    """Install and load WGCNA in R."""
    r('''
    if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
    if (!require("WGCNA", quietly = TRUE)) BiocManager::install("WGCNA")
    library(WGCNA)
    ''')
    print("✅ WGCNA loaded")


def build_network(log_norm, cols, cond_name, beta, weight_threshold, networks_path):
    """Build adjacency matrix and save edge list for one condition."""
    print(f"\nConstructing {cond_name} network...")

    datExpr = log_norm[cols].T
    ro.globalenv['datExpr_cond'] = pandas2ri.py2rpy(datExpr)

    r(f'''
    adj = adjacency(datExpr_cond,
                    power = {beta},
                    type  = "{WGCNA_NETWORK_TYPE}")
    ''')

    adj_array = np.array(ro.globalenv['adj'])
    n         = len(adj_array)

    i_idx, j_idx = np.triu_indices(n, k=1)
    weights      = adj_array[i_idx, j_idx]
    mask         = weights >= weight_threshold

    gene_names = log_norm.index
    edge_df = pd.DataFrame({
        'source': gene_names[i_idx[mask]],
        'target': gene_names[j_idx[mask]],
        'weight': weights[mask]
    })

    density = len(edge_df) / (n * (n - 1) / 2)
    print(f"  Nodes  : {n:,}")
    print(f"  Edges  : {len(edge_df):,}")
    print(f"  Density: {density:.4%}")

    out_path = f"{networks_path}/{cond_name}_WGCNA_weighted_t04.csv"
    edge_df.to_csv(out_path, index=False)
    print(f"  Saved  : {out_path}")


def run(data_path=DATA_PATH, networks_path=NETWORKS_PATH, beta=WGCNA_BETA):
    print("=" * 55)
    print("  Step 2: WGCNA Network Construction")
    print("=" * 55)
    print(f"  soft-thresholding power β = {beta}")
    print(f"  edge weight threshold     = {WGCNA_WEIGHT_THRESHOLD}")

    load_r_wgcna()

    norm = pd.read_csv(f"{data_path}/Normalised_counts_GMT.csv", index_col=0)
    norm_filtered = norm[(norm > 0).any(axis=1)]
    log_norm = np.log2(norm_filtered + 1)

    all_conditions = {'UT': UT_COLS}
    all_conditions.update(CONDITIONS)

    for cond_name, cols in all_conditions.items():
        build_network(log_norm, cols, cond_name, beta,
                      WGCNA_WEIGHT_THRESHOLD, networks_path)

    print("\n✅ All WGCNA networks created.")


if __name__ == "__main__":
    run()
