# =============================================================
# src/gsea_analysis.py
# Runs preranked GSEA for both DEG and DEML frameworks against
# MSigDB Hallmark and KEGG gene sets.
# =============================================================

import numpy as np
import pandas as pd
import gseapy as gp
import warnings
warnings.filterwarnings('ignore')

from config import (DATA_PATH, RESULTS_PATH, CONDITIONS,
                    GSEA_MIN_SIZE, GSEA_MAX_SIZE,
                    GSEA_PERMUTATIONS, GSEA_FDR_THRESHOLD,
                    GSEA_GENE_SETS, RANDOM_SEED)

DEG_FILES = {
    'GM_TNF': 'GMTvUT_DEGs.csv',
    'GMCSF' : 'GMCSFvUT_DEGs.csv',
    'TNF'   : 'TNFvUT_DEGs.csv',
}


def run_prerank(ranked_genes, gene_set_name, label):
    """Run preranked GSEA and return significant results (FDR < threshold)."""
    try:
        res = gp.prerank(
            rnk            = ranked_genes,
            gene_sets      = gene_set_name,
            min_size       = GSEA_MIN_SIZE,
            max_size       = GSEA_MAX_SIZE,
            permutation_num= GSEA_PERMUTATIONS,
            outdir         = None,
            seed           = RANDOM_SEED,
            verbose        = False,
        )
        df = res.res2d.copy()
        df = df[df['FDR q-val'] < GSEA_FDR_THRESHOLD].copy()
        df['label'] = label
        return df
    except Exception as e:
        print(f"    ⚠️  GSEA error: {e}")
        return pd.DataFrame()


def run(data_path=DATA_PATH, results_path=RESULTS_PATH):
    print("=" * 55)
    print("  Step 6: GSEA Analysis (DEG and DEML)")
    print("=" * 55)

    all_results = []

    for cond in CONDITIONS:
        print(f"\n{'─'*55}\n  {cond}\n{'─'*55}")

        # ── DEG ranked list (log2FC) ──────────────────────────
        deg_df = pd.read_csv(f"{data_path}/{DEG_FILES[cond]}", index_col=0)
        deg_df = deg_df[deg_df['log2FoldChange'].notna()]
        deg_ranked = deg_df['log2FoldChange'].sort_values(ascending=False)

        # ── DEML ranked list (directional vote probability) ───
        deml_df     = pd.read_csv(
            f"{results_path}/Ensemble_ALL_genes_XGBoost_{cond}.csv")
        deml_ranked = deml_df.set_index('gene')['rank_score'].sort_values(ascending=False)

        for gs_name, gs_key in GSEA_GENE_SETS.items():
            print(f"  Running {gs_name}...")

            # DEG GSEA
            deg_res = run_prerank(deg_ranked, gs_key, 'DEG')
            if not deg_res.empty:
                deg_res['source']    = 'DEG'
                deg_res['condition'] = cond
                deg_res['gene_set']  = gs_name
                all_results.append(deg_res)
                print(f"    DEG  → {len(deg_res)} significant terms")

            # DEML GSEA
            deml_res = run_prerank(deml_ranked, gs_key, 'DEML')
            if not deml_res.empty:
                deml_res['source']    = 'DEML'
                deml_res['condition'] = cond
                deml_res['gene_set']  = gs_name
                all_results.append(deml_res)
                print(f"    DEML → {len(deml_res)} significant terms")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = f"{results_path}/GSEA_XGBoost_combined.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n✅ GSEA results saved: {out_path}")
        print(f"   Total significant terms: {len(combined):,}")
    else:
        print("\n⚠️  No significant GSEA results found.")


if __name__ == "__main__":
    run()
