# =============================================================
# src/gsea_analysis.py — Step 6
# Preranked GSEA (gseapy) for the DEG ranking and the DEML ranking of every classifier,
# on a common gene universe (genes with an embedding and non-missing DESeq2 statistics).
#   DEG ranking  = sign(log2FC) × −log10(p-value)
#   DEML ranking = 2 × out-of-fold vote probability − 1
# FDR is the permutation-based estimate of gseapy (gene set permutations).
# Terms with FDR < 0.25 are reported for discovery and FDR < 0.05 is marked separately.
# =============================================================

import os

import numpy as np
import pandas as pd
import gseapy as gp

import config as C
from src.utils.io_utils import exists, load_df, save_df, load_json
from src.label_preparation import load_expression, load_deg_tables
from src.ensemble_prediction import models_run, load_ensemble

G = C.CONFIG["gsea"]


def load_genesets():
    genesets = {}
    for short, lib in G["libraries"].items():
        src = C.GENESET_FILES.get(short, lib)
        genesets[short] = gp.read_gmt(src) if str(src).endswith(".gmt") else gp.get_library(name=src, organism="Human")
        print(f"  {short}: {len(genesets[short]):,} gene sets")
    return genesets


def universe(deg, ens, mr, cond):
    d = deg[cond]
    ok = d["log2FoldChange"].notna() & d["pvalue"].notna()
    u = pd.Index(ens[(mr[0], cond)]["gene"]).intersection(d.index[ok])
    return u[~u.duplicated()]


def deg_rank(deg, cond, u):
    d = deg[cond].loc[~deg[cond].index.duplicated()].loc[u]
    return np.sign(d["log2FoldChange"]) * -np.log10(d["pvalue"].clip(lower=1e-300))


def deml_rank(ens, m, cond, u):
    return ens[(m, cond)].set_index("gene").loc[u, "rank_score"]


def run_prerank(rnk, gene_sets):
    res = gp.prerank(rnk=rnk.sort_values(ascending=False), gene_sets=gene_sets,
                     min_size=G["min_size"], max_size=G["max_size"], permutation_num=G["permutations"],
                     seed=G["seed"], outdir=None, verbose=False, threads=max(1, (os.cpu_count() or 2) - 1))
    df = res.res2d.copy()
    for c in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def gsea_cached(source, cond, gs_short, rnk, gene_sets):
    path = f"{C.GSEA_DIR}/raw/{source}_{cond}_{gs_short}.csv"
    if exists(path):
        return load_df(path)
    df = run_prerank(rnk, gene_sets)
    df["source"], df["condition"], df["gene_set"], df["n_ranked"] = source, cond, gs_short, len(rnk)
    save_df(df, path)
    return df


def compare_deg_deml(gsea_all, selected, gs_names):
    def term_set(source, cond, gs_short, direction, fdr=G["fdr_discovery"]):
        d = gsea_all[(gsea_all["source"] == source) & (gsea_all["condition"] == cond) &
                     (gsea_all["gene_set"] == gs_short) & (gsea_all["FDR q-val"] < fdr)]
        d = d[d["NES"] > 0] if direction == "positive" else d[d["NES"] < 0]
        return set(d["Term"])

    rows = []
    for cond in C.CONFIG["conditions"]:
        for gs_short in gs_names:
            for direction in ["positive", "negative"]:
                a = term_set(f"DEML_{selected}", cond, gs_short, direction)
                b = term_set("DEG", cond, gs_short, direction)
                rows.append({"condition": cond, "gene_set": gs_short, "direction": direction,
                             "DEML": len(a), "DEG": len(b), "DEML_only": len(a - b), "DEG_only": len(b - a),
                             "shared": len(a & b), "DEML_only_list": "|".join(sorted(a - b)),
                             "DEG_only_list": "|".join(sorted(b - a)), "shared_list": "|".join(sorted(a & b))})
    cmp = pd.DataFrame(rows)
    save_df(cmp, f"{C.GSEA_DIR}/GSEA_{selected}_vs_DEG_comparison.csv")
    print(cmp.drop(columns=[c for c in cmp if c.endswith("_list")]).to_string(index=False))


def run():
    _, master_genes, _ = load_expression()
    deg = load_deg_tables(master_genes)
    mr = models_run()
    ens = {(m, c): load_ensemble(m, c) for m in mr for c in C.CONFIG["conditions"]}
    selected = load_json(f"{C.RES_DIR}/Model_selection.json")["selected_model"]
    genesets = load_genesets()

    all_gsea = []
    for cond in C.CONFIG["conditions"]:
        u = universe(deg, ens, mr, cond)
        print(f"\n{cond}: universe = {len(u):,} genes")
        rankings = {"DEG": deg_rank(deg, cond, u)} | {f"DEML_{m}": deml_rank(ens, m, cond, u) for m in mr}
        for src, rnk in rankings.items():
            for gs_short, gene_sets in genesets.items():
                df = gsea_cached(src, cond, gs_short, rnk, gene_sets)
                sig = df[df["FDR q-val"] < G["fdr_discovery"]]
                print(f"  {src:<18} {gs_short:<8} FDR<{G['fdr_discovery']}: {len(sig):>4} "
                      f"(NES+ {int((sig['NES'] > 0).sum())}, NES- {int((sig['NES'] < 0).sum())})")
                all_gsea.append(df)
    gsea_all = pd.concat(all_gsea, ignore_index=True)
    save_df(gsea_all, f"{C.GSEA_DIR}/GSEA_all_rankings_full.csv")

    sel = gsea_all[gsea_all["source"].isin(["DEG", f"DEML_{selected}"]) & (gsea_all["FDR q-val"] < G["fdr_discovery"])].copy()
    sel["source"] = sel["source"].str.replace(f"DEML_{selected}", "DEML", regex=False)
    save_df(sel, f"{C.GSEA_DIR}/GSEA_{selected}_combined.csv")
    print(f"\nSaved GSEA_all_rankings_full.csv and GSEA_{selected}_combined.csv (selected model: {selected})")
    compare_deg_deml(gsea_all, selected, list(genesets))
