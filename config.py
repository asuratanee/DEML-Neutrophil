# =============================================================
# config.py — DEML-Neutrophil pipeline configuration
# =============================================================
# Set DATA_BACKEND to "gcs" to read and write Google Cloud Storage,
# or "local" to use files on disk (for example the data in ./data/example).
#
# QUICK_DEMO (or `python main.py --quick`) shrinks the number of repeats,
# Optuna trials, epochs, walks and permutations so that the whole pipeline
# finishes in minutes on the example data. Results from QUICK_DEMO are only
# a check that the code runs. The manuscript used QUICK_DEMO = False.
# =============================================================

import os
import copy
import datetime

DATA_BACKEND = "local"  # "gcs" or "local"

GCS_BUCKET = "gs://your-bucket/your-project"
LOCAL_ROOT = "./data/example"

QUICK_DEMO = os.environ.get("DEML_QUICK_DEMO", "0") == "1"

# ── Paths ─────────────────────────────────────────────────────
OUT = GCS_BUCKET if DATA_BACKEND == "gcs" else LOCAL_ROOT
DATA      = f"{OUT}/Data"          # Normalised_counts_GMT.csv and the three *_DEGs.csv files
NET_DIR   = f"{OUT}/Networks"
FEAT_DIR  = f"{OUT}/Features"
TRAIN_DIR = f"{OUT}/Training"
RES_DIR   = f"{OUT}/Results"
GSEA_DIR  = f"{OUT}/GSEA"

# ── Locked analysis configuration (the settings used for the manuscript) ──
CONFIG = {
    "version": "DEML-primary-v1",
    "samples": {
        "UT":     ["UT_1", "UT_2", "UT_3", "UT_5", "UT_6"],
        "GMCSF":  ["GMCSF_1", "GMCSF_2", "GMCSF_4"],
        "TNF":    ["TNF_1", "TNF_2", "TNF_4"],
        "GM_TNF": ["GMT_1", "GMT_2", "GMT_3"],
    },
    "deg_files": {"GM_TNF": "GMTvUT_DEGs.csv", "GMCSF": "GMCSFvUT_DEGs.csv", "TNF": "TNFvUT_DEGs.csv"},
    "conditions": ["GM_TNF", "GMCSF", "TNF"],
    "labels": {"padj_max": 0.05, "padj_rule": "padj <= 0.05",
               "lfc_min": 1.0, "lfc_rule": "up: log2FC >= 1.0; down: log2FC <= -1.0"},
    "wgcna": {"network_type": "signed", "correlation": "pearson",
              "transform": "log2(normalized counts + 1)",
              "beta": 15, "beta_rule": "pickSoftThreshold on UT; 15 used if the estimate is NA",
              "tau": 0.4, "tau_rule": "edge kept if adjacency >= 0.4",
              "per_condition": True},
    "node2vec": {"p": 1, "q": 1, "walk_length": 20, "num_walks": 10, "dimensions": 256,
                 "window": 5, "min_count": 0, "epochs": 10, "sg": 1, "negative": 5,
                 "seed": 42, "workers": 1, "l2_normalize": True},
    "cv": {"outer_repeats": 5, "outer_folds": 5, "inner_folds": 3, "seed": 42,
           "undersampling": "outer training portion only (majority class down to minority size)"},
    "optuna": {"sampler": "TPE", "objective": "mean AUC over inner folds",
               "trials": {"CNN": 10, "MLP": 10, "XGBoost": 50, "LightGBM": 50,
                          "LogReg": 50, "RandomForest": 50, "SVM": 50}},
    "early_stopping": {"tree_rounds": 20, "nn_patience": 10, "nn_max_epochs": 100,
                       "final_iterations": "median of best iterations over inner folds of the best trial"},
    "decision_threshold": 0.5,
    "models": ["CNN", "MLP", "XGBoost", "LightGBM", "LogReg", "RandomForest", "SVM"],
    "scoring": {"labeled_genes": "out-of-fold: mean over repeats of the prediction from the model whose outer validation fold contained the gene",
                "unlabeled_genes": "mean over all 25 outer-fold models",
                "rank_score": "2 * score - 1"},
    "selection": {"rule": "eligible if mean outer-fold AUC >= (best mean AUC - auc_tolerance) in every condition; "
                          "among eligible models choose the highest separation score averaged over conditions; "
                          "separation = mean OOF score of up genes - mean OOF score of down genes",
                  "auc_tolerance": 0.03},
    "gsea": {"tool": "gseapy prerank", "libraries": {"Hallmark": "MSigDB_Hallmark_2020",
                                                     "KEGG": "KEGG_2021_Human",
                                                     "GO_BP": "GO_Biological_Process_2023"},
             "min_size": 10, "max_size": 500, "permutations": 1000, "seed": 42,
             "fdr_discovery": 0.25, "fdr_high": 0.05, "universe": "common to DEG and DEML rankings"},
    "deg_ranking": "sign(log2FC) * -log10(pvalue), pvalue clipped at 1e-300",
}

# Gene set libraries are downloaded from Enrichr by gseapy. Without internet access, put
# local .gmt files here, for example {"Hallmark": "./genesets/h.all.v2023.2.Hs.symbols.gmt"}.
GENESET_FILES = {}

# Device for XGBoost ("cuda" on a GPU machine). The manuscript ran XGBoost on CPU.
XGB_DEVICE = "cpu"

# ── QUICK_DEMO overrides (never used for the manuscript) ──────
if QUICK_DEMO:
    CONFIG = copy.deepcopy(CONFIG)
    CONFIG["version"] = "DEML-quick-demo"
    CONFIG["node2vec"].update({"num_walks": 2, "epochs": 2, "dimensions": 32})
    CONFIG["cv"].update({"outer_repeats": 1})
    CONFIG["optuna"]["trials"] = {m: 2 for m in CONFIG["models"]}
    CONFIG["early_stopping"].update({"nn_max_epochs": 5, "nn_patience": 2})
    CONFIG["gsea"].update({"permutations": 100})
    CONFIG["gsea"]["libraries"] = {"Hallmark": "MSigDB_Hallmark_2020"}
    TRAIN_DIR = f"{OUT}/Training_quick_demo"
    RES_DIR   = f"{OUT}/Results_quick_demo"
    GSEA_DIR  = f"{OUT}/GSEA_quick_demo"
    FEAT_DIR  = f"{OUT}/Features_quick_demo"

N_EMB = 2 * CONFIG["node2vec"]["dimensions"]   # classifier input size (UT + condition embedding)
