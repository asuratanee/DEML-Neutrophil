# =============================================================
# src/ensemble_prediction.py — Step 5
# Genome-wide scores without in-sample predictions:
#   labeled genes   = out-of-fold. In each repeat a labeled gene is scored by the model
#                     whose outer validation fold contained it; the repeat-level scores
#                     are averaged.
#   unlabeled genes = mean prediction of all outer-fold models (none saw them in training).
# rank_score = 2 * score - 1 is the DEML ranking used for GSEA.
# =============================================================

import numpy as np
import pandas as pd

import config as C
from src.utils.io_utils import exists, load_df, save_df, load_npz
from src.trainers.nested_cv import load_condition, iter_fold_records, CV


def models_run():
    return [m for m in C.CONFIG["models"] if all(
        exists(f"{C.RES_DIR}/Predictions_allgenes_{m}_{c}.npz") for c in C.CONFIG["conditions"])]


def oof_scores(model_name, cond, calibrated=False):
    genes_all, _, idx_lab, y = load_condition(cond)
    arr = load_npz(f"{C.RES_DIR}/Predictions_allgenes_{model_name}_{cond}.npz")
    assert (arr["genes"] == genes_all).all()
    P = arr["P_cal"] if calibrated else arr["P"]
    s, n = np.zeros(len(genes_all)), np.zeros(len(genes_all), dtype=int)
    for k, (rec, fa) in enumerate(iter_fold_records(model_name, cond)):
        gi = idx_lab[fa["val_pos"]]
        s[gi] += P[k, gi]; n[gi] += 1
    assert (n[idx_lab] == CV["outer_repeats"]).all(), "each labeled gene must be scored once per repeat"
    assert (np.delete(n, idx_lab) == 0).all()
    score = P.mean(axis=0)
    score[idx_lab] = s[idx_lab] / n[idx_lab]
    category = np.full(len(genes_all), "non-DEG", dtype=object)
    category[idx_lab[y == 1]] = "up"; category[idx_lab[y == 0]] = "down"
    return pd.DataFrame({"gene": genes_all, "true_category": category, "score": score,
                         "n_models_used": np.where(n > 0, n, P.shape[0]),
                         "score_in_sample": P.mean(axis=0), "score_std_25": P.std(axis=0)})


def load_ensemble(m, cond):
    df = load_df(f"{C.RES_DIR}/Ensemble_OOF_{m}_{cond}.csv")
    for c in df.columns:              # scores were computed in float32; restore the exact values
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def run():
    mr = models_run()
    print(f"Models with complete outputs: {mr}")
    for m in mr:
        for cond in C.CONFIG["conditions"]:
            path = f"{C.RES_DIR}/Ensemble_OOF_{m}_{cond}.csv"
            if exists(path):
                continue
            df = oof_scores(m, cond)
            df["score_calibrated"] = oof_scores(m, cond, calibrated=True)["score"].values
            df["rank_score"] = 2 * df["score"] - 1
            save_df(df, path)
            print(f"  saved {path}")
