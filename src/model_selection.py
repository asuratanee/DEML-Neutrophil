# =============================================================
# src/model_selection.py — Step 5 (continued)
# Outer-fold performance summary and the pre-specified selection rule:
# a classifier is eligible if its mean outer-fold AUC is within 0.03 of the best
# classifier in every condition; among eligible classifiers the one with the highest
# out-of-fold separation score (mean score of up genes minus mean score of down genes),
# averaged over conditions, is selected.
# =============================================================

import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from src.utils.io_utils import load_df, save_df, save_json
from src.ensemble_prediction import models_run, load_ensemble

METRICS = ["auc", "aupr_up", "aupr_down", "f1_up", "f1_down", "f1_macro", "mcc", "balanced_acc", "accuracy"]


def performance_summary(mr, ens):
    perf_rows = []
    for m in mr:
        for cond in C.CONFIG["conditions"]:
            fr = load_df(f"{C.RES_DIR}/Fold_results_{m}_{cond}.csv")
            e = ens[(m, cond)]; lab = e[e["true_category"].isin(["up", "down"])]
            row = {"model": m, "condition": cond, "prevalence_up": fr["prevalence_up"].mean(),
                   "n_train_balanced": int(fr["n_train_balanced"].median()), "n_val": int(fr["n_val"].median()),
                   "pooled_oof_auc": roc_auc_score((lab["true_category"] == "up").astype(int), lab["score"])}
            for k in METRICS:
                row[f"{k}_mean"], row[f"{k}_sd"] = fr[k].mean(), fr[k].std()
            perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    save_df(perf, f"{C.RES_DIR}/Summary_performance_all_models.csv")
    show = perf[["model", "condition", "n_train_balanced", "n_val", "prevalence_up"]].copy()
    for k in ["auc", "aupr_up", "aupr_down", "f1_macro", "mcc"]:
        show[k] = perf[f"{k}_mean"].map("{:.3f}".format) + " ± " + perf[f"{k}_sd"].map("{:.3f}".format)
    print(show.to_string(index=False))
    return perf


def select_model(mr, ens, perf):
    sep_rows = []
    for m in mr:
        for cond in C.CONFIG["conditions"]:
            e = ens[(m, cond)]
            up, dn = e["true_category"] == "up", e["true_category"] == "down"
            sep_rows.append({"model": m, "condition": cond,
                             "sep_oof": e.loc[up, "score"].mean() - e.loc[dn, "score"].mean(),
                             "sep_in_sample": e.loc[up, "score_in_sample"].mean() - e.loc[dn, "score_in_sample"].mean(),
                             "auc_mean": perf.query("model == @m and condition == @cond")["auc_mean"].item()})
    sep = pd.DataFrame(sep_rows)

    tol = C.CONFIG["selection"]["auc_tolerance"]
    best_auc = sep.groupby("condition")["auc_mean"].max()
    sep["eligible_in_condition"] = sep.apply(lambda r: r["auc_mean"] >= best_auc[r["condition"]] - tol, axis=1)
    elig = sep.groupby("model")["eligible_in_condition"].all()
    avg_sep = sep.groupby("model")["sep_oof"].mean()
    eligible = elig[elig].index.tolist()
    if eligible:
        selected = avg_sep.loc[eligible].idxmax(); fallback = False
    else:
        selected = sep.groupby("model")["auc_mean"].mean().idxmax(); fallback = True

    save_df(sep, f"{C.RES_DIR}/Separation_scores.csv")
    save_json({"selected_model": selected, "eligible_models": eligible, "fallback_to_auc": fallback,
               "avg_sep_oof": avg_sep.to_dict(), "best_auc_by_condition": best_auc.to_dict(),
               "rule": C.CONFIG["selection"]["rule"]}, f"{C.RES_DIR}/Model_selection.json")

    print(sep.pivot(index="model", columns="condition", values="sep_oof").round(3).assign(avg=avg_sep.round(3)))
    print(f"\nEligible (AUC within {tol} of best in every condition): {eligible}")
    print(f"SELECTED MODEL: {selected}" + ("  (fallback: no model eligible, chose highest mean AUC)" if fallback else ""))
    return selected


def run():
    mr = models_run()
    ens = {(m, c): load_ensemble(m, c) for m in mr for c in C.CONFIG["conditions"]}
    perf = performance_summary(mr, ens)
    return select_model(mr, ens, perf)
