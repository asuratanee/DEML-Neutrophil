# =============================================================
# src/trainers/nested_cv.py — nested cross-validation engine (Step 4)
#
# For each condition and classifier:
# 1. Outer loop: 5 repeats × 5 stratified folds over all labeled genes. The outer
#    validation fold keeps the natural up/down ratio.
# 2. Undersampling: only the outer training portion is undersampled to 1:1.
# 3. Inner loop: Optuna (TPE) maximizes the mean AUC over a 3-fold stratified split of
#    the balanced outer training portion. Early stopping uses the inner fold only.
# 4. Final model: refit on the whole balanced outer training portion with the best
#    hyperparameters and a fixed number of trees or epochs (median of the best
#    iterations over the 3 inner folds). No validation data is passed to fit.
# 5. Evaluation: the outer validation fold is used once, after the model is fixed,
#    with a fixed 0.5 threshold.
# 6. Platt scaling is fitted on the inner out-of-fold predictions of the best trial
#    (used only to check that calibration does not change the gene ordering).
# 7. The final model also predicts every gene with an embedding; Step 5 uses these
#    predictions so that labeled genes are scored out-of-fold only.
#
# Every outer fold writes a JSON record and an NPZ file. A fold that already has both
# files is skipped, so an interrupted run can resume.
# TensorFlow models (CNN, MLP) run each outer fold in a fresh Python process
# (src/trainers/nn_worker.py) to avoid memory growth across folds.
# =============================================================

import os
import sys
import gc
import json
import time
import subprocess

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.trial import FixedTrial
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
                             balanced_accuracy_score, precision_score, recall_score, accuracy_score)

import config as C
from src.utils.io_utils import exists, save_json, load_json, save_npz, load_npz, save_df, load_df, fold_seed
from src.node2vec_embedding import emb512_path

optuna.logging.set_verbosity(optuna.logging.WARNING)

CV = C.CONFIG["cv"]
THR = C.CONFIG["decision_threshold"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_condition(cond):
    E = load_df(emb512_path(cond))
    lab = load_df(f"{C.FEAT_DIR}/labels_UpvDown_{cond}.csv", index_col=0)
    genes_all = E.index.values.astype(str)
    X_all = E.values.astype(np.float32)
    idx_lab = np.where(np.isin(genes_all, lab.index.values.astype(str)))[0]
    y = lab.loc[genes_all[idx_lab], "label"].astype(int).values
    return genes_all, X_all, idx_lab, y


def undersample(pos_idx, y_sub, seed):
    """pos_idx: positions (into the labeled arrays) of the outer training portion."""
    rng = np.random.default_rng(seed)
    p_, n_ = pos_idx[y_sub == 1], pos_idx[y_sub == 0]
    k = min(len(p_), len(n_))
    return rng.permutation(np.concatenate([rng.choice(p_, k, replace=False),
                                           rng.choice(n_, k, replace=False)]))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(pred, y):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(pred).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(pred, ab):
    return 1.0 / (1.0 + np.exp(-(ab[0] * _logit(pred) + ab[1])))


def outer_metrics(y_true, p):
    yhat = (p >= THR).astype(int)
    return {"auc": roc_auc_score(y_true, p),
            "aupr_up": average_precision_score(y_true, p),
            "aupr_down": average_precision_score(1 - y_true, 1 - p),
            "prevalence_up": float(y_true.mean()),
            "f1_up": f1_score(y_true, yhat, zero_division=0),
            "f1_down": f1_score(1 - y_true, 1 - yhat, zero_division=0),
            "f1_macro": f1_score(y_true, yhat, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_true, yhat),
            "balanced_acc": balanced_accuracy_score(y_true, yhat),
            "accuracy": accuracy_score(y_true, yhat),
            "precision_up": precision_score(y_true, yhat, zero_division=0),
            "recall_up": recall_score(y_true, yhat, zero_division=0)}


class ModelSpec:
    """space(trial) -> params; fit_inner(params, Xa, ya, Xb, yb, seed) -> (pred_b, best_iter or None);
    fit_final(params, n_iter, X, y, seed) -> model; predict(model, X) -> P(up)."""
    def __init__(self, name, space, fit_inner, fit_final, predict, cleanup=None, isolate=False):
        self.name, self.space, self.fit_inner, self.fit_final, self.predict = name, space, fit_inner, fit_final, predict
        self.cleanup = cleanup or (lambda: None)
        self.n_trials = C.CONFIG["optuna"]["trials"][name]
        self.isolate = isolate              # True: each outer fold runs in a fresh Python process (TensorFlow models)


def fold_paths(model_name, cond, rep, fold):
    base = f"{C.TRAIN_DIR}/{model_name}/{cond}/R{rep + 1}F{fold + 1}"
    return base + ".json", base + ".npz"


def run_one_fold(spec, model_name, cond, rep, fold, data):
    """One outer fold: undersample, nested tuning, final fit, evaluation, save. Returns a one-line summary."""
    genes_all, X_all, idx_lab, y = data
    X_lab, g_lab = X_all[idx_lab], genes_all[idx_lab]
    skf = StratifiedKFold(CV["outer_folds"], shuffle=True, random_state=CV["seed"] + rep)
    tr, va = list(skf.split(X_lab, y))[fold]
    jpath, npath = fold_paths(model_name, cond, rep, fold)
    t0 = time.time()
    fs = fold_seed(rep, fold)
    tr_bal = undersample(tr, y[tr], fs)
    Xtr, ytr = X_lab[tr_bal], y[tr_bal]
    inner = list(StratifiedKFold(CV["inner_folds"], shuffle=True, random_state=fs).split(Xtr, ytr))

    def objective(trial):
        params = spec.space(trial)
        aucs, iters, oof = [], [], np.full(len(ytr), np.nan)
        for k, (a, b) in enumerate(inner):
            pred_b, it = spec.fit_inner(params, Xtr[a], ytr[a], Xtr[b], ytr[b], fs * 10 + k)
            aucs.append(roc_auc_score(ytr[b], pred_b)); iters.append(it); oof[b] = pred_b
            spec.cleanup()
        trial.set_user_attr("iters", iters)
        trial.set_user_attr("inner_oof", oof.tolist())
        trial.set_user_attr("inner_aucs", aucs)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=fs))
    study.optimize(objective, n_trials=spec.n_trials, show_progress_bar=False)
    bt = study.best_trial
    params = spec.space(FixedTrial(bt.params))
    its = [i for i in bt.user_attrs["iters"] if i is not None]
    n_iter = int(max(1, round(float(np.median(its))))) if its else None
    platt = fit_platt(np.array(bt.user_attrs["inner_oof"]), ytr)

    model = spec.fit_final(params, n_iter, Xtr, ytr, fs)      # outer validation fold not passed
    p_val = spec.predict(model, X_lab[va])
    p_all = spec.predict(model, X_all)
    spec.cleanup()

    rec = {"model": model_name, "condition": cond, "repeat": rep + 1, "fold": fold + 1, "seed": fs,
           "best_params": bt.params, "best_inner_auc": bt.value, "inner_aucs": bt.user_attrs["inner_aucs"],
           "n_iter_final": n_iter, "inner_best_iters": bt.user_attrs["iters"], "platt": platt,
           "metrics": outer_metrics(y[va], p_val),
           "n_train": len(tr), "n_train_up": int(y[tr].sum()), "n_train_down": int((y[tr] == 0).sum()),
           "n_train_balanced": len(tr_bal), "n_val": len(va),
           "n_val_up": int(y[va].sum()), "n_val_down": int((y[va] == 0).sum()),
           "genes_train": g_lab[tr].tolist(), "genes_train_balanced": g_lab[tr_bal].tolist(),
           "genes_val": g_lab[va].tolist(),
           "inner_val_genes": [g_lab[tr_bal][b].tolist() for _, b in inner],
           "outer_val_used_in_fit": False, "outer_val_used_in_tuning": False,
           "runtime_s": time.time() - t0}
    save_npz(npath, val_pos=va.astype(np.int32), p_val=p_val.astype(np.float32),
             p_all=p_all.astype(np.float32), p_all_cal=apply_platt(p_all, platt).astype(np.float32))
    save_json(rec, jpath)
    m = rec["metrics"]
    return (f"    R{rep + 1}F{fold + 1}: AUC={m['auc']:.4f} AUPR(up)={m['aupr_up']:.4f} "
            f"AUPR(down)={m['aupr_down']:.4f} F1(macro)={m['f1_macro']:.4f} MCC={m['mcc']:.4f} | "
            f"inner AUC={bt.value:.4f}, n_iter={n_iter} | {(time.time() - t0) / 60:.1f} min")


def _run_worker(model_name, cond, rep, fold):
    """Run one outer fold of a TensorFlow model in a fresh Python process."""
    r = subprocess.run([sys.executable, "-m", "src.trainers.nn_worker", model_name, cond, str(rep), str(fold)],
                       capture_output=True, text=True, cwd=REPO_ROOT, env=os.environ.copy())
    res = [l for l in r.stdout.splitlines() if l.startswith("RESULT::")]
    if r.returncode != 0 or not res:
        raise RuntimeError(f"Worker failed (exit code {r.returncode}).\n--- stderr (last 4000 characters) ---\n"
                           + r.stderr[-4000:])
    return res[-1][len("RESULT::"):]


def run_nested_cv(spec, cond):
    model_name = spec.name
    data = load_condition(cond)
    genes_all, y = data[0], data[3]
    print(f"\n  {model_name} | {cond}: labeled {len(y):,} (Up={y.sum():,}, Down={(y == 0).sum():,}), "
          f"all genes {len(genes_all):,}, trials/fold = {spec.n_trials}"
          + (" | fresh Python process per fold" if spec.isolate else ""))
    t_all = time.time()
    for rep in range(CV["outer_repeats"]):
        for fold in range(CV["outer_folds"]):
            jpath, npath = fold_paths(model_name, cond, rep, fold)
            if exists(jpath) and exists(npath):
                continue
            if spec.isolate:
                line = _run_worker(model_name, cond, rep, fold)
            else:
                line = run_one_fold(spec, model_name, cond, rep, fold, data)
            print(line)
            gc.collect()
    collect_model_outputs(model_name, cond)
    print(f"  {model_name} | {cond} done in {(time.time() - t_all) / 60:.1f} min")


def iter_fold_records(model_name, cond):
    for rep in range(CV["outer_repeats"]):
        for fold in range(CV["outer_folds"]):
            jpath, npath = fold_paths(model_name, cond, rep, fold)
            yield load_json(jpath), load_npz(npath)


def collect_model_outputs(model_name, cond):
    """Stack per-fold outputs into one file per model and condition."""
    genes_all, _, idx_lab, y = load_condition(cond)
    g_lab = genes_all[idx_lab]
    fold_rows, val_rows, P, PC = [], [], [], []
    for rec, arr in iter_fold_records(model_name, cond):
        fold_rows.append({k: rec[k] for k in ["model", "condition", "repeat", "fold", "best_inner_auc",
                                               "n_iter_final", "n_train", "n_train_up", "n_train_down",
                                               "n_train_balanced", "n_val", "n_val_up", "n_val_down"]}
                         | rec["metrics"] | {"best_params": json.dumps(rec["best_params"])})
        for pos, p in zip(arr["val_pos"], arr["p_val"]):
            val_rows.append({"repeat": rec["repeat"], "fold": rec["fold"], "gene": g_lab[pos],
                             "true_label": int(y[pos]), "predicted_prob": float(p)})
        P.append(arr["p_all"]); PC.append(arr["p_all_cal"])
    save_df(pd.DataFrame(fold_rows), f"{C.RES_DIR}/Fold_results_{model_name}_{cond}.csv")
    save_df(pd.DataFrame(val_rows), f"{C.RES_DIR}/Predictions_outer_val_{model_name}_{cond}.csv")
    save_npz(f"{C.RES_DIR}/Predictions_allgenes_{model_name}_{cond}.npz",
             genes=genes_all.astype("U"), P=np.vstack(P), P_cal=np.vstack(PC))
