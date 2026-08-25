# =============================================================
# src/trainers/base_trainer.py
# Base CV training loop shared by all classifiers.
# =============================================================

import numpy as np
import pandas as pd
import os
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from config import RANDOM_SEED, N_REPEATS, N_FOLDS, N_TRIALS_TREE, N_TRIALS_NN
from src.utils.data_utils import balance_dataset
from src.utils.eval_utils import evaluate_fold
from src.trainers.hyperparams import get_parameter_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_n_trials(classifier):
    return N_TRIALS_NN if classifier in ('cnn', 'mlp') else N_TRIALS_TREE


def run_cv(classifier, X, y, models_path, results_path, cond,
           fit_fn, predict_fn, save_fn):
    """
    Generic 5x5 repeated stratified CV loop.

    Parameters
    ----------
    classifier : str
        One of 'cnn', 'mlp', 'xgboost', 'lightgbm'
    X, y : arrays
        Features and labels (Up=1, Down=0)
    fit_fn : callable(params, X_tr, y_tr, X_val, y_val, seed) -> model
        Train a model and return it
    predict_fn : callable(model, X) -> np.ndarray
        Return predicted probabilities
    save_fn : callable(model, path)
        Save model to path
    """
    sampler      = TPESampler(seed=RANDOM_SEED)
    pruner       = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    n_trials     = get_n_trials(classifier)
    fold_results = []
    saved_models = []

    for rep in range(N_REPEATS):
        rep_seed = RANDOM_SEED + rep
        X_bal, y_bal = balance_dataset(X, y, rep_seed)

        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep_seed)

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_bal, y_bal)):
            fold_seed = rep_seed + fold
            X_tr, X_val = X_bal[tr_idx], X_bal[val_idx]
            y_tr, y_val = y_bal[tr_idx], y_bal[val_idx]

            # Optuna hyperparameter search
            def objective(trial):
                params = get_parameter_space(trial, classifier)
                model  = fit_fn(params, X_tr, y_tr, X_val, y_val, fold_seed + trial.number)
                y_pred = predict_fn(model, X_val)
                auc    = roc_auc_score(y_val, y_pred)
                del model
                gc.collect()
                return auc

            study = optuna.create_study(direction='maximize',
                                        sampler=sampler, pruner=pruner)
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            # Retrain with best hyperparameters
            final_model = fit_fn(study.best_params, X_tr, y_tr, X_val, y_val, fold_seed)
            y_pred      = predict_fn(final_model, X_val)

            # Save model
            model_path = f"{models_path}/{classifier.upper()}/{cond}_Rep{rep+1}_Fold{fold+1}"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            save_fn(final_model, model_path)

            saved_models.append({
                'condition': cond, 'repeat': rep+1, 'fold': fold+1,
                'model_path': model_path,
                'best_params': str(study.best_params),
            })

            metrics = evaluate_fold(y_val, y_pred)
            metrics.update({'condition': cond, 'repeat': rep+1, 'fold': fold+1})
            fold_results.append(metrics)

            print(f"  Rep{rep+1} Fold{fold+1}: "
                  f"AUC={metrics['auc']:.4f}  "
                  f"AUPR={metrics['aupr']:.4f}  "
                  f"ACC={metrics['acc']:.4f}")

            del final_model
            gc.collect()

    return pd.DataFrame(fold_results), pd.DataFrame(saved_models)
