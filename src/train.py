# =============================================================
# src/train.py
# Unified training entry point for all four classifiers.
# Usage (via main.py):
#   python main.py --only 4                    # train all classifiers
#   python main.py --only 4 --classifier xgboost
# =============================================================

import os
import gc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from config import (FEATURES_PATH, MODELS_PATH, RESULTS_PATH, CONDITIONS,
                    RANDOM_SEED, EPOCHS, PATIENCE, EARLY_STOPPING_ROUNDS)
from src.utils.data_utils import load_data
from src.trainers.base_trainer import run_cv
from src.trainers.models import create_model

ALL_CLASSIFIERS = ['cnn', 'mlp', 'xgboost', 'lightgbm']


# ── Fit functions ─────────────────────────────────────────────

def fit_nn(classifier, params, X_tr, y_tr, X_val, y_val, seed):
    """Fit CNN or MLP."""
    tf.keras.utils.set_random_seed(seed)
    if classifier == 'cnn':
        X_tr  = np.expand_dims(X_tr,  axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        input_shape = (X_tr.shape[1], 1)
        model = create_model('cnn', params, input_shape=input_shape, seed=seed)
    else:
        input_dim = X_tr.shape[1]
        model = create_model('mlp', params, input_dim=input_dim, seed=seed)

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=params['batch_size'],
        callbacks=[EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                  restore_best_weights=True, verbose=0)],
        verbose=0,
    )
    return model


def fit_xgboost(params, X_tr, y_tr, X_val, y_val, seed):
    model = create_model('xgboost', params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_lightgbm(params, X_tr, y_tr, X_val, y_val, seed):
    model = create_model('lightgbm', params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS,
                                             verbose=False)])
    return model


# ── Predict functions ─────────────────────────────────────────

def predict_nn(classifier, model, X):
    if classifier == 'cnn':
        X = np.expand_dims(X, axis=-1)
    return model.predict(X, verbose=0).ravel()


def predict_tree(model, X):
    return model.predict_proba(X)[:, 1]


# ── Save functions ────────────────────────────────────────────

def save_nn(model, path):
    model.save(path + '.keras')


def save_xgboost(model, path):
    model.save_model(path + '.json')


def save_lightgbm(model, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(model, f)


# ── Main training function ────────────────────────────────────

def train_classifier(classifier, features_path=FEATURES_PATH,
                     models_path=MODELS_PATH, results_path=RESULTS_PATH):
    print(f"\n{'='*55}")
    print(f"  Training: {classifier.upper()}")
    print(f"{'='*55}")

    all_folds  = []
    all_models = []

    for cond in CONDITIONS:
        print(f"\n{'─'*55}\n  {cond}\n{'─'*55}")
        X, y, genes = load_data(features_path, cond)

        # Define fit, predict, save functions for this classifier
        if classifier == 'cnn':
            fit_fn     = lambda params, X_tr, y_tr, X_val, y_val, seed: fit_nn('cnn', params, X_tr, y_tr, X_val, y_val, seed)
            predict_fn = lambda model, X: predict_nn('cnn', model, X)
            save_fn    = save_nn
        elif classifier == 'mlp':
            fit_fn     = lambda params, X_tr, y_tr, X_val, y_val, seed: fit_nn('mlp', params, X_tr, y_tr, X_val, y_val, seed)
            predict_fn = lambda model, X: predict_nn('mlp', model, X)
            save_fn    = save_nn
        elif classifier == 'xgboost':
            fit_fn     = lambda params, X_tr, y_tr, X_val, y_val, seed: fit_xgboost(params, X_tr, y_tr, X_val, y_val, seed)
            predict_fn = predict_tree
            save_fn    = save_xgboost
        elif classifier == 'lightgbm':
            fit_fn     = lambda params, X_tr, y_tr, X_val, y_val, seed: fit_lightgbm(params, X_tr, y_tr, X_val, y_val, seed)
            predict_fn = predict_tree
            save_fn    = save_lightgbm

        folds, models = run_cv(classifier, X, y, models_path, results_path,
                                cond, fit_fn, predict_fn, save_fn)
        all_folds.append(folds)
        all_models.append(models)

        print(f"\n  → {cond}: "
              f"AUC={folds['auc'].mean():.4f}  "
              f"AUPR={folds['aupr'].mean():.4f}  "
              f"ACC={folds['acc'].mean():.4f}")

    pd.concat(all_folds).to_csv(
        f"{results_path}/Fold_results_{classifier.upper()}.csv", index=False)
    pd.concat(all_models).to_csv(
        f"{results_path}/Model_registry_{classifier.upper()}.csv", index=False)

    print(f"\n✅ {classifier.upper()} training complete.")


def run(classifier='all', features_path=FEATURES_PATH,
        models_path=MODELS_PATH, results_path=RESULTS_PATH):
    classifiers = ALL_CLASSIFIERS if classifier == 'all' else [classifier]
    for clf in classifiers:
        train_classifier(clf, features_path, models_path, results_path)
