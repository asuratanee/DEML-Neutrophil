# =============================================================
# src/trainers/hyperparams.py
# Optuna hyperparameter search spaces for all four classifiers.
# =============================================================

from config import RANDOM_SEED, EARLY_STOPPING_ROUNDS


def get_parameter_space(trial, classifier):
    """Return hyperparameter space for the given classifier."""
    if classifier == 'cnn':
        return _cnn_space(trial)
    elif classifier == 'mlp':
        return _mlp_space(trial)
    elif classifier == 'xgboost':
        return _xgboost_space(trial)
    elif classifier == 'lightgbm':
        return _lightgbm_space(trial)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


def _cnn_space(trial):
    return {
        'initial_filters': trial.suggest_categorical('initial_filters', [16, 32, 64]),
        'kernel_size'    : trial.suggest_categorical('kernel_size', [3, 4, 5]),
        'conv_dropout'   : trial.suggest_categorical('conv_dropout', [0.1, 0.2, 0.3]),
        'dense_units'    : trial.suggest_categorical('dense_units', [32, 64, 96, 128]),
        'dense_dropout'  : trial.suggest_categorical('dense_dropout', [0.1, 0.2, 0.3]),
        'learning_rate'  : trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3]),
        'second_conv'    : trial.suggest_categorical('second_conv', [True, False]),
        'batch_size'     : trial.suggest_categorical('batch_size', [8, 16, 32]),
    }


def _mlp_space(trial):
    return {
        'hidden1'        : trial.suggest_categorical('hidden1', [128, 256, 512]),
        'hidden2'        : trial.suggest_categorical('hidden2', [64, 128, 256]),
        'hidden3'        : trial.suggest_categorical('hidden3', [32, 64, 96, 128]),
        'dropout1'       : trial.suggest_categorical('dropout1', [0.1, 0.2, 0.3]),
        'dropout2'       : trial.suggest_categorical('dropout2', [0.1, 0.2, 0.3]),
        'dropout3'       : trial.suggest_categorical('dropout3', [0.1, 0.2, 0.3]),
        'l1'             : trial.suggest_float('l1', 1e-5, 1e-3, log=True),
        'l2'             : trial.suggest_float('l2', 1e-5, 1e-3, log=True),
        'use_third_layer': trial.suggest_categorical('use_third_layer', [True, False]),
        'learning_rate'  : trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3]),
        'batch_size'     : trial.suggest_categorical('batch_size', [8, 16, 32]),
    }


def _xgboost_space(trial):
    return {
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 7),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 1000, step=100),
        'gamma'            : trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective'        : 'binary:logistic',
        'eval_metric'      : 'auc',
        'random_state'     : RANDOM_SEED,
        'tree_method'      : 'hist',
        'verbosity'        : 0,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
    }


def _lightgbm_space(trial):
    return {
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 1000, step=100),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective'        : 'binary',
        'metric'           : 'auc',
        'random_state'     : RANDOM_SEED,
        'verbosity'        : -1,
        'force_col_wise'   : True,
        'boosting_type'    : 'gbdt',
    }
