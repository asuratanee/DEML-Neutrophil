# =============================================================
# src/trainers/models.py
# Search spaces, inner fit (with early stopping on the inner fold), final fit
# (fixed number of iterations) and prediction for the seven classifiers:
# CNN, MLP (TensorFlow), XGBoost, LightGBM, logistic regression, random forest, SVM.
# =============================================================

import os
import gc

import numpy as np

import config as C
from src.trainers.nested_cv import ModelSpec

ES = C.CONFIG["early_stopping"]
PREDICT_BATCH = 1024
N_JOBS = os.cpu_count()
N_INPUT = C.N_EMB          # 512 with the manuscript settings (256 UT + 256 condition)

# ── TensorFlow helpers (CNN, MLP) ─────────────────────────────
_TF = {}


def tf_init():
    # Import and configure TensorFlow on first use. Called inside worker processes only.
    if "tf" not in _TF:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.experimental.enable_op_determinism()
        _TF["tf"] = tf
    return _TF["tf"]


def set_all_seeds(seed):
    tf_init().keras.utils.set_random_seed(int(seed))


def tf_cleanup():
    if "tf" in _TF:
        _TF["tf"].keras.backend.clear_session()
    gc.collect()


def _batched_predict(m, X):
    tf = tf_init()
    out = [m(tf.convert_to_tensor(X[i:i + PREDICT_BATCH]), training=False).numpy().ravel()
           for i in range(0, len(X), PREDICT_BATCH)]
    return np.concatenate(out)


def _nn_fit_inner(builder, reshape):
    def fit_inner(params, Xa, ya, Xb, yb, seed):
        tf = tf_init()
        tf_cleanup()
        m = builder(params, seed)
        h = m.fit(reshape(Xa), ya, validation_data=(reshape(Xb), yb), epochs=ES["nn_max_epochs"],
                  batch_size=params["batch_size"], verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ES["nn_patience"],
                                                              restore_best_weights=True)])
        best_epoch = int(np.argmin(h.history["val_loss"])) + 1
        pred = _batched_predict(m, reshape(Xb))
        del m, h
        return pred, best_epoch
    return fit_inner


def _nn_fit_final(builder, reshape):
    def fit_final(params, n_iter, X, y, seed):
        tf_init()
        tf_cleanup()
        m = builder(params, seed)
        m.fit(reshape(X), y, epochs=n_iter, batch_size=params["batch_size"], verbose=0)
        return m
    return fit_final


def _nn_predict(reshape):
    return lambda m, X: _batched_predict(m, reshape(X))


_id = lambda X: X


# ── CNN ───────────────────────────────────────────────────────
def create_cnn(input_shape, params, seed):
    tf = tf_init(); layers = tf.keras.layers
    set_all_seeds(seed)
    init = tf.keras.initializers.GlorotUniform(seed=int(seed))
    m = tf.keras.Sequential([layers.Input(shape=input_shape)])
    f = params["initial_filters"]
    m.add(layers.Conv1D(f, params["kernel_size"], activation="relu", strides=2, kernel_initializer=init))
    m.add(layers.BatchNormalization()); m.add(layers.Dropout(params["conv_dropout"], seed=int(seed)))
    if params["second_conv"]:
        m.add(layers.Conv1D(f * 2, params["kernel_size"], activation="relu", strides=2, kernel_initializer=init))
        m.add(layers.BatchNormalization()); m.add(layers.Dropout(params["conv_dropout"], seed=int(seed)))
    m.add(layers.Flatten())
    m.add(layers.Dense(params["dense_units"], activation="relu", kernel_initializer=init))
    m.add(layers.BatchNormalization()); m.add(layers.Dropout(params["dense_dropout"], seed=int(seed)))
    m.add(layers.Dense(1, activation="sigmoid", kernel_initializer=init))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="binary_crossentropy")
    return m


def cnn_space(t):
    return {"initial_filters": t.suggest_categorical("initial_filters", [16, 32, 64]),
            "kernel_size": t.suggest_categorical("kernel_size", [3, 4, 5]),
            "conv_dropout": t.suggest_categorical("conv_dropout", [0.1, 0.2, 0.3]),
            "dense_units": t.suggest_categorical("dense_units", [32, 64, 96, 128]),
            "dense_dropout": t.suggest_categorical("dense_dropout", [0.1, 0.2, 0.3]),
            "learning_rate": t.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3]),
            "second_conv": t.suggest_categorical("second_conv", [True, False]),
            "batch_size": t.suggest_categorical("batch_size", [8, 16, 32])}


def cnn_spec():
    reshape = lambda X: X[..., None]
    builder = lambda params, seed: create_cnn((N_INPUT, 1), params, seed)
    return ModelSpec("CNN", cnn_space, _nn_fit_inner(builder, reshape), _nn_fit_final(builder, reshape),
                     _nn_predict(reshape), tf_cleanup, isolate=True)


# ── MLP ───────────────────────────────────────────────────────
def create_mlp(input_dim, params, seed):
    tf = tf_init(); layers = tf.keras.layers
    set_all_seeds(seed)
    init = tf.keras.initializers.GlorotUniform(seed=int(seed))
    reg = lambda: tf.keras.regularizers.L1L2(l1=params["l1"], l2=params["l2"])
    m = tf.keras.Sequential([layers.Input(shape=(input_dim,))])
    for units, drop in [(params["hidden1"], params["dropout1"]), (params["hidden2"], params["dropout2"])] + \
                       ([(params["hidden3"], params["dropout3"])] if params["use_third_layer"] else []):
        m.add(layers.Dense(units, activation="relu", kernel_regularizer=reg(), kernel_initializer=init))
        m.add(layers.BatchNormalization()); m.add(layers.Dropout(drop, seed=int(seed)))
    m.add(layers.Dense(1, activation="sigmoid", kernel_initializer=init))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="binary_crossentropy")
    return m


def mlp_space(t):
    return {"hidden1": t.suggest_categorical("hidden1", [128, 256, 512]),
            "hidden2": t.suggest_categorical("hidden2", [64, 128, 256]),
            "hidden3": t.suggest_categorical("hidden3", [32, 64, 96, 128]),
            "dropout1": t.suggest_categorical("dropout1", [0.1, 0.2, 0.3]),
            "dropout2": t.suggest_categorical("dropout2", [0.1, 0.2, 0.3]),
            "dropout3": t.suggest_categorical("dropout3", [0.1, 0.2, 0.3]),
            "l1": t.suggest_float("l1", 1e-5, 1e-3, log=True),
            "l2": t.suggest_float("l2", 1e-5, 1e-3, log=True),
            "use_third_layer": t.suggest_categorical("use_third_layer", [True, False]),
            "learning_rate": t.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3]),
            "batch_size": t.suggest_categorical("batch_size", [8, 16, 32])}


def mlp_spec():
    builder = lambda params, seed: create_mlp(N_INPUT, params, seed)
    return ModelSpec("MLP", mlp_space, _nn_fit_inner(builder, _id), _nn_fit_final(builder, _id),
                     _nn_predict(_id), tf_cleanup, isolate=True)


# ── XGBoost ───────────────────────────────────────────────────
def xgb_space(t):
    return {"max_depth": t.suggest_int("max_depth", 3, 10),
            "min_child_weight": t.suggest_int("min_child_weight", 1, 7),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": t.suggest_int("n_estimators", 100, 1000, step=100),
            "gamma": t.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)}


_XGB_FIXED = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", verbosity=0)


def xgb_spec():
    import xgboost as xgb

    def xgb_fit_inner(params, Xa, ya, Xb, yb, seed):
        m = xgb.XGBClassifier(**params, **_XGB_FIXED, device=C.XGB_DEVICE, n_jobs=N_JOBS, random_state=int(seed),
                              early_stopping_rounds=ES["tree_rounds"])
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m.predict_proba(Xb)[:, 1], int(m.best_iteration) + 1

    def xgb_fit_final(params, n_iter, X, y, seed):
        p = dict(params); p["n_estimators"] = n_iter
        return xgb.XGBClassifier(**p, **_XGB_FIXED, device=C.XGB_DEVICE, n_jobs=N_JOBS, random_state=int(seed)).fit(X, y)

    return ModelSpec("XGBoost", xgb_space, xgb_fit_inner, xgb_fit_final, lambda m, X: m.predict_proba(X)[:, 1])


# ── LightGBM ──────────────────────────────────────────────────
def lgb_space(t):
    return {"max_depth": t.suggest_int("max_depth", 3, 10),
            "num_leaves": t.suggest_int("num_leaves", 20, 100),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 50),
            "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": t.suggest_int("n_estimators", 100, 1000, step=100),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0)}


_LGB_FIXED = dict(objective="binary", subsample_freq=1, verbosity=-1, force_col_wise=True, boosting_type="gbdt")


def lgb_spec():
    import lightgbm as lgb

    def lgb_fit_inner(params, Xa, ya, Xb, yb, seed):
        m = lgb.LGBMClassifier(**params, **_LGB_FIXED, random_state=int(seed), n_jobs=N_JOBS)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgb.early_stopping(ES["tree_rounds"], verbose=False)])
        return m.predict_proba(Xb)[:, 1], int(m.best_iteration_)

    def lgb_fit_final(params, n_iter, X, y, seed):
        p = dict(params); p["n_estimators"] = n_iter
        return lgb.LGBMClassifier(**p, **_LGB_FIXED, random_state=int(seed), n_jobs=N_JOBS).fit(X, y)

    return ModelSpec("LightGBM", lgb_space, lgb_fit_inner, lgb_fit_final, lambda m, X: m.predict_proba(X)[:, 1])


# ── scikit-learn models: logistic regression, random forest, SVM ──
def _sk_spec(name, space, build):
    def fit_inner(params, Xa, ya, Xb, yb, seed):
        return build(params, seed).fit(Xa, ya).predict_proba(Xb)[:, 1], None
    def fit_final(params, n_iter, X, y, seed):
        return build(params, seed).fit(X, y)
    return ModelSpec(name, space, fit_inner, fit_final, lambda m, X: m.predict_proba(X)[:, 1])


def logreg_spec():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return _sk_spec(
        "LogReg",
        lambda t: {"C": t.suggest_float("C", 1e-4, 1e2, log=True)},
        lambda p, s: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=5000, random_state=int(s))))


def rf_spec():
    from sklearn.ensemble import RandomForestClassifier
    return _sk_spec(
        "RandomForest",
        lambda t: {"n_estimators": t.suggest_int("n_estimators", 100, 1000, step=100),
                   "max_depth": t.suggest_categorical("max_depth", [None, 5, 10, 20, 30]),
                   "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.3]),
                   "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10)},
        lambda p, s: RandomForestClassifier(**p, n_jobs=N_JOBS, random_state=int(s)))


def svm_spec():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return _sk_spec(
        "SVM",
        lambda t: {"C": t.suggest_float("C", 1e-2, 1e2, log=True),
                   "gamma": t.suggest_float("gamma", 1e-5, 1e-1, log=True)},
        lambda p, s: make_pipeline(StandardScaler(), SVC(C=p["C"], gamma=p["gamma"], kernel="rbf",
                                                         probability=True, random_state=int(s))))


SPEC_BUILDERS = {"CNN": cnn_spec, "MLP": mlp_spec, "XGBoost": xgb_spec, "LightGBM": lgb_spec,
                 "LogReg": logreg_spec, "RandomForest": rf_spec, "SVM": svm_spec}


def get_spec(name):
    return SPEC_BUILDERS[name]()
