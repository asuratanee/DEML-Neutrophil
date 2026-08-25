# =============================================================
# src/trainers/models.py
# Model creation for all four classifiers.
# =============================================================

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import lightgbm as lgb
from config import RANDOM_SEED


def create_model(classifier, params, **kwargs):
    """Factory function: create a model instance for the given classifier."""
    if classifier == 'cnn':
        return _create_cnn(params, kwargs.get('input_shape'), kwargs.get('seed', RANDOM_SEED))
    elif classifier == 'mlp':
        return _create_mlp(params, kwargs.get('input_dim'), kwargs.get('seed', RANDOM_SEED))
    elif classifier == 'xgboost':
        return xgb.XGBClassifier(**{k: v for k, v in params.items()
                                     if k != 'early_stopping_rounds'},
                                  early_stopping_rounds=params.get('early_stopping_rounds', 20))
    elif classifier == 'lightgbm':
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


def _create_cnn(params, input_shape, seed):
    tf.keras.utils.set_random_seed(seed)
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)
    n_filters = params['initial_filters']

    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(filters=n_filters, kernel_size=params['kernel_size'],
                             activation='relu', strides=2,
                             kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(params['conv_dropout'], seed=seed))

    if params.get('second_conv', False):
        model.add(layers.Conv1D(filters=n_filters * 2,
                                 kernel_size=params['kernel_size'],
                                 activation='relu', strides=2,
                                 kernel_initializer=kernel_init))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params['conv_dropout'], seed=seed))

    model.add(layers.Flatten())
    model.add(layers.Dense(params['dense_units'], activation='relu',
                            kernel_initializer=kernel_init))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(params['dense_dropout'], seed=seed))
    model.add(layers.Dense(1, activation='sigmoid',
                            kernel_initializer=kernel_init))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def _create_mlp(params, input_dim, seed):
    tf.keras.utils.set_random_seed(seed)
    kernel_init = tf.keras.initializers.GlorotUniform(seed=seed)
    reg = tf.keras.regularizers.L1L2(l1=params['l1'], l2=params['l2'])

    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for i, (units, dropout) in enumerate([
        (params['hidden1'], params['dropout1']),
        (params['hidden2'], params['dropout2']),
    ]):
        model.add(layers.Dense(units, activation='relu',
                                kernel_regularizer=reg,
                                kernel_initializer=kernel_init))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout, seed=seed + i))

    if params.get('use_third_layer', False):
        model.add(layers.Dense(params['hidden3'], activation='relu',
                                kernel_regularizer=reg,
                                kernel_initializer=kernel_init))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(params['dropout3'], seed=seed + 2))

    model.add(layers.Dense(1, activation='sigmoid',
                            kernel_initializer=kernel_init))
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model
