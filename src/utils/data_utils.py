# =============================================================
# src/utils/data_utils.py
# Data loading and class balancing utilities.
# =============================================================

import numpy as np
import pandas as pd
from sklearn.utils import resample


def load_data(features_path, cond):
    """Load Node2Vec embeddings and Up/Down labels for a condition."""
    embed = pd.read_csv(
        f"{features_path}/Embeddings512_UT_{cond}_WGCNA_concat.csv", index_col=0)
    label = pd.read_csv(
        f"{features_path}/labels_UpvDown_{cond}.csv", index_col=0)
    common = embed.index.intersection(label.index)
    X = embed.loc[common].values.astype(np.float32)
    y = label.loc[common, 'label'].values.astype(int)
    genes = np.array(common)
    print(f"  {cond}: {len(common):,} genes "
          f"(Up={y.sum()}, Down={(y==0).sum()})")
    return X, y, genes


def balance_dataset(X, y, seed):
    """
    Undersample majority class to match minority class size.
    Applied at repeat level prior to cross-validation splitting.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = min(len(pos_idx), len(neg_idx))
    pos_bal = resample(pos_idx, n_samples=n, random_state=seed, replace=False)
    neg_bal = resample(neg_idx, n_samples=n, random_state=seed, replace=False)
    rng = np.random.RandomState(seed)
    bal_idx = rng.permutation(np.concatenate([pos_bal, neg_bal]))
    return X[bal_idx], y[bal_idx]
