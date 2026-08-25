# =============================================================
# src/utils/eval_utils.py
# Evaluation metrics for cross-validation folds.
# =============================================================

import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              average_precision_score, roc_curve)


def evaluate_fold(y_val, y_pred):
    """Compute AUC, AUPR, and accuracy for a validation fold."""
    auc  = roc_auc_score(y_val, y_pred)
    aupr = average_precision_score(y_val, y_pred)
    fpr, tpr, thresh = roc_curve(y_val, y_pred)
    opt_thresh = thresh[np.argmax(tpr - fpr)]
    acc = accuracy_score(y_val, (y_pred >= opt_thresh).astype(int))
    return {'auc': auc, 'aupr': aupr, 'acc': acc}
