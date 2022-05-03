import numpy as np
from sklearn.metrics import roc_auc_score


def AUC(y_true, y_score):
    if np.isnan(y_score).all():
        return np.nan

    return roc_auc_score(y_true=y_true, y_score=y_score)
