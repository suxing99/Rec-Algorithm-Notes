"""训练与评估指标。"""

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算二分类 AUC，标签只有单一类别时返回 nan。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))
