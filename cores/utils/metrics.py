"""训练与评估指标。"""

from collections import defaultdict
from typing import Dict, Sequence, Union

import numpy as np
from sklearn.metrics import roc_auc_score


def group_ranking_metrics(
    group_ids: Sequence[Union[str, int]],
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    positive_threshold: float = 0.0,
    ks: Sequence[int] = (1, 3),
) -> Dict[str, float]:
    """
    按 group_id 分组计算 MRR 与 Hit@K。

    仅统计组内至少有一个正样本（label > positive_threshold）的推荐组；
    组内按 scores 降序排列，取第一个正样本的排名计算 RR 与 Hit@K。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    groups: dict = defaultdict(list)
    for gid, score, label in zip(group_ids, scores, labels):
        groups[gid].append((score, label))

    reciprocal_ranks: list[float] = []
    hits: dict[int, list[float]] = {k: [] for k in ks}

    for items in groups.values():
        if max(label for _, label in items) <= positive_threshold:
            continue

        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)
        first_pos_rank = next(
            rank
            for rank, (_, label) in enumerate(sorted_items, start=1)
            if label > positive_threshold
        )
        reciprocal_ranks.append(1.0 / first_pos_rank)
        for k in ks:
            hits[k].append(1.0 if first_pos_rank <= k else 0.0)

    num_groups = len(reciprocal_ranks)
    if num_groups == 0:
        result: Dict[str, float] = {"mrr": float("nan"), "num_groups": 0.0}
        for k in ks:
            result[f"hit@{k}"] = float("nan")
        return result

    result = {
        "mrr": float(np.mean(reciprocal_ranks)),
        "num_groups": float(num_groups),
    }
    for k in ks:
        result[f"hit@{k}"] = float(np.mean(hits[k]))
    return result


def binary_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算二分类 AUC，标签只有单一类别时返回 nan。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))
