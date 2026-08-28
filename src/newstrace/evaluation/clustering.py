"""Clustering metrics: pairwise, B-cubed, ARI, fragmentation and merge errors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from sklearn.metrics import adjusted_rand_score


@dataclass
class ClusteringMetrics:
    pairwise_precision: float = 0.0
    pairwise_recall: float = 0.0
    pairwise_f1: float = 0.0
    bcubed_precision: float = 0.0
    bcubed_recall: float = 0.0
    bcubed_f1: float = 0.0
    adjusted_rand_index: float = 0.0
    fragmentation_rate: float = 0.0
    merge_error_rate: float = 0.0
    predicted_clusters: int = 0
    true_clusters: int = 0
    items: int = 0

    def as_dict(self) -> dict[str, float]:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def pairwise_scores(
    truth: Mapping[int, int], predicted: Mapping[int, int]
) -> tuple[float, float, float]:
    """Precision/recall/F1 over co-membership pairs."""
    items = sorted(set(truth) & set(predicted))
    tp = fp = fn = 0
    for a, b in combinations(items, 2):
        same_true = truth[a] == truth[b]
        same_pred = predicted[a] == predicted[b]
        if same_pred and same_true:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, _f1(precision, recall)


def bcubed_scores(
    truth: Mapping[int, int], predicted: Mapping[int, int]
) -> tuple[float, float, float]:
    """B-cubed precision/recall/F1 (per-item, cluster-size normalised)."""
    items = sorted(set(truth) & set(predicted))
    if not items:
        return 0.0, 0.0, 0.0
    by_pred: dict[int, list[int]] = defaultdict(list)
    by_true: dict[int, list[int]] = defaultdict(list)
    for item in items:
        by_pred[predicted[item]].append(item)
        by_true[truth[item]].append(item)

    precision_sum = 0.0
    recall_sum = 0.0
    for item in items:
        pred_cluster = by_pred[predicted[item]]
        true_cluster = by_true[truth[item]]
        correct = sum(1 for other in pred_cluster if truth[other] == truth[item])
        precision_sum += correct / len(pred_cluster)
        recall_sum += correct / len(true_cluster)
    precision = precision_sum / len(items)
    recall = recall_sum / len(items)
    return precision, recall, _f1(precision, recall)


def fragmentation_and_merge(
    truth: Mapping[int, int], predicted: Mapping[int, int]
) -> tuple[float, float]:
    """Fraction of true stories split apart, and of predicted stories that merge events."""
    items = sorted(set(truth) & set(predicted))
    if not items:
        return 0.0, 0.0
    true_groups: dict[int, set[int]] = defaultdict(set)
    pred_groups: dict[int, set[int]] = defaultdict(set)
    for item in items:
        true_groups[truth[item]].add(predicted[item])
        pred_groups[predicted[item]].add(truth[item])
    fragmented = sum(1 for labels in true_groups.values() if len(labels) > 1)
    merged = sum(1 for labels in pred_groups.values() if len(labels) > 1)
    return fragmented / len(true_groups), merged / len(pred_groups)


def evaluate_clustering(
    truth: Mapping[int, int], predicted: Mapping[int, int]
) -> ClusteringMetrics:
    items = sorted(set(truth) & set(predicted))
    p, r, f = pairwise_scores(truth, predicted)
    bp, br, bf = bcubed_scores(truth, predicted)
    frag, merge = fragmentation_and_merge(truth, predicted)
    ari = (
        float(adjusted_rand_score([truth[i] for i in items], [predicted[i] for i in items]))
        if len(items) > 1
        else 0.0
    )
    return ClusteringMetrics(
        pairwise_precision=p,
        pairwise_recall=r,
        pairwise_f1=f,
        bcubed_precision=bp,
        bcubed_recall=br,
        bcubed_f1=bf,
        adjusted_rand_index=ari,
        fragmentation_rate=frag,
        merge_error_rate=merge,
        predicted_clusters=len({predicted[i] for i in items}),
        true_clusters=len({truth[i] for i in items}),
        items=len(items),
    )


def labels_from_pairs(pairs: Sequence[tuple[int, int, int]]) -> dict[int, int]:
    """Union-find over labelled ``(article_a, article_b, same_story)`` triples."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, same in pairs:
        find(a)
        find(b)
        if same:
            union(a, b)
    return {item: find(item) for item in parent}
