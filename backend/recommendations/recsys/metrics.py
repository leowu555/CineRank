from __future__ import annotations

import math


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float | None:
    if not relevant:
        return None
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float | None:
    if not relevant:
        return None
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg


def catalog_coverage(recommended_lists: list[list[int]], catalog: set[int], k: int = 10) -> float:
    """Share of the train catalog that appears in at least one eligible user's top-k.

    Coverage@10 = |union of recommended[:10] over eligible users| / |train catalog|.
    Rankings shorter than 10 (item–item with few neighbors) still contribute
    only the items they actually returned. Not a per-user metric.
    """
    if not catalog:
        return 0.0
    produced = {item_id for ranking in recommended_lists for item_id in ranking[:k]}
    return len(produced) / len(catalog)


def mean_skip_none(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
