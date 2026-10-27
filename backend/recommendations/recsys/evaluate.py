from __future__ import annotations

import time
from dataclasses import dataclass

from recommendations.recsys.config import EVAL_LIST_K, EVAL_NEIGHBOR_K_GRID
from recommendations.recsys.item_item import ItemItemModel
from recommendations.recsys.metrics import catalog_coverage, mean_skip_none, ndcg_at_k, recall_at_k
from recommendations.recsys.popularity import rank_popularity
from recommendations.recsys.splits import history_and_relevant
from recommendations.recsys.types import RatingEvent

K = EVAL_LIST_K
RELEVANT_THRESHOLD = 4.0


@dataclass
class StrategyResult:
    name: str
    recall: float | None
    ndcg: float | None
    coverage: float
    eligible_users: int
    stats: dict[str, int]
    extra: dict


def _ids(recs) -> list[int]:
    return [row.item_id for row in recs]


def _evaluate_rankings(
    name: str,
    rankings: dict[int, list[int]],
    relevant: dict[int, set[int]],
    catalog: set[int],
    stats: dict[str, int],
    extra: dict | None = None,
) -> StrategyResult:
    recalls = []
    ndcgs = []
    lists = []
    for user_id, rel in relevant.items():
        ranking = rankings.get(user_id, [])
        recalls.append(recall_at_k(ranking, rel, K))
        ndcgs.append(ndcg_at_k(ranking, rel, K))
        lists.append(ranking)
    return StrategyResult(
        name=name,
        recall=mean_skip_none(recalls),
        ndcg=mean_skip_none(ndcgs),
        coverage=catalog_coverage(lists, catalog, K),
        eligible_users=len(relevant),
        stats=stats,
        extra=extra or {},
    )


def evaluate_popularity(
    train: list[RatingEvent],
    eval_split: list[RatingEvent],
) -> StrategyResult:
    catalog = {event.item_id for event in train}
    history, relevant, stats = history_and_relevant(
        train, eval_split, catalog, relevant_threshold=RELEVANT_THRESHOLD
    )
    rankings: dict[int, list[int]] = {}
    for user_id in relevant:
        exclude = {event.item_id for event in history[user_id]}
        rankings[user_id] = _ids(rank_popularity(train, exclude, limit=K))
    return _evaluate_rankings('popularity', rankings, relevant, catalog, stats)


def evaluate_item_item(
    train: list[RatingEvent],
    eval_split: list[RatingEvent],
    neighbor_k: int,
) -> StrategyResult:
    catalog = {event.item_id for event in train}
    history, relevant, stats = history_and_relevant(
        train, eval_split, catalog, relevant_threshold=RELEVANT_THRESHOLD
    )
    started = time.perf_counter()
    model = ItemItemModel(neighbor_k=neighbor_k).fit(train)
    fit_s = time.perf_counter() - started
    rankings: dict[int, list[int]] = {}
    for user_id in relevant:
        exclude = {event.item_id for event in history[user_id]}
        rankings[user_id] = _ids(
            model.recommend(history[user_id], user_id, exclude_item_ids=exclude, limit=K)
        )
    return _evaluate_rankings(
        'item_item',
        rankings,
        relevant,
        catalog,
        stats,
        extra={'neighbor_k': neighbor_k, 'fit_seconds': round(fit_s, 3)},
    )


def tune_neighbor_k(
    train: list[RatingEvent],
    validation: list[RatingEvent],
    candidates: tuple[int, ...] = EVAL_NEIGHBOR_K_GRID,
) -> tuple[int, list[StrategyResult]]:
    results = [evaluate_item_item(train, validation, neighbor_k=k) for k in candidates]
    best = max(results, key=lambda row: (row.ndcg is not None, row.ndcg or 0.0, -row.extra['neighbor_k']))
    return int(best.extra['neighbor_k']), results
