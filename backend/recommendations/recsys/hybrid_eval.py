"""Offline TF-IDF + RRF helpers. Not used by the Django recommendations API.

Stage 4 is deferred for the interview demo. Enable only via:
`python -m recommendations.recsys.cli --include-hybrid`
"""

from __future__ import annotations

from recommendations.recsys.content import CONTENT_CANDIDATE_LIMIT, ContentModel
from recommendations.recsys.evaluate import StrategyResult, _evaluate_rankings
from recommendations.recsys.hybrid import CANDIDATE_LIMIT, fuse_recommendations
from recommendations.recsys.item_item import ItemItemModel
from recommendations.recsys.splits import history_and_relevant
from recommendations.recsys.types import RatingEvent

K = 10
RELEVANT_THRESHOLD = 4.0


def _catalog_history(train, eval_split):
    catalog = {event.item_id for event in train}
    history, relevant, stats = history_and_relevant(
        train, eval_split, catalog, relevant_threshold=RELEVANT_THRESHOLD
    )
    return catalog, history, relevant, stats


def evaluate_content(
    train: list[RatingEvent],
    eval_split: list[RatingEvent],
    movies: dict[int, dict[str, str]],
    content_model: ContentModel | None = None,
) -> StrategyResult:
    catalog, history, relevant, stats = _catalog_history(train, eval_split)
    model = content_model or ContentModel().fit(movies, catalog=catalog)
    rankings: dict[int, list[int]] = {}
    for user_id in relevant:
        exclude = {event.item_id for event in history[user_id]}
        recs = model.recommend(history[user_id], user_id, exclude, limit=K)
        rankings[user_id] = [row.item_id for row in recs]
    return _evaluate_rankings('content', rankings, relevant, catalog, stats)


def evaluate_hybrid(
    train: list[RatingEvent],
    eval_split: list[RatingEvent],
    movies: dict[int, dict[str, str]],
    neighbor_k: int,
    item_model: ItemItemModel | None = None,
    content_model: ContentModel | None = None,
) -> StrategyResult:
    catalog, history, relevant, stats = _catalog_history(train, eval_split)
    item_model = item_model or ItemItemModel(neighbor_k=neighbor_k).fit(train)
    content_model = content_model or ContentModel().fit(movies, catalog=catalog)
    rankings: dict[int, list[int]] = {}
    for user_id in relevant:
        exclude = {event.item_id for event in history[user_id]}
        cf = item_model.recommend(
            history[user_id], user_id, exclude_item_ids=exclude, limit=CANDIDATE_LIMIT
        )
        content = content_model.recommend(
            history[user_id], user_id, exclude, limit=CONTENT_CANDIDATE_LIMIT
        )
        fused = fuse_recommendations(cf, content, limit=K)
        rankings[user_id] = [row.item_id for row in fused]
    return _evaluate_rankings(
        'hybrid_rrf',
        rankings,
        relevant,
        catalog,
        stats,
        extra={'rrf_k': 60, 'candidate_limit': CANDIDATE_LIMIT, 'neighbor_k': neighbor_k},
    )


def evaluate_content_and_hybrid(
    train: list[RatingEvent],
    validation: list[RatingEvent],
    test: list[RatingEvent],
    movies: dict[int, dict[str, str]],
    neighbor_k: int,
) -> dict:
    catalog = {event.item_id for event in train}
    content_model = ContentModel().fit(movies, catalog=catalog)
    item_model = ItemItemModel(neighbor_k=neighbor_k).fit(train)
    return {
        'validation': {
            'content': _as_dict(evaluate_content(train, validation, movies, content_model)),
            'hybrid_rrf': _as_dict(
                evaluate_hybrid(train, validation, movies, neighbor_k, item_model, content_model)
            ),
        },
        'test': {
            'content': _as_dict(evaluate_content(train, test, movies, content_model)),
            'hybrid_rrf': _as_dict(
                evaluate_hybrid(train, test, movies, neighbor_k, item_model, content_model)
            ),
        },
        'offline_only': True,
        'note': (
            'Content/hybrid use MovieLens movieId + title/genres. They are not served by the '
            'Django app. links.csv maps movieId to tmdbId; unmapped rows are dropped.'
        ),
    }


def _as_dict(result: StrategyResult) -> dict:
    return {
        'name': result.name,
        'recall_at_10': result.recall,
        'ndcg_at_10': result.ndcg,
        'catalog_coverage_at_10': result.coverage,
        'eligible_users': result.eligible_users,
        'stats': result.stats,
        'extra': result.extra,
    }
