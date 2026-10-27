"""User–user cosine CF used only as an intermediate serving path.

Unobserved cells are stored as 0 in a dense user–item matrix (limitation).
Ranking score is similarity × neighbor rating, not a calibrated predicted rating.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recommendations.recsys.meta import item_meta, rated_item_ids
from recommendations.recsys.types import (
    DEFAULT_LIMIT,
    POSITIVE_RATING_THRESHOLD,
    RatingEvent,
    Recommendation,
)

NEIGHBOR_K = 5


def rank_user_user(
    ratings: list[RatingEvent],
    user_id: int,
    exclude_item_ids: set[int] | None = None,
    limit: int = DEFAULT_LIMIT,
    neighbor_k: int = NEIGHBOR_K,
    positive_threshold: int = POSITIVE_RATING_THRESHOLD,
) -> list[Recommendation]:
    if not ratings:
        return []

    exclude = exclude_item_ids if exclude_item_ids is not None else rated_item_ids(ratings, user_id)
    user_ids = sorted({event.user_id for event in ratings})
    item_ids = sorted({event.item_id for event in ratings})
    if user_id not in user_ids or len(user_ids) < 2:
        return []

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}
    matrix = np.zeros((len(user_ids), len(item_ids)))
    for event in ratings:
        matrix[user_index[event.user_id], item_index[event.item_id]] = event.score

    current_idx = user_index[user_id]
    similarities = cosine_similarity(matrix)
    neighbor_sims = similarities[current_idx].copy()
    neighbor_sims[current_idx] = -1.0
    top = np.argsort(neighbor_sims)[::-1][:neighbor_k]

    weights: dict[int, float] = {}
    for other_idx in top:
        sim = float(neighbor_sims[other_idx])
        if sim <= 0:
            continue
        for col, score in enumerate(matrix[other_idx]):
            if score < positive_threshold:
                continue
            item_id = item_ids[col]
            if item_id in exclude:
                continue
            weighted = sim * float(score)
            if weighted > weights.get(item_id, float('-inf')):
                weights[item_id] = weighted

    ranked = sorted(weights.items(), key=lambda row: (-row[1], row[0]))[:limit]
    meta = item_meta(ratings)
    results: list[Recommendation] = []
    for item_id, weight in ranked:
        title, poster = meta.get(item_id, ('', None))
        results.append(
            Recommendation(
                item_id=item_id,
                score=float(weight),
                reason='Users with similar taste also liked this',
                title=title,
                poster_url=poster,
            )
        )
    return results
