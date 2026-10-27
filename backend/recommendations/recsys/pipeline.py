from recommendations.recsys.item_item import rank_item_item
from recommendations.recsys.popularity import rank_popularity
from recommendations.recsys.types import DEFAULT_LIMIT, RatingEvent, RecommendResult
from recommendations.recsys.user_cf import rank_user_user


EMPTY_MESSAGE = 'No unseen movies are available to recommend.'


def recommend(
    ratings: list[RatingEvent],
    user_id: int,
    limit: int = DEFAULT_LIMIT,
    cf: str = 'item_item',
) -> RecommendResult:
    """Return unseen recommendations. Never echoes the user's own rated titles.

    cf: 'user_user' (stage 1) or 'item_item' (stage 2+).
    """
    rated = {event.item_id for event in ratings if event.user_id == user_id}
    catalog = {event.item_id for event in ratings}
    if not (catalog - rated):
        return RecommendResult(items=(), method='empty', message=EMPTY_MESSAGE)

    cf_items = []
    method = 'item_item'
    if cf == 'user_user':
        cf_items = rank_user_user(ratings, user_id, exclude_item_ids=rated, limit=limit)
        method = 'collaborative_filtering'
    else:
        cf_items = rank_item_item(ratings, user_id, exclude_item_ids=rated, limit=limit)

    if cf_items:
        return RecommendResult(items=tuple(cf_items), method=method)

    popular = rank_popularity(ratings, exclude_item_ids=rated, limit=limit)
    if popular:
        return RecommendResult(items=tuple(popular), method='fallback_popular')

    return RecommendResult(items=(), method='empty', message=EMPTY_MESSAGE)
