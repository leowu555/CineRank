from recommendations.recsys.types import RatingEvent


def item_meta(ratings: list[RatingEvent]) -> dict[int, tuple[str, str | None]]:
    """Last observed title/poster per item (stable enough for display)."""
    meta: dict[int, tuple[str, str | None]] = {}
    for event in ratings:
        meta[event.item_id] = (event.title, event.poster_url)
    return meta


def rated_item_ids(ratings: list[RatingEvent], user_id: int) -> set[int]:
    return {event.item_id for event in ratings if event.user_id == user_id}


def catalog_item_ids(ratings: list[RatingEvent]) -> set[int]:
    return {event.item_id for event in ratings}
