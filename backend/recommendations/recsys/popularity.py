"""Catalog popularity from observed ratings only.

Ordering (deterministic):
1. rating count descending
2. mean score descending
3. item_id ascending
"""

from collections import defaultdict

from recommendations.recsys.meta import item_meta
from recommendations.recsys.types import DEFAULT_LIMIT, RatingEvent, Recommendation


def popularity_stats(ratings: list[RatingEvent]) -> dict[int, tuple[int, float]]:
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for event in ratings:
        sums[event.item_id] += event.score
        counts[event.item_id] += 1
    return {item_id: (counts[item_id], sums[item_id] / counts[item_id]) for item_id in counts}


def rank_popularity(
    ratings: list[RatingEvent],
    exclude_item_ids: set[int],
    limit: int = DEFAULT_LIMIT,
) -> list[Recommendation]:
    stats = popularity_stats(ratings)
    meta = item_meta(ratings)
    eligible = [
        (item_id, count, mean)
        for item_id, (count, mean) in stats.items()
        if item_id not in exclude_item_ids
    ]
    eligible.sort(key=lambda row: (-row[1], -row[2], row[0]))
    results: list[Recommendation] = []
    for item_id, count, mean in eligible[:limit]:
        title, poster = meta.get(item_id, ('', None))
        results.append(
            Recommendation(
                item_id=item_id,
                score=float(count),
                reason=f'Popular among raters ({count} ratings, mean {mean:.1f})',
                title=title,
                poster_url=poster,
            )
        )
    return results
