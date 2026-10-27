"""Reciprocal rank fusion of item–item CF and content lists.

RRF(d) = sum_s 1 / (k + rank_s(d))
k = 60 (Cormack, Clarke, Buettcher).
Each system contributes its top `candidate_limit` ranks (default 50), 1-based.
Items only in one list get that system's term only.
Tie-break: RRF score descending, then item_id ascending.
"""

from __future__ import annotations

from recommendations.recsys.types import DEFAULT_LIMIT, Recommendation

RRF_K = 60
CANDIDATE_LIMIT = 50


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = RRF_K,
    limit: int = DEFAULT_LIMIT,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:limit]


def fuse_recommendations(
    cf_items: list[Recommendation],
    content_items: list[Recommendation],
    limit: int = DEFAULT_LIMIT,
    rrf_k: int = RRF_K,
) -> list[Recommendation]:
    fused = reciprocal_rank_fusion(
        [[row.item_id for row in cf_items], [row.item_id for row in content_items]],
        k=rrf_k,
        limit=limit,
    )
    reasons = {}
    for row in cf_items:
        reasons.setdefault(row.item_id, []).append('item-item')
    for row in content_items:
        reasons.setdefault(row.item_id, []).append('content')
    return [
        Recommendation(
            item_id=item_id,
            score=score,
            reason='RRF of ' + ' + '.join(reasons.get(item_id, [])),
        )
        for item_id, score in fused
    ]
