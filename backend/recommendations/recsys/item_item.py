"""Sparse item–item cosine CF.

Fit builds a CSR item–user matrix from observed ratings only. For each item it
keeps the top-k cosine neighbors (one dense similarity row at a time — never a
full item×item matrix).

Scoring for user u:
    score(j) = sum_{i in I_u^+}  sim(i, j) * r(u, i)
where I_u^+ is items u rated >= POSITIVE_RATING_THRESHOLD (default 4),
sim(i, j) is cosine of item rating vectors, unobserved pairs omitted from CSR
(they contribute 0 to the dot product without filling a dense catalog).

Neighbor list is a fit artifact; recommend() only aggregates. The Django view
currently re-fits per request (no persisted artifacts).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from recommendations.recsys.config import APP_NEIGHBOR_K
from recommendations.recsys.meta import item_meta, rated_item_ids
from recommendations.recsys.types import (
    DEFAULT_LIMIT,
    POSITIVE_RATING_THRESHOLD,
    RatingEvent,
    Recommendation,
)

DEFAULT_NEIGHBOR_K = APP_NEIGHBOR_K


class ItemItemModel:
    def __init__(self, neighbor_k: int = DEFAULT_NEIGHBOR_K):
        self.neighbor_k = neighbor_k
        self.neighbors: dict[int, list[tuple[int, float]]] = {}
        self.item_ids: list[int] = []

    def fit(self, ratings: list[RatingEvent]) -> ItemItemModel:
        self.neighbors = {}
        if not ratings:
            return self

        user_ids = sorted({event.user_id for event in ratings})
        item_ids = sorted({event.item_id for event in ratings})
        self.item_ids = item_ids
        user_index = {uid: i for i, uid in enumerate(user_ids)}
        item_index = {iid: i for i, iid in enumerate(item_ids)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for event in ratings:
            rows.append(item_index[event.item_id])
            cols.append(user_index[event.user_id])
            data.append(float(event.score))

        item_user = csr_matrix(
            (data, (rows, cols)),
            shape=(len(item_ids), len(user_ids)),
            dtype=np.float64,
        )

        k = min(self.neighbor_k, max(len(item_ids) - 1, 0))
        if k == 0:
            self.neighbors = {item_id: [] for item_id in item_ids}
            return self

        for i, item_id in enumerate(item_ids):
            sims = cosine_similarity(item_user[i], item_user, dense_output=True).ravel()
            sims[i] = -1.0
            candidate_idx = np.argpartition(sims, -k)[-k:]
            order = candidate_idx[np.argsort(sims[candidate_idx])[::-1]]
            stored: list[tuple[int, float]] = []
            for j in order:
                sim = float(sims[j])
                if sim <= 0:
                    continue
                stored.append((item_ids[int(j)], sim))
            self.neighbors[item_id] = stored
        return self

    def recommend(
        self,
        ratings: list[RatingEvent],
        user_id: int,
        exclude_item_ids: set[int] | None = None,
        limit: int = DEFAULT_LIMIT,
        positive_threshold: int = POSITIVE_RATING_THRESHOLD,
    ) -> list[Recommendation]:
        exclude = (
            exclude_item_ids
            if exclude_item_ids is not None
            else rated_item_ids(ratings, user_id)
        )
        weights: dict[int, float] = defaultdict(float)
        for event in ratings:
            if event.user_id != user_id or event.score < positive_threshold:
                continue
            for neighbor_id, sim in self.neighbors.get(event.item_id, ()):
                if neighbor_id in exclude:
                    continue
                weights[neighbor_id] += sim * float(event.score)

        ranked = sorted(weights.items(), key=lambda row: (-row[1], row[0]))[:limit]
        meta = item_meta(ratings)
        results: list[Recommendation] = []
        for item_id, weight in ranked:
            title, poster = meta.get(item_id, ('', None))
            results.append(
                Recommendation(
                    item_id=item_id,
                    score=float(weight),
                    reason='Because you liked similar titles',
                    title=title,
                    poster_url=poster,
                )
            )
        return results


def rank_item_item(
    ratings: list[RatingEvent],
    user_id: int,
    exclude_item_ids: set[int] | None = None,
    limit: int = DEFAULT_LIMIT,
    neighbor_k: int = DEFAULT_NEIGHBOR_K,
    positive_threshold: int = POSITIVE_RATING_THRESHOLD,
) -> list[Recommendation]:
    model = ItemItemModel(neighbor_k=neighbor_k)
    model.fit(ratings)
    return model.recommend(
        ratings,
        user_id,
        exclude_item_ids=exclude_item_ids,
        limit=limit,
        positive_threshold=positive_threshold,
    )
