"""TF-IDF content ranking over MovieLens title + genres.

Document is: title, then genres with '|' replaced by space. Lowercased English
stop words via scikit-learn TfidfVectorizer. Items with empty text are skipped.
Missing metadata => item ineligible for content ranking.

User profile: mean TF-IDF vector of items the user rated >= positive_threshold
that have content. Rank remaining catalog items by cosine(profile, item).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from recommendations.recsys.types import DEFAULT_LIMIT, POSITIVE_RATING_THRESHOLD, RatingEvent, Recommendation

CONTENT_CANDIDATE_LIMIT = 50


def _document(title: str, genres: str) -> str:
    return f'{title} {genres.replace("|", " ")}'.strip()


class ContentModel:
    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.item_ids: list[int] = []
        self.matrix = None
        self.index: dict[int, int] = {}

    def fit(self, movies: dict[int, dict[str, str]], catalog: set[int] | None = None) -> ContentModel:
        docs: list[str] = []
        item_ids: list[int] = []
        for item_id, meta in sorted(movies.items()):
            if catalog is not None and item_id not in catalog:
                continue
            text = _document(meta.get('title', ''), meta.get('genres', ''))
            if not text:
                continue
            item_ids.append(item_id)
            docs.append(text)
        if not docs:
            self.item_ids = []
            self.matrix = None
            self.index = {}
            return self
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.matrix = self.vectorizer.fit_transform(docs)
        self.item_ids = item_ids
        self.index = {item_id: i for i, item_id in enumerate(item_ids)}
        return self

    def recommend(
        self,
        history: list[RatingEvent],
        user_id: int,
        exclude_item_ids: set[int],
        limit: int = DEFAULT_LIMIT,
        positive_threshold: float = POSITIVE_RATING_THRESHOLD,
    ) -> list[Recommendation]:
        if self.matrix is None or not self.item_ids:
            return []
        profile_rows = []
        for event in history:
            if event.user_id != user_id or event.score < positive_threshold:
                continue
            idx = self.index.get(event.item_id)
            if idx is None:
                continue
            profile_rows.append(idx)
        if not profile_rows:
            return []
        profile = np.asarray(self.matrix[profile_rows].mean(axis=0)).reshape(1, -1)
        sims = cosine_similarity(profile, self.matrix).ravel()
        order = np.argsort(-sims, kind='mergesort')
        results: list[Recommendation] = []
        for idx in order:
            item_id = self.item_ids[int(idx)]
            if item_id in exclude_item_ids:
                continue
            sim = float(sims[idx])
            if sim <= 0:
                continue
            results.append(
                Recommendation(
                    item_id=item_id,
                    score=sim,
                    reason='Similar title/genre text to movies you liked',
                )
            )
            if len(results) >= limit:
                break
        return results


def rank_content(
    movies: dict[int, dict[str, str]],
    history: list[RatingEvent],
    user_id: int,
    exclude_item_ids: set[int],
    catalog: set[int] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[Recommendation]:
    return (
        ContentModel()
        .fit(movies, catalog=catalog)
        .recommend(history, user_id, exclude_item_ids, limit=limit)
    )
