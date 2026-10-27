"""Django-free recommendation algorithms.

Serving and the MovieLens CLI import this package. Similarity is computed from
sparse (or small dense) rating matrices; missing user–item pairs are unobserved,
not explicit zeros in a fully materialized catalog matrix.

Positive-rating threshold for neighbor-style candidate generation: score >= 4.
Popularity ranking: rating_count desc, mean_score desc, item_id asc.
"""

from recommendations.recsys.pipeline import recommend
from recommendations.recsys.types import Recommendation, RecommendResult

__all__ = ['recommend', 'Recommendation', 'RecommendResult']
