from dataclasses import dataclass


POSITIVE_RATING_THRESHOLD = 4
DEFAULT_LIMIT = 10


@dataclass(frozen=True)
class RatingEvent:
    user_id: int
    item_id: int
    score: float
    timestamp: float | None = None
    title: str = ''
    poster_url: str | None = None


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    score: float
    reason: str
    title: str = ''
    poster_url: str | None = None


@dataclass(frozen=True)
class RecommendResult:
    items: tuple[Recommendation, ...]
    method: str
    message: str = ''
