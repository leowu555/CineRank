from __future__ import annotations

from dataclasses import dataclass

from recommendations.recsys.types import RatingEvent


@dataclass(frozen=True)
class ChronoSplit:
    train: list[RatingEvent]
    validation: list[RatingEvent]
    test: list[RatingEvent]
    train_end_ts: float
    validation_end_ts: float
    test_end_ts: float


def chronological_split(
    ratings: list[RatingEvent],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> ChronoSplit:
    """Global time cutoffs on the rating stream (not per-user leave-one-out).

    Sort key: timestamp, user_id, item_id. Fractions apply to rating counts.
    """
    if not ratings:
        raise ValueError('Cannot split an empty rating list.')
    ordered = sorted(
        ratings,
        key=lambda event: (event.timestamp or 0, event.user_id, event.item_id),
    )
    n_train = int(len(ordered) * train_fraction)
    n_val = int(len(ordered) * validation_fraction)
    if n_train == 0 or n_train + n_val >= len(ordered):
        raise ValueError('Split fractions leave an empty train or test set.')
    train = ordered[:n_train]
    validation = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]
    return ChronoSplit(
        train=train,
        validation=validation,
        test=test,
        train_end_ts=float(train[-1].timestamp or 0),
        validation_end_ts=float(validation[-1].timestamp or 0),
        test_end_ts=float(test[-1].timestamp or 0),
    )


def history_and_relevant(
    train: list[RatingEvent],
    eval_split: list[RatingEvent],
    catalog: set[int],
    relevant_threshold: float = 4.0,
) -> tuple[dict[int, list[RatingEvent]], dict[int, set[int]], dict[str, int]]:
    """Build per-user train history and eval-split relevant items in the train catalog.

    Users with no train history or no in-catalog relevant items are excluded.
    """
    history: dict[int, list[RatingEvent]] = {}
    for event in train:
        history.setdefault(event.user_id, []).append(event)

    relevant: dict[int, set[int]] = {}
    skipped_users_no_history = 0
    skipped_users_no_positive_eval_ratings = 0
    skipped_users_all_relevant_outside_catalog = 0
    dropped_relevant_not_in_catalog = 0
    eval_users = {event.user_id for event in eval_split}

    relevant_raw: dict[int, set[int]] = {}
    for event in eval_split:
        if event.score >= relevant_threshold:
            relevant_raw.setdefault(event.user_id, set()).add(event.item_id)

    kept_relevant: dict[int, set[int]] = {}
    for user_id in eval_users:
        if user_id not in history:
            skipped_users_no_history += 1
            continue
        raw = relevant_raw.get(user_id, set())
        if not raw:
            skipped_users_no_positive_eval_ratings += 1
            continue
        in_catalog = {item_id for item_id in raw if item_id in catalog}
        dropped_relevant_not_in_catalog += len(raw) - len(in_catalog)
        if not in_catalog:
            skipped_users_all_relevant_outside_catalog += 1
            continue
        kept_relevant[user_id] = in_catalog

    skipped_users_no_relevant = (
        skipped_users_no_positive_eval_ratings + skipped_users_all_relevant_outside_catalog
    )
    stats = {
        'eval_users': len(eval_users),
        'eligible_users': len(kept_relevant),
        'skipped_users_no_history': skipped_users_no_history,
        'skipped_users_no_positive_eval_ratings': skipped_users_no_positive_eval_ratings,
        'skipped_users_all_relevant_outside_catalog': skipped_users_all_relevant_outside_catalog,
        'skipped_users_no_relevant': skipped_users_no_relevant,
        'dropped_relevant_not_in_catalog': dropped_relevant_not_in_catalog,
        'relevant_items': sum(len(items) for items in kept_relevant.values()),
        'accounting': (
            f'{len(eval_users)} eval users = {skipped_users_no_history} no train history + '
            f'{skipped_users_no_positive_eval_ratings} no eval rating>={relevant_threshold:g} + '
            f'{skipped_users_all_relevant_outside_catalog} relevant only outside train catalog + '
            f'{len(kept_relevant)} eligible'
        ),
    }
    return history, kept_relevant, stats
