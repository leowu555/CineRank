import math

from django.test import SimpleTestCase

from recommendations.recsys.content import ContentModel
from recommendations.recsys.hybrid import reciprocal_rank_fusion
from recommendations.recsys.id_map import mapped_tmdb_ids
from recommendations.recsys.metrics import catalog_coverage, dcg_at_k, ndcg_at_k, recall_at_k
from recommendations.recsys.splits import chronological_split, history_and_relevant
from recommendations.recsys.types import RatingEvent


def ev(user, item, score, ts):
    return RatingEvent(user_id=user, item_id=item, score=score, timestamp=ts)


class MetricTests(SimpleTestCase):
    def test_recall_and_ndcg_hand_checked(self):
        recommended = [1, 2, 3]
        relevant = {2, 4}
        self.assertEqual(recall_at_k(recommended, relevant, k=2), 0.5)
        dcg = 0 + 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(dcg_at_k(recommended, relevant, 2), dcg)
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, 2), dcg / idcg)

    def test_empty_relevant_is_undefined(self):
        self.assertIsNone(recall_at_k([1], set(), 10))
        self.assertIsNone(ndcg_at_k([1], set(), 10))

    def test_coverage(self):
        self.assertEqual(catalog_coverage([[1, 2], [2, 3]], {1, 2, 3, 4}, k=10), 0.75)


class SplitTests(SimpleTestCase):
    def test_global_time_order_and_tie_break(self):
        ratings = [
            ev(2, 1, 5, 10),
            ev(1, 2, 4, 10),
            ev(1, 1, 5, 5),
            ev(3, 9, 5, 20),
            ev(3, 8, 5, 30),
        ]
        split = chronological_split(ratings, train_fraction=0.4, validation_fraction=0.2)
        self.assertEqual([e.item_id for e in split.train], [1, 2])
        self.assertEqual(len(split.validation), 1)
        self.assertEqual(len(split.test), 2)

    def test_excludes_users_without_history_or_in_catalog_relevance(self):
        train = [ev(1, 10, 5, 1), ev(1, 11, 4, 2)]
        eval_split = [
            ev(1, 99, 5, 3),
            ev(1, 11, 5, 4),
            ev(2, 10, 5, 5),
        ]
        catalog = {10, 11}
        _history, relevant, stats = history_and_relevant(train, eval_split, catalog)
        self.assertEqual(relevant, {1: {11}})
        self.assertEqual(stats['skipped_users_no_history'], 1)
        self.assertEqual(stats['skipped_users_no_positive_eval_ratings'], 0)
        self.assertEqual(stats['skipped_users_all_relevant_outside_catalog'], 0)
        self.assertEqual(stats['dropped_relevant_not_in_catalog'], 1)

    def test_splits_no_positive_eval_versus_outside_catalog(self):
        train = [ev(1, 10, 5, 1), ev(2, 10, 5, 1)]
        eval_split = [
            ev(1, 10, 3, 2),
            ev(2, 99, 5, 2),
        ]
        _history, relevant, stats = history_and_relevant(train, eval_split, catalog={10})
        self.assertEqual(relevant, {})
        self.assertEqual(stats['skipped_users_no_positive_eval_ratings'], 1)
        self.assertEqual(stats['skipped_users_all_relevant_outside_catalog'], 1)
        self.assertEqual(stats['skipped_users_no_relevant'], 2)


class ContentHybridTests(SimpleTestCase):
    def test_tfidf_ranks_same_genre_ahead_of_unrelated(self):
        movies = {
            1: {'title': 'Toy Story', 'genres': 'Adventure|Animation'},
            2: {'title': 'A Bug\'s Life', 'genres': 'Adventure|Animation'},
            3: {'title': 'Saw', 'genres': 'Horror|Thriller'},
        }
        history = [ev(9, 1, 5, 1)]
        recs = ContentModel().fit(movies).recommend(history, 9, exclude_item_ids={1}, limit=2)
        self.assertEqual(recs[0].item_id, 2)

    def test_rrf_formula_and_ties(self):
        fused = reciprocal_rank_fusion([[1, 2], [2, 3]], k=60, limit=10)
        scores = dict(fused)
        self.assertAlmostEqual(scores[1], 1 / 61)
        self.assertAlmostEqual(scores[2], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores[3], 1 / 62)
        self.assertEqual([item for item, _ in fused], [2, 1, 3])
        tied = reciprocal_rank_fusion([[10], [11]], k=60, limit=2)
        self.assertEqual([item for item, _ in tied], [10, 11])

    def test_missing_tmdb_links_are_dropped(self):
        links = {1: 27205, 2: None, 3: 0}
        self.assertEqual(mapped_tmdb_ids(links, [1, 2, 3, 4]), [27205])
