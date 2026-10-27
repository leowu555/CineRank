from django.test import SimpleTestCase

from recommendations.recsys.item_item import ItemItemModel, rank_item_item
from recommendations.recsys.popularity import rank_popularity
from recommendations.recsys.pipeline import recommend
from recommendations.recsys.types import RatingEvent
from recommendations.recsys.user_cf import rank_user_user


def ev(user, item, score):
    return RatingEvent(user_id=user, item_id=item, score=score, title=f'I{item}')


class PopularityTests(SimpleTestCase):
    def test_ordering_and_exclusions(self):
        ratings = [
            ev(1, 10, 5),
            ev(2, 10, 4),
            ev(1, 11, 5),
            ev(2, 11, 5),
            ev(1, 12, 5),
            ev(1, 1, 5),
        ]
        ranked = rank_popularity(ratings, exclude_item_ids={1}, limit=10)
        self.assertEqual([row.item_id for row in ranked], [11, 10, 12])


class UserUserTests(SimpleTestCase):
    def test_excludes_rated_and_ranks_by_weight_then_id(self):
        ratings = [
            ev(1, 1, 5),
            ev(1, 2, 5),
            ev(2, 1, 5),
            ev(2, 3, 5),
            ev(2, 4, 4),
        ]
        ranked = rank_user_user(ratings, user_id=1)
        ids = [row.item_id for row in ranked]
        self.assertEqual(ids[0], 3)
        self.assertNotIn(1, ids)
        self.assertNotIn(2, ids)


class ItemItemTests(SimpleTestCase):
    def test_positive_history_aggregates_neighbors_and_skips_rated(self):
        ratings = [
            ev(1, 10, 5),
            ev(1, 20, 5),
            ev(2, 10, 5),
            ev(2, 20, 4),
            ev(2, 30, 5),
            ev(3, 10, 5),
        ]
        ranked = rank_item_item(ratings, user_id=3, neighbor_k=5)
        ids = [row.item_id for row in ranked]
        self.assertEqual(ids[0], 20)
        self.assertNotIn(10, ids)

    def test_fit_then_score_without_refit(self):
        train = [ev(1, 10, 5), ev(1, 20, 5), ev(2, 10, 5), ev(2, 20, 5)]
        model = ItemItemModel(neighbor_k=5).fit(train)
        self.assertTrue(model.neighbors[10])
        query = train + [ev(9, 10, 5)]
        ranked = model.recommend(query, user_id=9, exclude_item_ids={10})
        self.assertEqual([row.item_id for row in ranked], [20])

    def test_scores_below_threshold_do_not_seed_candidates(self):
        ratings = [
            ev(1, 10, 5),
            ev(1, 20, 5),
            ev(2, 10, 2),
        ]
        ranked = rank_item_item(ratings, user_id=2)
        self.assertEqual(ranked, [])


class PipelineTests(SimpleTestCase):
    def test_empty_when_no_unseen(self):
        result = recommend([ev(1, 1, 5)], user_id=1)
        self.assertEqual(result.method, 'empty')
        self.assertEqual(result.items, ())

    def test_cf_then_popularity(self):
        cf = recommend(
            [ev(1, 1, 5), ev(2, 1, 5), ev(2, 9, 5)],
            user_id=1,
        )
        self.assertEqual(cf.method, 'item_item')
        self.assertEqual([item.item_id for item in cf.items], [9])

        pop = recommend([ev(1, 1, 5), ev(2, 8, 5)], user_id=1)
        self.assertEqual(pop.method, 'fallback_popular')
        self.assertEqual([item.item_id for item in pop.items], [8])
