from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from ratings.models import Rating


def _rate(user, movie_id, score, title=None):
    return Rating.objects.create(
        user=user,
        movie_id=movie_id,
        title=title or f'Movie {movie_id}',
        score=score,
    )


class RecommendationAPITests(APITestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='secret1')
        self.bob = User.objects.create_user(username='bob', password='secret1')
        self.carol = User.objects.create_user(username='carol', password='secret1')
        self.alice_token = Token.objects.create(user=self.alice)
        self.bob_token = Token.objects.create(user=self.bob)
        self.carol_token = Token.objects.create(user=self.carol)

    def _auth(self, token):
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')

    def test_unauthenticated_rejected(self):
        response = self.client.get('/api/recommendations/')
        self.assertEqual(response.status_code, 401)

    def test_empty_when_user_rated_entire_catalog(self):
        _rate(self.alice, 1, 5)
        self._auth(self.alice_token)
        response = self.client.get('/api/recommendations/')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['method'], 'empty')
        self.assertEqual(body['results'], [])
        self.assertIn('unseen', body['message'].lower())

    def test_does_not_recommend_already_rated_or_self_titles(self):
        _rate(self.alice, 1, 5, 'Inception')
        _rate(self.alice, 2, 5, 'Batman')
        _rate(self.bob, 1, 5, 'Inception')
        _rate(self.bob, 3, 5, 'Parasite')
        self._auth(self.alice_token)
        body = self.client.get('/api/recommendations/').json()
        ids = [row['movie_id'] for row in body['results']]
        self.assertNotIn(1, ids)
        self.assertNotIn(2, ids)
        self.assertIn(3, ids)
        self.assertEqual(body['method'], 'item_item')
        self.assertTrue(all('predicted_score' not in row for row in body['results']))

    def test_popularity_fallback_is_count_then_mean_then_id(self):
        # Orthogonal histories so user-user cosine is 0; popularity must kick in.
        _rate(self.alice, 1, 5)
        _rate(self.bob, 10, 5, 'Ten')
        _rate(self.carol, 10, 4, 'Ten')
        _rate(self.bob, 11, 5, 'Eleven')
        _rate(self.carol, 11, 5, 'Eleven')
        _rate(self.bob, 12, 5, 'Twelve')
        # 11: count 2 mean 5; 10: count 2 mean 4.5; 12: count 1 mean 5
        self._auth(self.alice_token)
        body = self.client.get('/api/recommendations/').json()
        self.assertEqual(body['method'], 'fallback_popular')
        ids = [row['movie_id'] for row in body['results']]
        self.assertEqual(ids, [11, 10, 12])
        self.assertNotIn(1, ids)

    def test_popularity_tie_breaks_on_item_id(self):
        _rate(self.alice, 1, 5)
        _rate(self.bob, 30, 5, 'Thirty')
        _rate(self.bob, 20, 5, 'Twenty')
        self._auth(self.alice_token)
        ids = [
            row['movie_id']
            for row in self.client.get('/api/recommendations/').json()['results']
        ]
        self.assertEqual(ids[:2], [20, 30])

    def test_cold_start_user_gets_popular_unseen_not_empty_if_catalog_exists(self):
        _rate(self.bob, 5, 5, 'Five')
        _rate(self.carol, 5, 5, 'Five')
        _rate(self.bob, 6, 4, 'Six')
        self._auth(self.alice_token)
        body = self.client.get('/api/recommendations/').json()
        self.assertEqual(body['method'], 'fallback_popular')
        self.assertEqual([row['movie_id'] for row in body['results']], [5, 6])

    def test_create_update_and_delete_rating_change_recommendations(self):
        _rate(self.alice, 1, 5, 'One')
        _rate(self.bob, 1, 5, 'One')
        _rate(self.bob, 3, 5, 'Three')
        self._auth(self.alice_token)
        first = self.client.get('/api/recommendations/').json()
        self.assertIn(3, [row['movie_id'] for row in first['results']])

        created = self.client.post(
            '/api/ratings/',
            {'movie_id': 3, 'title': 'Three', 'score': 5},
            format='json',
        )
        self.assertEqual(created.status_code, 201)
        after_create = self.client.get('/api/recommendations/').json()
        self.assertNotIn(3, [row['movie_id'] for row in after_create['results']])

        updated = self.client.post(
            '/api/ratings/',
            {'movie_id': 3, 'title': 'Three', 'score': 2},
            format='json',
        )
        self.assertEqual(updated.status_code, 201)
        after_update = self.client.get('/api/recommendations/').json()
        self.assertNotIn(3, [row['movie_id'] for row in after_update['results']])
        self.assertEqual(Rating.objects.get(user=self.alice, movie_id=3).score, 2)

        deleted = self.client.delete('/api/ratings/3/')
        self.assertEqual(deleted.status_code, 204)
        after_delete = self.client.get('/api/recommendations/').json()
        self.assertIn(3, [row['movie_id'] for row in after_delete['results']])
