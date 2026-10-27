from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase

from ratings.models import Rating


class RatingAPITests(APITestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='secret1')
        self.bob = User.objects.create_user(username='bob', password='secret1')
        self.alice_token = Token.objects.create(user=self.alice)
        self.bob_token = Token.objects.create(user=self.bob)

    def _auth(self, token):
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')

    def test_unauthenticated_list_rejected(self):
        response = self.client.get('/api/ratings/')
        self.assertEqual(response.status_code, 401)

    def test_unauthenticated_create_rejected(self):
        response = self.client.post(
            '/api/ratings/',
            {'movie_id': 1, 'title': 'X', 'score': 5},
            format='json',
        )
        self.assertEqual(response.status_code, 401)

    def test_score_must_be_between_1_and_5(self):
        self._auth(self.alice_token)
        low = self.client.post(
            '/api/ratings/',
            {'movie_id': 1, 'title': 'X', 'score': 0},
            format='json',
        )
        high = self.client.post(
            '/api/ratings/',
            {'movie_id': 1, 'title': 'X', 'score': 6},
            format='json',
        )
        self.assertEqual(low.status_code, 400)
        self.assertEqual(high.status_code, 400)
        self.assertEqual(Rating.objects.count(), 0)

    def test_create_is_owned_by_authenticated_user(self):
        self._auth(self.alice_token)
        response = self.client.post(
            '/api/ratings/',
            {'movie_id': 27205, 'title': 'Inception', 'score': 5},
            format='json',
        )
        self.assertEqual(response.status_code, 201)
        rating = Rating.objects.get()
        self.assertEqual(rating.user_id, self.alice.id)
        self.assertEqual(rating.movie_id, 27205)

    def test_cannot_assign_another_user_via_body(self):
        self._auth(self.alice_token)
        response = self.client.post(
            '/api/ratings/',
            {
                'movie_id': 155,
                'title': 'The Dark Knight',
                'score': 5,
                'user': self.bob.id,
                'user_id': self.bob.id,
            },
            format='json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(Rating.objects.count(), 0)

    def test_list_and_detail_are_owner_scoped(self):
        Rating.objects.create(user=self.alice, movie_id=1, title='A', score=5)
        Rating.objects.create(user=self.bob, movie_id=2, title='B', score=4)
        self._auth(self.alice_token)
        listing = self.client.get('/api/ratings/')
        ids = {row['movie_id'] for row in listing.json()['results']}
        self.assertEqual(ids, {1})
        self.assertEqual(self.client.get('/api/ratings/2/').json(), {'rated': False})
        own = self.client.get('/api/ratings/1/')
        self.assertTrue(own.json()['rated'])
        self.assertEqual(own.json()['score'], 5)

    def test_cannot_delete_another_users_rating(self):
        Rating.objects.create(user=self.bob, movie_id=9, title='B', score=5)
        self._auth(self.alice_token)
        response = self.client.delete('/api/ratings/9/')
        self.assertEqual(response.status_code, 404)
        self.assertTrue(Rating.objects.filter(user=self.bob, movie_id=9).exists())
