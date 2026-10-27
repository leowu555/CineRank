from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase


class AuthAPITests(APITestCase):
    def test_register_returns_token(self):
        response = self.client.post(
            '/api/auth/register/',
            {'username': 'pat', 'password': 'secret1', 'email': 'pat@example.com'},
            format='json',
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertIn('token', body)
        self.assertEqual(body['user']['username'], 'pat')
        self.assertTrue(Token.objects.filter(user__username='pat').exists())

    def test_login_required_fields(self):
        response = self.client.post('/api/auth/login/', {}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_login_rejects_bad_password(self):
        User.objects.create_user(username='pat', password='secret1')
        response = self.client.post(
            '/api/auth/login/',
            {'username': 'pat', 'password': 'wrong'},
            format='json',
        )
        self.assertEqual(response.status_code, 400)

    def test_me_requires_token(self):
        response = self.client.get('/api/auth/me/')
        self.assertEqual(response.status_code, 401)

    def test_me_and_logout(self):
        user = User.objects.create_user(username='pat', password='secret1')
        token = Token.objects.create(user=user)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')
        me = self.client.get('/api/auth/me/')
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()['username'], 'pat')
        logout = self.client.post('/api/auth/logout/')
        self.assertEqual(logout.status_code, 200)
        self.assertFalse(Token.objects.filter(user=user).exists())
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')
        self.assertEqual(self.client.get('/api/auth/me/').status_code, 401)
