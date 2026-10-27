"""PostgreSQL test runner that creates/drops an isolated TEST database.

Uses the same connection user/host as local Django settings, but Django's
test runner creates `cinerank_isolated_test` and does not write into
`movie_recommender` (or whatever DB_NAME is). Do not point TEST.NAME at the
development database.
"""

import os

os.environ.setdefault('SECRET_KEY', 'test-secret-key-not-for-production')
os.environ.setdefault('DEBUG', 'true')

from config.settings import *  # noqa: E402,F403

DATABASES['default']['TEST'] = {
    'NAME': 'cinerank_isolated_test',
}
