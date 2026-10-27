# CineRank

CineRank is a local movie discovery app: search titles from TMDB, sign in, rate films 1–5, and get a **For You** list of titles you have not rated. Recommendations use **sparse item–item cosine** collaborative filtering on the ratings table, with a **popularity** fallback when that list is empty.

Personalized For You, accounts, and ratings run on **Django + PostgreSQL**. Movie search and details are a **TMDB HTTP proxy**. A React (Vite) UI talks to the API over REST.

---

## What you can do

| Flow | Where |
|------|--------|
| Search by title, sort the result list in the browser | Home `/` |
| Open details (overview, genres, runtime, poster) | `/movie/:id` |
| Register, log in, log out | `/register`, `/login` |
| Rate 1–5 (create or overwrite) | Movie details, while signed in |
| See your ratings | Profile `/profile` |
| Unseen recommendations | For You `/recommendations` (auth required) |
| Watchlist | `/watchlist` — **this browser only** (`localStorage`) |

There is no in-app control to delete a rating. `DELETE /api/ratings/<movie_id>/` exists on the API.

---

## Technologies in use

| Technology | Role |
|------------|------|
| Python, Django 6, Django REST Framework | HTTP API, auth, ORM |
| DRF `TokenAuthentication` | `Authorization: Token …` |
| PostgreSQL (`psycopg2-binary`) | Users, tokens, ratings |
| `requests` | TMDB search and details |
| NumPy, SciPy, scikit-learn | Sparse matrices and cosine similarity |
| React 19, Vite 8, React Router 7 | UI (JavaScript, not TypeScript) |
| python-dotenv, django-cors-headers | Local env and CORS |
| Mangum | Optional AWS Lambda **adapter** for the same ASGI app (packaging only; see below) |

---

## Architecture

```text
Browser (Vite, default http://127.0.0.1:5173)
    REST JSON
Django / DRF (default http://127.0.0.1:8000)
    ├── TMDB  (search + details; no movie table)
    └── PostgreSQL  (auth.User, authtoken.Token, ratings.Rating)
            └── recommendations/recsys  (fit + score in the API process)
```

Movie identity in the app is the **TMDB integer** stored on `ratings.Rating.movie_id`. Title and poster are copied onto the rating row for display. There is no `Movie` model.

**For You:** `GET /api/recommendations/` (token required) loads **all** rating rows, converts them to `RatingEvent`s, and calls `recommendations.recsys.pipeline.recommend` (default `cf='item_item'`). The process **fits item neighbors on that full table every request**. Nothing is written to a model file.

---

## Authentication and data

- Register / login return a DRF token. The UI stores token and user JSON in `localStorage` and sends `Authorization: Token <key>`.
- Logout deletes the server token. `/api/auth/me/` requires a valid token.
- Ratings are always attached to `request.user`. A body field `user` or `user_id` is rejected. List/get/delete are scoped to the current user (`404` if the row is not yours).
- Unique `(user, movie_id)` is enforced in the database. Score 1–5 is enforced in `RatingSerializer`, not as a database check constraint.

---

## Recommendations (live app)

**Item–item cosine** (`ItemItemModel` in `recommendations/recsys/item_item.py`):

1. Build a SciPy **CSR** matrix of shape **(items × users)** from **observed** ratings only.
2. For each item, compute cosine similarity to all items (one dense similarity **row** at a time; no full dense item×item matrix is allocated). Ignore self-similarity. Keep up to **`APP_NEIGHBOR_K = 20`** neighbors with similarity **> 0**.
3. For the current user, take items they rated **≥ 4**. Accumulate  
   `score(j) = Σ_i sim(i,j) × r(u,i)`  
   for neighbors `j` they have not already rated.
4. Sort by that weight descending, then `item_id` ascending. Return up to 10 titles.

That weight is a **ranking score**, not a predicted star rating. The JSON does not include `predicted_score`.

**Missing ratings:** they are simply absent from the CSR (a structural zero in the sparse vector). Cosine still uses the usual dot product, so a missing rating contributes 0 to the product. Sparsity saves memory; it does not encode “unknown” as a distinct value.

**Popularity fallback** (`rank_popularity`): if item–item returns nothing but other titles remain, rank unseen items by **rating count** (desc), **mean score** (desc), **item_id** (asc). `method` is `fallback_popular`.

**Empty:** if the user has already rated every movie that appears in the ratings table, `method` is `empty`.

A signed-in user with **no** ratings still gets popularity of the whole catalog (`fallback_popular`) when other people have rated something.

HTTP default is item–item. A legacy user–user implementation remains in the package (`cf='user_user'`) and is **not** what the API calls.

---

## Live app vs MovieLens evaluation

| | Application | Offline eval |
|--|-------------|--------------|
| Users | Django `auth.User` | MovieLens `userId` |
| Movies | TMDB ids on `Rating` | MovieLens `movieId` |
| Data | PostgreSQL | `data/ml-latest-small/` (not loaded into Django) |
| Neighbor `k` | `APP_NEIGHBOR_K = 20` | grid `{10,20,40}` on **validation**; test used **40** |
| Fit | Every For You request | Once on the **train** split |

Those metrics are **not** measurements of live For You quality.

**Dataset:** [ml-latest-small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip), SHA-256 `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`. 100,836 ratings → 80,668 / 10,083 / 10,085 by global time (80/10/10). Train catalog 7,867 movies. Fit train only. Relevance: eval-split rating ≥ 4 **and** movie in the train catalog. History for scoring is **train only** (validation ratings are not added at test time).

**Test cohort (same 18 users for both methods):** 74 test users = 54 with no train history + 2 with no test rating ≥ 4 + 0 excluded only for catalog-miss + **18 eligible**. Among users who had ≥4 test titles, 235 **user–movie** relevant pairs (193 distinct movies, 15 of the 18 eligible users) were dropped because the movie was not in the train catalog.

**Test results** (from `docs/experiments/movielens_small.json`, CLI run 2026-10-27):

| Strategy | Recall@10 | NDCG@10 | Coverage@10 |
|----------|-----------|---------|-------------|
| Popularity | **0.047** | **0.104** | 0.007 |
| Item–item (`k=40`) | 0.034 | 0.061 | **0.014** |

**Popularity is ahead on this test set.** Item–item was ahead on validation. Coverage@10 is `|union of the 18 lists’ top-10| / 7867`. Higher coverage here is not higher accuracy. Eighteen users is a small cohort; a global time split leaves most test users without train history.

**Coverage@10** uses only ids actually returned (item–item lists may be shorter than 10).

---

## Demo (local)

Seed accounts (optional; writes the development database):

```bash
cd backend
python manage.py shell -c "exec(open('seed_demo_data.py').read())"
```

`alice` / `bob` / `carol`, password `demo1234`.

1. Open **http://127.0.0.1:5173** (prefer `127.0.0.1` so CORS matches the API).
2. Search **Inception**.
3. Log in as `alice`.
4. Open **For You**. With only the seed ratings, item–item should list **Pulp Fiction**, **Parasite**, and **Forrest Gump** (`method`: `item_item`).
5. Profile shows saved ratings. Watchlist is device-local.

This walkthrough is implied by the seed file plus the current recommender; a **browser session was not re-verified** in the last check (dev servers were not running).

---

## Setup

Needs Python 3.13 (lock verified on 3.13), Node for the UI, PostgreSQL, a TMDB API key. Copy `backend/.env.example` to `backend/.env` if needed; set `SECRET_KEY`, `TMDB_API_KEY`, and `DB_*`. Do not overwrite an existing `.env`. Variables: `docs/CONFIGURATION.md`.

```bash
# API
cd backend
python3.13 -m venv venv          # only if venv/ does not exist
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

```bash
# UI
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

```bash
# Tests (do not use the development database)
cd backend
DJANGO_SETTINGS_MODULE=config.test_settings python manage.py test
# optional: isolated Postgres database cinerank_isolated_test
DJANGO_SETTINGS_MODULE=config.test_settings_postgres python manage.py test
```

Recorded: **55 tests OK** on in-memory SQLite and on isolated PostgreSQL (2026-10-27). **`npm run build`** succeeded (Vite 8.0.7). Live login in a browser was **not** recorded in that pass.

```bash
# MovieLens eval (writes docs/experiments/movielens_small.json)
cd backend
python -m recommendations.recsys.cli
# if data/ml-latest-small is already extracted:
python -m recommendations.recsys.cli --skip-download
```

`--include-hybrid` is an **experimental** extra (TF-IDF + rank fusion). It is **not** used by the API. The default JSON has `"hybrid": null`. Do not treat it as a shipped feature.

**Lambda:** `config.lambda_handler.handler` wraps Django ASGI with Mangum (`lifespan="off"`). Zip packaging is documented under `docs/DEVELOPMENT.md`. That is integration code, not a verified live deploy. Even if a Function URL were configured, **For You still needs PostgreSQL**; the documented Lambda path is the TMDB proxy, not personalized recommendations.

---

## Layout

```text
backend/config/           settings, URLs, env, Lambda adapter, test settings
backend/accounts/         register, login, logout, me
backend/movies/           TMDB search + details
backend/ratings/           Rating model + CRUD
backend/recommendations/  HTTP view + recsys/ (algorithms, eval CLI)
backend/seed_demo_data.py
frontend/src/pages/        UI screens
frontend/src/context/      AuthContext
docs/experiments/          MovieLens result JSON
```

---

## API

Auth header: `Authorization: Token <token>`

| Method | Path | Auth |
|--------|------|------|
| GET | `/api/movies/search?query=` | no |
| GET | `/api/movies/<tmdb_id>/` | no |
| POST | `/api/auth/register/`, `/login/` | no |
| GET | `/api/auth/me/` | yes |
| POST | `/api/auth/logout/` | yes |
| GET, POST | `/api/ratings/` | yes (own rows; POST upserts, HTTP 201) |
| GET, DELETE | `/api/ratings/<movie_id>/` | yes |
| GET | `/api/recommendations/` | yes |

Rating body: `{ "movie_id", "title", "poster_url", "score" }` with `score` in 1–5.

For You body: `{ "results": [{ "movie_id", "title", "poster_url", "reason" }], "method", "message"? }`  
`method` is `item_item`, `fallback_popular`, or `empty`.

---

## Limitations (verified)

- For You **re-fits on every request** over the full ratings table; there is no saved neighbor artifact.
- App neighbor **k = 20**; MovieLens test used **k = 40**.
- Watchlist and recent searches are **localStorage**, not per account.
- Rating **delete** is API-only; the details page only create/updates via POST.
- Score range is serializer-only; `PositiveSmallIntegerField` has no 1–5 check in SQL.
- MovieLens test **N = 18**; popularity **outperformed** item–item there.
- No browser walkthrough or live AWS function was verified in the last check.
- The Django `watchlist` app is installed but has no URLs or models.
