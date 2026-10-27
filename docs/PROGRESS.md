# CineRank progress

## Current phase

Stages **1–3** of the interview milestone are implemented and re-verified. Stage **4** (TF-IDF / RRF hybrid) is **deferred**: source remains in the repo but is not on the API and is not in the default eval artifact.

## Completed (this milestone)

- **Stage 1.** No self-recommendations; popularity fallback is count → mean → item_id; honest `empty`; ratings ignore/reject body `user`/`user_id`; no `predicted_score`. Tests: auth, rating ownership, score bounds, already-rated exclusion, ranking, sparse/cold-start.
- **Stage 2.** `recommendations/recsys` (Django-free): sparse item–item cosine, `ItemItemModel.fit` then `recommend`. `GET /api/recommendations/` uses it (`method: item_item`). Per-request re-fit; no versioned artifacts.
- **Stage 3.** MovieLens `ml-latest-small` (URL + SHA-256), chronological 80/10/10, train-only fit, tune k on validation, Recall@10 / NDCG@10 / coverage. Results: [`docs/experiments/movielens_small.json`](experiments/movielens_small.json). Popularity beat item–item on **test**.

## Verification pass (stages 1–3)

- SQLite: `DJANGO_SETTINGS_MODULE=config.test_settings` → **55 tests OK** (in-memory).
- Isolated Postgres: `DJANGO_SETTINGS_MODULE=config.test_settings_postgres` → **55 tests OK**; Django created/dropped `cinerank_isolated_test` (not `movie_recommender`).
- Eval: `python -m recommendations.recsys.cli --skip-download` → 74 = 54 no history + 2 no test rating≥4 + 0 outside-catalog-only + 18 eligible; popularity wins test NDCG@10.
- Frontend: `npm run build` OK (Vite 8.0.7). Live login→rate→For You against the **development** DB was **not** run (would mutate seed ratings). Create/update/delete vs recs is covered by `test_create_update_and_delete_rating_change_recommendations`.

## Checks run (2026-10-27)

```text
DJANGO_SETTINGS_MODULE=config.test_settings python manage.py test
→ 53 tests OK (in-memory SQLite; development Postgres not used)

python -m recommendations.recsys.cli --skip-download
→ wrote docs/experiments/movielens_small.json (hybrid key omitted)
  test popularity NDCG@10 ≈ 0.104; item-item k=40 NDCG@10 ≈ 0.061; 18 eligible users
```

Not re-run: live TMDB, Lambda, `npm run build`, pip-compile.

## Deferred / not demo

- Stage 4 hybrid (`--include-hybrid` on the CLI only).
- Docker, Redis, CI, TypeScript, PyTorch, LightGBM, persistent watchlist.

## Exact next task

Review stages 1–3. After that: either persist item–item neighbor artifacts so HTTP does not re-fit, or run stage 4 hybrid eval as a separate offline report. Do not wire hybrid into the app until that review.
