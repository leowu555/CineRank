"""Shared cutoffs. App and MovieLens eval do not share a rating table.

APP_NEIGHBOR_K is used when scoring the live TMDB/Postgres table (re-fit per
request). EVAL_NEIGHBOR_K_GRID is tuned on MovieLens validation only; the
chosen k is not copied into the Django view. Offline metrics are therefore
not measurements of the running app.
"""

# HTTP item–item: neighbors kept per item while scoring application ratings.
APP_NEIGHBOR_K = 20

# Offline list length for Recall@K / NDCG@K / Coverage@K.
EVAL_LIST_K = 10

# MovieLens-only hyperparameter grid (validation NDCG@EVAL_LIST_K).
EVAL_NEIGHBOR_K_GRID = (10, 20, 40)
