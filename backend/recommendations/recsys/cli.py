"""MovieLens evaluation CLI. Run from backend/: python -m recommendations.recsys.cli"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from recommendations.recsys.config import APP_NEIGHBOR_K, EVAL_LIST_K, EVAL_NEIGHBOR_K_GRID
from recommendations.recsys.evaluate import evaluate_item_item, evaluate_popularity, tune_neighbor_k
from recommendations.recsys.movielens import (
    DATASET_NAME,
    ML_SMALL_SHA256,
    ML_SMALL_URL,
    DatasetChecksumError,
    download_ml_small,
    load_links_csv,
    load_movies_csv,
    load_ratings_csv,
)
from recommendations.recsys.splits import chronological_split


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate CineRank recommenders on MovieLens.')
    parser.add_argument('--data-dir', type=Path, default=_repo_root() / 'data')
    parser.add_argument(
        '--output',
        type=Path,
        default=_repo_root() / 'docs' / 'experiments' / 'movielens_small.json',
    )
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument(
        '--include-hybrid',
        action='store_true',
        help='Deferred: TF-IDF + RRF hybrid (not part of the interview demo)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir
    extract_dir = data_dir / DATASET_NAME
    ratings_path = extract_dir / 'ratings.csv'

    download_info = {'attempted': False, 'ok': ratings_path.exists(), 'error': None}
    if not args.skip_download and not ratings_path.exists():
        download_info['attempted'] = True
        try:
            extract_dir = download_ml_small(data_dir)
            ratings_path = extract_dir / 'ratings.csv'
            download_info['ok'] = True
        except DatasetChecksumError as exc:
            download_info['error'] = str(exc)
            print(exc, file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 — CLI boundary
            download_info['error'] = str(exc)
            print(f'Download failed: {exc}', file=sys.stderr)
            if not ratings_path.exists():
                print('Place ml-latest-small/ratings.csv under --data-dir and rerun with --skip-download.', file=sys.stderr)
                return 2

    if not ratings_path.exists():
        print(f'Missing {ratings_path}', file=sys.stderr)
        return 2

    started = time.perf_counter()
    ratings = load_ratings_csv(ratings_path)
    split = chronological_split(ratings)
    catalog = {event.item_id for event in split.train}

    pop_val = evaluate_popularity(split.train, split.validation)
    best_k, val_grid = tune_neighbor_k(split.train, split.validation)
    item_val = next(row for row in val_grid if row.extra['neighbor_k'] == best_k)
    pop_test = evaluate_popularity(split.train, split.test)
    item_test = evaluate_item_item(split.train, split.test, neighbor_k=best_k)

    hybrid_block = None
    if args.include_hybrid:
        from recommendations.recsys.hybrid_eval import evaluate_content_and_hybrid

        movies = load_movies_csv(extract_dir / 'movies.csv')
        hybrid_block = evaluate_content_and_hybrid(
            split.train, split.validation, split.test, movies, neighbor_k=best_k
        )

    payload = {
        'dataset': {
            'name': DATASET_NAME,
            'source': ML_SMALL_URL,
            'sha256': ML_SMALL_SHA256,
            'identity': 'MovieLens userId/movieId only. Not Django auth.User pks. Not TMDB ids.',
            'rating_count': len(ratings),
            'train_count': len(split.train),
            'validation_count': len(split.validation),
            'test_count': len(split.test),
            'train_end_timestamp': split.train_end_ts,
            'validation_end_timestamp': split.validation_end_ts,
            'test_end_timestamp': split.test_end_ts,
            'train_catalog_size': len(catalog),
        },
        'protocol': {
            'split': 'global chronological 80/10/10 by rating count; sort timestamp, user_id, item_id',
            'fit': 'train ratings only (popularity counts/means and item cosine neighbors). Validation and test rows are never passed to fit.',
            'tune': 'item-item neighbor_k on validation NDCG@10 only; grid EVAL_NEIGHBOR_K_GRID; test is not used',
            'test': 'single evaluation with chosen MovieLens k; model still train-only',
            'relevance': 'rating >= 4.0 on the eval split, item must appear in the train catalog',
            'exclude': 'items the user rated in train (same for popularity and item-item)',
            'user_eligibility': 'same cohort for every strategy: train history AND at least one relevant in-catalog eval item',
            'item_candidates': (
                'Both methods may only return train-catalog ids not in the user train history. '
                'Popularity ranks that full unseen catalog. Item-item only ranks neighbor-linked items '
                'and may return fewer than 10.'
            ),
            'coverage_at_10': (
                '|union of each eligible user top-10| / |train catalog|. '
                'Shorter item-item lists still count only the ids they returned.'
            ),
            'k': EVAL_LIST_K,
            'app_neighbor_k': APP_NEIGHBOR_K,
            'eval_neighbor_k_grid': list(EVAL_NEIGHBOR_K_GRID),
            'app_vs_eval': (
                'Live API re-fits item-item on Postgres TMDB ratings with APP_NEIGHBOR_K=20. '
                'These MovieLens metrics use a different id space and a tuned neighbor_k. '
                'They are not measurements of the running app.'
            ),
            'positive_history_threshold': 4,
            'item_item_score': 'sum_i sim(i,j) * r(u,i) for i rated >= 4',
        },
        'test_cohort': pop_test.stats,
        'validation': {
            'popularity': _strategy_dict(pop_val),
            'item_item_grid': [_strategy_dict(row) for row in val_grid],
            'chosen_neighbor_k': best_k,
            'item_item': _strategy_dict(item_val),
        },
        'test': {
            'popularity': _strategy_dict(pop_test),
            'item_item': _strategy_dict(item_test),
        },
        'hybrid': hybrid_block,
        'runtime_seconds': round(time.perf_counter() - started, 3),
        'reproduce': [
            'cd backend',
            'python -m recommendations.recsys.cli',
        ],
        'download': download_info,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload['test'], indent=2))
    print(f'Wrote {args.output}')
    return 0


def _strategy_dict(result) -> dict:
    return {
        'name': result.name,
        'recall_at_10': result.recall,
        'ndcg_at_10': result.ndcg,
        'catalog_coverage_at_10': result.coverage,
        'eligible_users': result.eligible_users,
        'stats': result.stats,
        'extra': result.extra,
    }


if __name__ == '__main__':
    raise SystemExit(main())
