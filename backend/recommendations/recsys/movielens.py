"""MovieLens download and loaders. IDs here are MovieLens ids, never Django user pks."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

from recommendations.recsys.types import RatingEvent

ML_SMALL_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# Recorded after a successful download of ml-latest-small.zip (GroupLens).
ML_SMALL_SHA256 = '696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436'
DATASET_NAME = 'ml-latest-small'


class DatasetChecksumError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_ml_small(dest_dir: Path, url: str = ML_SMALL_URL) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / 'ml-latest-small.zip'
    extract_dir = dest_dir / DATASET_NAME
    ratings_path = extract_dir / 'ratings.csv'
    if ratings_path.exists() and (extract_dir / 'movies.csv').exists():
        return extract_dir

    with urlopen(url, timeout=60) as response:
        payload = response.read()
    digest = sha256_bytes(payload)
    if digest != ML_SMALL_SHA256:
        # First download: caller may pass expected=None by writing the file then
        # raising with the observed digest so we can pin it.
        zip_path.write_bytes(payload)
        raise DatasetChecksumError(
            f'Checksum mismatch for {url}: got {digest}, expected {ML_SMALL_SHA256}. '
            f'Zip saved at {zip_path} so the pin can be updated after inspection.'
        )
    zip_path.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest_dir)
    if not ratings_path.exists():
        raise FileNotFoundError(f'ratings.csv missing after extract in {dest_dir}')
    return extract_dir


def load_ratings_csv(path: Path) -> list[RatingEvent]:
    events: list[RatingEvent] = []
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            events.append(
                RatingEvent(
                    user_id=int(row['userId']),
                    item_id=int(row['movieId']),
                    score=float(row['rating']),
                    timestamp=float(row['timestamp']),
                )
            )
    return events


def load_movies_csv(path: Path) -> dict[int, dict[str, str]]:
    movies: dict[int, dict[str, str]] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            movies[int(row['movieId'])] = {
                'title': row.get('title', ''),
                'genres': row.get('genres', ''),
            }
    return movies


def load_links_csv(path: Path) -> dict[int, int | None]:
    """MovieLens movieId -> TMDB id. Missing/blank tmdbId maps to None."""
    mapping: dict[int, int | None] = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = (row.get('tmdbId') or '').strip()
            mapping[int(row['movieId'])] = int(raw) if raw else None
    return mapping
