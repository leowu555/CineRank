"""MovieLens movieId ↔ TMDB id. Application ratings use TMDB ids; eval uses MovieLens ids."""


def movielens_to_tmdb(links: dict[int, int | None], movie_id: int) -> int | None:
    tmdb_id = links.get(movie_id)
    return tmdb_id if tmdb_id else None


def tmdb_to_movielens(links: dict[int, int | None], tmdb_id: int) -> int | None:
    for movie_id, mapped in links.items():
        if mapped == tmdb_id:
            return movie_id
    return None


def mapped_tmdb_ids(links: dict[int, int | None], movie_ids: list[int]) -> list[int]:
    out: list[int] = []
    for movie_id in movie_ids:
        tmdb_id = movielens_to_tmdb(links, movie_id)
        if tmdb_id is not None:
            out.append(tmdb_id)
    return out
