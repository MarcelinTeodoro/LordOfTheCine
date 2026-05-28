
from pathlib import Path
import sys
from collections.abc import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MOVIES_PROCESSED_PATH = PROCESSED_DATA_DIR / "movies_processed.parquet"


def load_movies_processed(path: Path = MOVIES_PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed movie dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {path}. Execute primeiro: python src/data/preprocessing.py"
        )

    return pd.read_parquet(path)


def _as_genre_set(value: object) -> set[str]:
    if isinstance(value, Iterable) and not isinstance(value, str):
        return {genre for genre in value if genre}

    if isinstance(value, str) and value:
        return {genre for genre in value.split("|") if genre and genre != "(no genres listed)"}

    return set()


def _as_tag_set(value: object) -> set[str]:
    if not isinstance(value, str) or not value.strip():
        return set()

    return {tag for tag in value.split() if tag}


def build_preference_summary(
    user_ratings: dict[int, float],
    movies: pd.DataFrame,
    liked_threshold: float = 4.0,
) -> dict[str, set[str]]:
    """Collect genres and tags from movies the user rated positively."""
    liked_movie_ids = [movie_id for movie_id, rating in user_ratings.items() if rating >= liked_threshold]
    liked_movies = movies[movies["movieId"].isin(liked_movie_ids)]

    preferred_genres: set[str] = set()
    preferred_tags: set[str] = set()

    for _, movie in liked_movies.iterrows():
        preferred_genres.update(_as_genre_set(movie.get("genres_list", movie.get("genres"))))
        preferred_tags.update(_as_tag_set(movie.get("tags_text")))

    return {
        "genres": preferred_genres,
        "tags": preferred_tags,
    }


def explain_recommendation(
    recommendation: pd.Series,
    preference_summary: dict[str, set[str]],
) -> str:
    """Create a human-readable explanation for a recommendation."""
    movie_genres = _as_genre_set(recommendation.get("genres_list", recommendation.get("genres")))
    movie_tags = _as_tag_set(recommendation.get("tags_text"))

    shared_genres = sorted(movie_genres.intersection(preference_summary["genres"]))
    shared_tags = sorted(movie_tags.intersection(preference_summary["tags"]))

    reasons: list[str] = []
    if shared_genres:
        reasons.append(f"combina com seus generos preferidos: {', '.join(shared_genres[:3])}")

    if shared_tags:
        reasons.append(f"tem tags relacionadas ao seu perfil: {', '.join(shared_tags[:3])}")

    profile_score = recommendation.get("profile_score")
    if pd.notna(profile_score) and profile_score >= 0.30:
        reasons.append("possui alta similaridade textual com filmes que voce avaliou bem")

    rating_mean = recommendation.get("rating_mean")
    rating_count = recommendation.get("rating_count")
    if pd.notna(rating_mean) and pd.notna(rating_count) and rating_count >= 20:
        reasons.append(f"tambem e bem avaliado no dataset, com media {rating_mean:.2f}")

    if not reasons:
        reasons.append("apresenta sinais positivos no score hibrido do recomendador")

    return "Recomendado porque " + "; ".join(reasons) + "."


def add_explanations(
    recommendations: pd.DataFrame,
    user_ratings: dict[int, float],
    movies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add an explanation column to a recommendations dataframe."""
    if movies is None:
        movies = load_movies_processed()

    preference_summary = build_preference_summary(user_ratings, movies)

    explanation_input = recommendations.merge(
        movies[["movieId", "genres_list", "tags_text"]],
        on="movieId",
        how="left",
    )
    explained = recommendations.copy()
    explained["explanation"] = explanation_input.apply(
        lambda row: explain_recommendation(row, preference_summary),
        axis=1,
    )

    return explained


def main() -> None:
    """Run a small smoke test for recommendation explanations."""
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.hybrid_recommender import HybridRecommender

    user_ratings = {
        1: 5.0,     # Toy Story
        3114: 4.5,  # Toy Story 2
        260: 5.0,   # Star Wars
        296: 2.0,   # Pulp Fiction
    }

    recommender = HybridRecommender()
    recommendations = recommender.recommend(user_ratings, top_k=5, min_rating_count=20)
    explained = add_explanations(recommendations, user_ratings, recommender.movies)

    print(explained[["title", "hybrid_score", "explanation"]].to_string(index=False))


if __name__ == "__main__":
    main()
