"""Offline quantitative evaluation for recommendation models."""

from __future__ import annotations

from collections.abc import Iterable
import inspect
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Supports direct execution with: python src/evaluation/evaluate.py
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    catalog_coverage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.models.hybrid_recommender import HybridRecommender  # noqa: E402


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MOVIES_PATH = PROCESSED_DATA_DIR / "movies_processed.parquet"
RATINGS_PATH = PROCESSED_DATA_DIR / "ratings_processed.parquet"
RESULTS_PATH = PROJECT_ROOT / "reports" / "evaluation_results.csv"

SEED = 42
MIN_USER_RATINGS = 20
TRAIN_RATIO = 0.80
RELEVANT_RATING = 4.0
TOP_K = 10
HYBRID_CONFIGURATIONS = {
    "Hybrid 70/20/10": {
        "profile_weight": 0.70,
        "rating_weight": 0.20,
        "popularity_weight": 0.10,
    },
    "Hybrid 60/20/20": {
        "profile_weight": 0.60,
        "rating_weight": 0.20,
        "popularity_weight": 0.20,
    },
    "Hybrid 50/30/20": {
        "profile_weight": 0.50,
        "rating_weight": 0.30,
        "popularity_weight": 0.20,
    },
    "Hybrid 40/30/30": {
        "profile_weight": 0.40,
        "rating_weight": 0.30,
        "popularity_weight": 0.30,
    },
    "Hybrid 30/30/40": {
        "profile_weight": 0.30,
        "rating_weight": 0.30,
        "popularity_weight": 0.40,
    },
}


def load_evaluation_data(
    movies_path: Path = MOVIES_PATH,
    ratings_path: Path = RATINGS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the processed datasets used by the evaluation."""
    for path in (movies_path, ratings_path):
        if not path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    movies = pd.read_parquet(movies_path)
    ratings = pd.read_parquet(ratings_path)

    movie_columns = {"movieId", "rating_mean", "rating_count"}
    rating_columns = {"userId", "movieId", "rating"}
    missing_movies = movie_columns.difference(movies.columns)
    missing_ratings = rating_columns.difference(ratings.columns)
    if missing_movies:
        raise ValueError(f"Colunas ausentes em movies_processed: {sorted(missing_movies)}")
    if missing_ratings:
        raise ValueError(f"Colunas ausentes em ratings_processed: {sorted(missing_ratings)}")

    return movies.reset_index(drop=True), ratings.reset_index(drop=True)


def select_eligible_ratings(
    ratings: pd.DataFrame,
    min_user_ratings: int = MIN_USER_RATINGS,
) -> pd.DataFrame:
    """Keep ratings from users with enough interactions for an 80/20 split."""
    if min_user_ratings < 2:
        raise ValueError("min_user_ratings must be at least 2.")

    user_counts = ratings.groupby("userId")["movieId"].transform("size")
    return ratings[user_counts >= min_user_ratings].copy()


def split_ratings_by_user(
    ratings: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Create reproducible random train/test partitions independently per user."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    user_splits: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for user_id, user_ratings in ratings.groupby("userId", sort=True):
        user_ratings = user_ratings.reset_index(drop=True)
        indices = rng.permutation(len(user_ratings))
        train_size = min(max(int(len(user_ratings) * train_ratio), 1), len(user_ratings) - 1)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        user_splits[int(user_id)] = (
            user_ratings.iloc[train_indices].reset_index(drop=True),
            user_ratings.iloc[test_indices].reset_index(drop=True),
        )

    return user_splits


def build_popularity_ranking(movies: pd.DataFrame) -> list[int]:
    """Rank the catalog by rating quality and log-scaled rating volume."""
    ranking = movies[["movieId", "rating_mean", "rating_count"]].copy()
    ranking["popularity_score"] = (
        ranking["rating_mean"].fillna(0.0)
        * np.log1p(ranking["rating_count"].fillna(0.0))
    )

    ranking = ranking.sort_values(
        ["popularity_score", "rating_mean", "rating_count", "movieId"],
        ascending=[False, False, False, True],
    )
    return ranking["movieId"].astype(int).tolist()


def recommend_popular(
    popularity_ranking: Iterable[int],
    seen_ids: set[int],
    k: int = TOP_K,
) -> list[int]:
    """Return the first K globally popular movies not seen in training."""
    recommendations: list[int] = []
    for movie_id in popularity_ranking:
        if movie_id not in seen_ids:
            recommendations.append(int(movie_id))
        if len(recommendations) == k:
            break

    return recommendations


def _invoke_hybrid(
    recommender: Any,
    user_ratings: dict[int, float],
    top_k: int,
    weights: dict[str, float],
) -> Any:
    """Call the current project recommender while tolerating common API names."""
    method = getattr(recommender, "recommend", None)
    if method is None:
        method = getattr(recommender, "recommend_for_user", None)
    if method is None:
        raise AttributeError("HybridRecommender nao possui um metodo de recomendacao compativel.")

    parameters = inspect.signature(method).parameters
    kwargs: dict[str, Any] = {}
    for size_parameter in ("top_k", "k", "n_recommendations"):
        if size_parameter in parameters:
            kwargs[size_parameter] = top_k
            break
    kwargs.update(
        {
            parameter: value
            for parameter, value in weights.items()
            if parameter in parameters
        }
    )

    if "user_ratings" in parameters:
        return method(user_ratings=user_ratings, **kwargs)
    return method(user_ratings, **kwargs)


def _extract_movie_ids(recommendations: Any) -> list[int]:
    if isinstance(recommendations, pd.DataFrame):
        if "movieId" not in recommendations.columns:
            raise ValueError("As recomendacoes hibridas nao possuem a coluna movieId.")
        values = recommendations["movieId"].tolist()
    elif isinstance(recommendations, pd.Series):
        values = recommendations.tolist()
    else:
        values = list(recommendations)

    movie_ids: list[int] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("movieId")
        if value is not None:
            movie_ids.append(int(value))
    return movie_ids


def recommend_hybrid(
    recommender: Any,
    train_ratings: pd.DataFrame,
    weights: dict[str, float],
    k: int = TOP_K,
) -> list[int]:
    """Generate hybrid recommendations and explicitly remove training items."""
    user_ratings = {
        int(row.movieId): float(row.rating)
        for row in train_ratings[["movieId", "rating"]].itertuples(index=False)
    }
    seen_ids = set(user_ratings)
    recommendations = _extract_movie_ids(
        _invoke_hybrid(recommender, user_ratings, k, weights)
    )
    unseen_recommendations = [movie_id for movie_id in recommendations if movie_id not in seen_ids]

    if len(unseen_recommendations) < k:
        expanded_k = min(k + len(seen_ids), len(recommender.movies))
        recommendations = _extract_movie_ids(
            _invoke_hybrid(recommender, user_ratings, expanded_k, weights)
        )
        unseen_recommendations = [
            movie_id for movie_id in recommendations if movie_id not in seen_ids
        ]

    return unseen_recommendations[:k]


def recommend_hybrid_configurations(
    recommender: Any,
    train_ratings: pd.DataFrame,
    k: int = TOP_K,
) -> dict[str, list[int]]:
    """Generate recommendations for every configured hybrid weight combination."""
    user_ratings = {
        int(row.movieId): float(row.rating)
        for row in train_ratings[["movieId", "rating"]].itertuples(index=False)
    }
    seen_ids = set(user_ratings)

    score_method = getattr(recommender, "score_movies", None)
    if score_method is None:
        return {
            name: recommend_hybrid(recommender, train_ratings, weights, k)
            for name, weights in HYBRID_CONFIGURATIONS.items()
        }

    scored_movies = score_method(user_ratings=user_ratings)
    component_columns = [
        "movieId",
        "profile_score",
        "rating_score",
        "popularity_score",
    ]
    if not set(component_columns).issubset(scored_movies.columns):
        return {
            name: recommend_hybrid(recommender, train_ratings, weights, k)
            for name, weights in HYBRID_CONFIGURATIONS.items()
        }

    unseen_scores = scored_movies.loc[
        ~scored_movies["movieId"].isin(seen_ids),
        list(component_columns),
    ].copy()
    recommendations: dict[str, list[int]] = {}

    for name, weights in HYBRID_CONFIGURATIONS.items():
        hybrid_score = (
            weights["profile_weight"] * unseen_scores["profile_score"]
            + weights["rating_weight"] * unseen_scores["rating_score"]
            + weights["popularity_weight"] * unseen_scores["popularity_score"]
        )
        recommendations[name] = (
            unseen_scores.assign(hybrid_score=hybrid_score)
            .sort_values("hybrid_score", ascending=False)
            .head(k)["movieId"]
            .astype(int)
            .tolist()
        )

    return recommendations


def _empty_metric_accumulator() -> dict[str, Any]:
    return {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "recommended_ids": [],
    }


def _add_user_metrics(
    accumulator: dict[str, Any],
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> None:
    accumulator["precision"].append(precision_at_k(recommended_ids, relevant_ids, k))
    accumulator["recall"].append(recall_at_k(recommended_ids, relevant_ids, k))
    accumulator["ndcg"].append(ndcg_at_k(recommended_ids, relevant_ids, k))
    accumulator["recommended_ids"].extend(recommended_ids)


def evaluate_recommenders(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    recommender: Any,
    k: int = TOP_K,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate popularity and hybrid recommendations on matching user splits."""
    eligible_ratings = select_eligible_ratings(ratings)
    user_splits = split_ratings_by_user(eligible_ratings, seed=seed)
    popularity_ranking = build_popularity_ranking(movies)
    accumulators = {
        "Popularity Baseline": _empty_metric_accumulator(),
        **{
            model_name: _empty_metric_accumulator()
            for model_name in HYBRID_CONFIGURATIONS
        },
    }
    evaluated_users = 0
    skipped_users = 0

    for train_ratings, test_ratings in user_splits.values():
        relevant_ids = set(
            test_ratings.loc[test_ratings["rating"] >= RELEVANT_RATING, "movieId"]
            .astype(int)
            .tolist()
        )
        if not relevant_ids:
            skipped_users += 1
            continue

        seen_ids = set(train_ratings["movieId"].astype(int))
        try:
            hybrid_recommendations = recommend_hybrid_configurations(
                recommender, train_ratings, k
            )
        except ValueError:
            # A profile cannot be built when all usable training ratings are neutral.
            skipped_users += 1
            continue

        popularity_ids = recommend_popular(popularity_ranking, seen_ids, k)
        _add_user_metrics(
            accumulators["Popularity Baseline"], popularity_ids, relevant_ids, k
        )
        for model_name, recommended_ids in hybrid_recommendations.items():
            _add_user_metrics(
                accumulators[model_name], recommended_ids, relevant_ids, k
            )
        evaluated_users += 1

    if evaluated_users == 0:
        raise ValueError("Nenhum usuario possui dados suficientes para a avaliacao.")

    rows = []
    for model_name, values in accumulators.items():
        rows.append(
            {
                "Model": model_name,
                f"Precision@{k}": float(np.mean(values["precision"])),
                f"Recall@{k}": float(np.mean(values["recall"])),
                f"NDCG@{k}": float(np.mean(values["ndcg"])),
                "Coverage": catalog_coverage(
                    values["recommended_ids"],
                    total_items=len(movies),
                ),
                "Evaluated Users": evaluated_users,
                "Skipped Users": skipped_users,
            }
        )

    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame, output_path: Path = RESULTS_PATH) -> None:
    """Persist evaluation results as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def print_best_hybrid_configurations(
    results: pd.DataFrame,
    k: int = TOP_K,
) -> None:
    """Print the best hybrid weight configuration for each reported metric."""
    hybrid_results = results[results["Model"].isin(HYBRID_CONFIGURATIONS)]
    metrics = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", "Coverage"]

    print("\nMelhores configuracoes hibridas:")
    for metric in metrics:
        best_row = hybrid_results.loc[hybrid_results[metric].idxmax()]
        print(f"{metric}: {best_row['Model']} ({best_row[metric]:.4f})")


def main() -> None:
    """Run the complete offline evaluation and print its summary."""
    print("Carregando dados processados...")
    movies, ratings = load_evaluation_data()

    print("Inicializando o recomendador hibrido...")
    recommender = HybridRecommender()

    print("Avaliando Popularity Baseline e configuracoes hibridas...")
    results = evaluate_recommenders(movies, ratings, recommender)
    save_results(results)

    display_results = results.copy()
    metric_columns = [f"Precision@{TOP_K}", f"Recall@{TOP_K}", f"NDCG@{TOP_K}", "Coverage"]
    display_results[metric_columns] = display_results[metric_columns].map(
        lambda value: f"{value:.4f}"
    )

    print("\nResultados da avaliacao:")
    print(display_results.to_string(index=False))
    print_best_hybrid_configurations(results)
    print(f"\nResultados salvos em: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
