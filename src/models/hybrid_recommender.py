

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MOVIES_PROCESSED_PATH = PROCESSED_DATA_DIR / "movies_processed.parquet"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"


class HybridRecommender:
    """Recommend movies using user profile, rating quality, and popularity."""

    def __init__(
        self,
        movies_path: Path = MOVIES_PROCESSED_PATH,
        tfidf_matrix_path: Path = TFIDF_MATRIX_PATH,
    ) -> None:
        self.movies = self._load_movies(movies_path)
        self.tfidf_matrix = self._load_tfidf_matrix(tfidf_matrix_path)
        self.movie_id_to_index = {
            movie_id: index for index, movie_id in enumerate(self.movies["movieId"].tolist())
        }

    @staticmethod
    def _load_movies(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {path}. Execute primeiro: python src/data/preprocessing.py"
            )

        return pd.read_parquet(path).reset_index(drop=True)

    @staticmethod
    def _load_tfidf_matrix(path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {path}. Execute primeiro: python src/features/text_features.py"
            )

        return joblib.load(path)

    @staticmethod
    def _min_max_normalize(values: pd.Series) -> pd.Series:
        min_value = values.min()
        max_value = values.max()

        if max_value == min_value:
            return pd.Series(0.0, index=values.index)

        return (values - min_value) / (max_value - min_value)

    def build_user_profile(
        self,
        user_ratings: dict[int, float],
        neutral_rating: float = 3.0,
    ) -> csr_matrix:
        """Create a weighted user vector from explicit movie ratings."""
        valid_items = [
            (movie_id, rating)
            for movie_id, rating in user_ratings.items()
            if movie_id in self.movie_id_to_index and rating != neutral_rating
        ]
        if not valid_items:
            raise ValueError("Informe ao menos um movieId valido com rating diferente da nota neutra.")

        indices = [self.movie_id_to_index[movie_id] for movie_id, _ in valid_items]
        weights = [rating - neutral_rating for _, rating in valid_items]

        selected_vectors = self.tfidf_matrix[indices]
        profile_vector = selected_vectors.multiply(np.asarray(weights)[:, None]).sum(axis=0)

        return csr_matrix(profile_vector)

    def score_movies(
        self,
        user_ratings: dict[int, float],
        profile_weight: float = 0.70,
        rating_weight: float = 0.20,
        popularity_weight: float = 0.10,
    ) -> pd.DataFrame:
        """Calculate hybrid recommendation scores for all movies."""
        total_weight = profile_weight + rating_weight + popularity_weight
        if total_weight <= 0:
            raise ValueError("A soma dos pesos precisa ser maior que zero.")

        profile_weight = profile_weight / total_weight
        rating_weight = rating_weight / total_weight
        popularity_weight = popularity_weight / total_weight

        profile_vector = self.build_user_profile(user_ratings)
        profile_scores = cosine_similarity(profile_vector, self.tfidf_matrix).ravel()

        scored_movies = self.movies.copy()
        scored_movies["profile_score"] = profile_scores
        scored_movies["rating_score"] = self._min_max_normalize(scored_movies["rating_mean"].fillna(0.0))
        scored_movies["popularity_score"] = self._min_max_normalize(
            np.log1p(scored_movies["rating_count"].fillna(0))
        )
        scored_movies["hybrid_score"] = (
            profile_weight * scored_movies["profile_score"]
            + rating_weight * scored_movies["rating_score"]
            + popularity_weight * scored_movies["popularity_score"]
        )

        return scored_movies

    def recommend(
        self,
        user_ratings: dict[int, float],
        top_k: int = 10,
        min_rating_count: int = 0,
        profile_weight: float = 0.70,
        rating_weight: float = 0.20,
        popularity_weight: float = 0.10,
    ) -> pd.DataFrame:
        """Return hybrid recommendations for unseen movies."""
        recommendations = self.score_movies(
            user_ratings=user_ratings,
            profile_weight=profile_weight,
            rating_weight=rating_weight,
            popularity_weight=popularity_weight,
        )
        recommendations = recommendations[~recommendations["movieId"].isin(user_ratings.keys())]

        if min_rating_count > 0:
            recommendations = recommendations[recommendations["rating_count"] >= min_rating_count]

        columns = [
            "movieId",
            "title",
            "genres",
            "rating_mean",
            "rating_count",
            "profile_score",
            "rating_score",
            "popularity_score",
            "hybrid_score",
        ]

        return (
            recommendations.sort_values("hybrid_score", ascending=False)
            .head(top_k)[columns]
            .reset_index(drop=True)
        )


def main() -> None:
    """Run a small smoke test for hybrid recommendations."""
    recommender = HybridRecommender()
    user_ratings = {
        1: 5.0,     # Toy Story
        3114: 4.5,  # Toy Story 2
        260: 5.0,   # Star Wars
        296: 2.0,   # Pulp Fiction
    }

    print("Gerando recomendacoes hibridas...")
    recommendations = recommender.recommend(user_ratings, top_k=10, min_rating_count=20)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
