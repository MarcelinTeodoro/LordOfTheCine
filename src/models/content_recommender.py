

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MOVIES_PROCESSED_PATH = PROCESSED_DATA_DIR / "movies_processed.parquet"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"


class ContentRecommender:
    """Recommend movies based on TF-IDF content similarity."""

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

    def find_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Find movies whose titles contain the query text."""
        query_normalized = query.strip().lower()
        matches = self.movies[self.movies["title"].str.lower().str.contains(query_normalized, na=False)]

        return matches[["movieId", "title", "genres", "rating_mean", "rating_count"]].head(limit)

    def recommend_by_movie_id(
        self,
        movie_id: int,
        top_k: int = 10,
        min_rating_count: int = 0,
    ) -> pd.DataFrame:
        """Return the top K movies most similar to a given movie ID."""
        if movie_id not in self.movie_id_to_index:
            raise ValueError(f"movieId nao encontrado: {movie_id}")

        movie_index = self.movie_id_to_index[movie_id]
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix,
        ).ravel()

        recommendations = self.movies.copy()
        recommendations["similarity_score"] = similarity_scores
        recommendations = recommendations[recommendations["movieId"] != movie_id]

        if min_rating_count > 0:
            recommendations = recommendations[recommendations["rating_count"] >= min_rating_count]

        columns = [
            "movieId",
            "title",
            "genres",
            "rating_mean",
            "rating_count",
            "similarity_score",
        ]

        return (
            recommendations.sort_values(["similarity_score", "rating_mean"], ascending=False)
            .head(top_k)[columns]
            .reset_index(drop=True)
        )

    def recommend_by_title(
        self,
        title: str,
        top_k: int = 10,
        min_rating_count: int = 0,
    ) -> pd.DataFrame:
        """Find the first matching title and return content-based recommendations."""
        matches = self.find_movies(title, limit=1)
        if matches.empty:
            raise ValueError(f"Nenhum filme encontrado para o titulo: {title}")

        movie_id = int(matches.iloc[0]["movieId"])
        return self.recommend_by_movie_id(movie_id, top_k=top_k, min_rating_count=min_rating_count)


def main() -> None:
    """Run a small smoke test for the content recommender."""
    recommender = ContentRecommender()
    query = "Toy Story"

    print(f"Buscando recomendacoes baseadas em: {query}")
    recommendations = recommender.recommend_by_title(query, top_k=10, min_rating_count=5)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
