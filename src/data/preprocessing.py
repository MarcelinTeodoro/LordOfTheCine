"""Preprocessing routines for MovieLens metadata and ratings."""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
NO_GENRES_LABEL = "(no genres listed)"


def load_raw_data(
    raw_data_dir: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens raw CSV files."""
    movies = pd.read_csv(raw_data_dir / "movies.csv")
    ratings = pd.read_csv(raw_data_dir / "ratings.csv")
    tags = pd.read_csv(raw_data_dir / "tags.csv")
    links = pd.read_csv(raw_data_dir / "links.csv")

    return movies, ratings, tags, links


def normalize_text(text: str) -> str:
    """Normalize text for content-based feature extraction."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Create useful movie metadata columns."""
    movies_processed = movies.copy()

    movies_processed["genres"] = movies_processed["genres"].fillna(NO_GENRES_LABEL)
    movies_processed["genres_list"] = movies_processed["genres"].apply(
        lambda genres: [] if genres == NO_GENRES_LABEL else genres.split("|")
    )
    movies_processed["genres_text"] = movies_processed["genres_list"].apply(lambda genres: " ".join(genres))
    movies_processed["year"] = movies_processed["title"].str.extract(r"\((\d{4})\)\s*$").astype("float")
    movies_processed["title_clean"] = (
        movies_processed["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    )

    return movies_processed


def preprocess_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Normalize tags and aggregate them by movie."""
    tags_processed = tags.copy()
    tags_processed["tag"] = (
        tags_processed["tag"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )

    return (
        tags_processed[tags_processed["tag"] != ""]
        .groupby("movieId")["tag"]
        .apply(lambda values: " ".join(sorted(set(values))))
        .reset_index(name="tags_text")
    )


def build_rating_statistics(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build movie-level rating statistics and return normalized ratings."""
    ratings_processed = ratings.copy()
    ratings_processed["rating"] = ratings_processed["rating"].astype(float)

    rating_stats = (
        ratings_processed.groupby("movieId")
        .agg(
            rating_mean=("rating", "mean"),
            rating_count=("rating", "count"),
            rating_std=("rating", "std"),
        )
        .reset_index()
    )
    rating_stats["rating_std"] = rating_stats["rating_std"].fillna(0.0)

    return ratings_processed, rating_stats


def build_movies_dataset(
    movies: pd.DataFrame,
    tags_by_movie: pd.DataFrame,
    rating_stats: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:
    """Combine movie metadata, tags, rating statistics, and external IDs."""
    movies_processed = movies.merge(tags_by_movie, on="movieId", how="left")
    movies_processed = movies_processed.merge(rating_stats, on="movieId", how="left")
    movies_processed = movies_processed.merge(links, on="movieId", how="left")

    movies_processed["tags_text"] = movies_processed["tags_text"].fillna("")
    movies_processed["rating_count"] = movies_processed["rating_count"].fillna(0).astype(int)
    movies_processed["rating_mean"] = movies_processed["rating_mean"].fillna(0.0)
    movies_processed["rating_std"] = movies_processed["rating_std"].fillna(0.0)
    content_text = (
        movies_processed["title_clean"].fillna("")
        + " "
        + movies_processed["genres_text"].fillna("")
        + " "
        + movies_processed["tags_text"].fillna("")
    )
    movies_processed["content_text"] = content_text.apply(normalize_text)

    return movies_processed


def save_processed_data(
    movies_processed: pd.DataFrame,
    ratings_processed: pd.DataFrame,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    """Save processed datasets."""
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    movies_processed.to_parquet(processed_data_dir / "movies_processed.parquet", index=False)
    ratings_processed.to_parquet(processed_data_dir / "ratings_processed.parquet", index=False)
    movies_processed.to_csv(processed_data_dir / "movies_processed.csv", index=False)
    ratings_processed.to_csv(processed_data_dir / "ratings_processed.csv", index=False)


def main() -> None:
    """Run the MovieLens preprocessing pipeline."""
    print("Carregando dados brutos...")
    movies, ratings, tags, links = load_raw_data()

    print("Pre-processando filmes...")
    movies_metadata = preprocess_movies(movies)

    print("Pre-processando tags...")
    tags_by_movie = preprocess_tags(tags)

    print("Calculando estatisticas de ratings...")
    ratings_processed, rating_stats = build_rating_statistics(ratings)

    print("Construindo dataset final de filmes...")
    movies_processed = build_movies_dataset(movies_metadata, tags_by_movie, rating_stats, links)

    print("Salvando arquivos processados...")
    save_processed_data(movies_processed, ratings_processed)

    print("Pre-processamento concluido.")
    print(f"Filmes processados: {len(movies_processed)}")
    print(f"Ratings processados: {len(ratings_processed)}")


if __name__ == "__main__":
    main()
