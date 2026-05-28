

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MOVIES_PROCESSED_PATH = PROCESSED_DATA_DIR / "movies_processed.parquet"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX_PATH = MODELS_DIR / "tfidf_matrix.joblib"


def load_movies_processed(path: Path = MOVIES_PROCESSED_PATH) -> pd.DataFrame:
    """Load the processed movie dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {path}. Execute primeiro: python src/data/preprocessing.py"
        )

    return pd.read_parquet(path)


def validate_content_column(movies: pd.DataFrame, column: str = "content_text") -> None:
    """Validate that the processed dataset contains text features."""
    if column not in movies.columns:
        raise ValueError(f"Coluna obrigatoria ausente: {column}")

    if movies[column].fillna("").str.strip().eq("").all():
        raise ValueError(f"A coluna {column} esta vazia.")


def build_tfidf_features(
    movies: pd.DataFrame,
    text_column: str = "content_text",
    max_features: int = 10000,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[TfidfVectorizer, object]:
    """Build a TF-IDF vectorizer and matrix from movie content text."""
    validate_content_column(movies, text_column)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(movies[text_column].fillna(""))

    return vectorizer, tfidf_matrix


def save_tfidf_artifacts(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: object,
    models_dir: Path = MODELS_DIR,
) -> None:
    """Save TF-IDF artifacts for reuse by recommenders."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)


def main() -> None:
    """Run text feature extraction."""
    print("Carregando filmes processados...")
    movies = load_movies_processed()

    print("Criando matriz TF-IDF...")
    vectorizer, tfidf_matrix = build_tfidf_features(movies)

    print("Salvando artefatos TF-IDF...")
    save_tfidf_artifacts(vectorizer, tfidf_matrix)

    print("Features textuais concluidas.")
    print(f"Filmes vetorizados: {tfidf_matrix.shape[0]}")
    print(f"Features TF-IDF: {tfidf_matrix.shape[1]}")
    print(f"Vectorizer salvo em: {VECTORIZER_PATH}")
    print(f"Matriz TF-IDF salva em: {TFIDF_MATRIX_PATH}")


if __name__ == "__main__":
    main()
