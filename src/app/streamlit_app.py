

"""Streamlit entrypoint for Lord of the Cine."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.explanations import add_explanations
from src.models.hybrid_recommender import HybridRecommender


st.set_page_config(
    page_title="Lord of the Cine",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }

    .movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .movie-meta {
        color: #5f6673;
        font-size: 0.9rem;
        margin-bottom: 0.45rem;
    }

    .score-line {
        font-size: 0.86rem;
        color: #333947;
        margin-top: 0.35rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_recommender() -> HybridRecommender:
    """Load the hybrid recommender once per Streamlit session."""
    return HybridRecommender()


def initialize_state() -> None:
    """Initialize user preference state."""
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}


def format_movie_option(row: pd.Series) -> str:
    """Format movie options for selectboxes."""
    year = row.get("year")
    year_text = "" if pd.isna(year) else f" ({int(year)})"

    return f"{row['title_clean']}{year_text} | id {row['movieId']}"


def get_movie_options(movies: pd.DataFrame, query: str, limit: int = 80) -> pd.DataFrame:
    """Return movie options filtered by title query."""
    options = movies.copy()
    query = query.strip().lower()

    if query:
        options = options[options["title"].str.lower().str.contains(query, na=False)]

    return (
        options.sort_values(["rating_count", "rating_mean"], ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def add_selected_rating(movie_id: int, rating: float) -> None:
    """Add or update a movie rating in session state."""
    st.session_state.user_ratings[int(movie_id)] = float(rating)


def remove_selected_rating(movie_id: int) -> None:
    """Remove a movie rating from session state."""
    st.session_state.user_ratings.pop(int(movie_id), None)


def build_profile_dataframe(movies: pd.DataFrame, user_ratings: dict[int, float]) -> pd.DataFrame:
    """Create a readable dataframe from selected user ratings."""
    if not user_ratings:
        return pd.DataFrame(columns=["movieId", "title", "rating"])

    profile = movies[movies["movieId"].isin(user_ratings.keys())][["movieId", "title", "genres"]].copy()
    profile["rating"] = profile["movieId"].map(user_ratings)

    return profile.sort_values("rating", ascending=False).reset_index(drop=True)


def render_recommendation(row: pd.Series) -> None:
    """Render one recommendation result."""
    with st.container(border=True):
        st.markdown(f"<div class='movie-title'>{row['title']}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='movie-meta'>{row['genres']} | media {row['rating_mean']:.2f} | "
            f"{int(row['rating_count'])} avaliacoes</div>",
            unsafe_allow_html=True,
        )
        st.write(row["explanation"])
        st.markdown(
            "<div class='score-line'>"
            f"hybrid {row['hybrid_score']:.3f} | perfil {row['profile_score']:.3f} | "
            f"nota {row['rating_score']:.3f} | popularidade {row['popularity_score']:.3f}"
            "</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the Lord of the Cine app."""
    initialize_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    recommender = load_recommender()
    movies = recommender.movies

    st.title("Lord of the Cine")

    with st.sidebar:
        st.header("Perfil")
        query = st.text_input("Buscar filme", value="")
        options = get_movie_options(movies, query)

        if options.empty:
            st.warning("Nenhum filme encontrado.")
        else:
            labels = [format_movie_option(row) for _, row in options.iterrows()]
            selected_label = st.selectbox("Filme", labels)
            selected_index = labels.index(selected_label)
            selected_movie = options.iloc[selected_index]
            selected_rating = st.slider("Nota", min_value=0.5, max_value=5.0, value=4.0, step=0.5)

            if st.button("Adicionar", type="primary", use_container_width=True):
                add_selected_rating(int(selected_movie["movieId"]), selected_rating)
                st.rerun()

        st.divider()
        st.header("Pesos")
        profile_weight = st.slider("Perfil", 0.0, 1.0, 0.70, 0.05)
        rating_weight = st.slider("Nota media", 0.0, 1.0, 0.20, 0.05)
        popularity_weight = st.slider("Popularidade", 0.0, 1.0, 0.10, 0.05)

        st.divider()
        st.header("Filtros")
        top_k = st.slider("Resultados", 5, 30, 10, 1)
        min_rating_count = st.slider("Minimo de avaliacoes", 0, 200, 20, 5)

        if st.button("Limpar perfil", use_container_width=True):
            st.session_state.user_ratings = {}
            st.rerun()

    profile_df = build_profile_dataframe(movies, st.session_state.user_ratings)

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Filmes no catalogo", f"{len(movies):,}".replace(",", "."))
    metric_col_2.metric("Filmes no perfil", len(st.session_state.user_ratings))
    metric_col_3.metric("Media minima", min_rating_count)

    left_col, right_col = st.columns([0.95, 1.65], gap="large")

    with left_col:
        st.subheader("Preferencias")
        if profile_df.empty:
            st.info("Adicione filmes avaliados para gerar recomendacoes.")
        else:
            for _, row in profile_df.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    st.caption(row["genres"])
                    st.write(f"Nota: {row['rating']:.1f}")
                    if st.button("Remover", key=f"remove-{row['movieId']}", use_container_width=True):
                        remove_selected_rating(int(row["movieId"]))
                        st.rerun()

    with right_col:
        st.subheader("Recomendacoes")
        if not st.session_state.user_ratings:
            st.stop()

        try:
            recommendations = recommender.recommend(
                user_ratings=st.session_state.user_ratings,
                top_k=top_k,
                min_rating_count=min_rating_count,
                profile_weight=profile_weight,
                rating_weight=rating_weight,
                popularity_weight=popularity_weight,
            )
            recommendations = add_explanations(
                recommendations,
                st.session_state.user_ratings,
                recommender.movies,
            )
        except ValueError as error:
            st.warning(str(error))
            st.stop()

        if recommendations.empty:
            st.info("Nenhuma recomendacao encontrada com os filtros atuais.")
            st.stop()

        for _, row in recommendations.iterrows():
            render_recommendation(row)


if __name__ == "__main__":
    main()
