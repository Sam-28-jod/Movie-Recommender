import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process
import re

st.title("Movie Recommender System!")
st.write("Welcome to Sambhav's movie recommender system!")

@st.cache_data(show_spinner=False)
def load_movies():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")

    avg_ratings = ratings.groupby("movieId")["rating"].mean()
    movies["rating"] = movies["movieId"].map(avg_ratings).round(1)

    movies["year"] = movies["title"].apply(
        lambda t: int(re.search(r"\((\d{4})\)", t).group(1)) if re.search(r"\((\d{4})\)", t) else pd.NA
    ).astype("Int64")

    return movies

@st.cache_data(show_spinner=False)
def compute_count_matrix(movies):
    cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
    count_matrix = cv.fit_transform(movies["genres"])
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    return count_matrix, indices

def get_recommendations(title, top_n=10):
    idx = indices[title]
    sim_scores = cosine_similarity(count_matrix[idx], count_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    return movies.iloc[sim_indices][["title", "year", "rating", "genres"]]

movies = load_movies()
count_matrix, indices = compute_count_matrix(movies)

with st.form("search_form"):
    movie_input = st.text_input("Type a movie:")
    submitted = st.form_submit_button("Get Recommendations")

    if submitted and movie_input:
        with st.spinner("Finding recommendations for you..."):
            matches = [m[0] for m in process.extractBests(movie_input, movies["title"].tolist(), limit=10)]
            if matches:
                movie_name = st.selectbox("Select a movie from top matches:", matches)
                if movie_name:
                    recs = get_recommendations(movie_name, top_n=10).copy()
                    recs["year"] = recs["year"].fillna(0).astype(int)
                    recs["rating"] = recs["rating"].fillna("No rating")
                    recs.reset_index(drop=True, inplace=True)
                    recs.index += 1
                    recs = recs.rename(columns={"title": "Title"})
                    st.success(f"Showing recommendations for: {movie_name}")
                    st.dataframe(recs, use_container_width=True)
            else:
                st.warning("No matches found. Try typing a different title.")
