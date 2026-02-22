import streamlit as st
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --- 1. SETUP & CACHING ---

@st.cache_resource
def load_model():
    # Load the AI model (only happens once)
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


@st.cache_data
def load_data():
    # Load the cleaned movie text data
    df = pd.read_csv("cleaned_movies.csv")
    return df


@st.cache_data
def load_embeddings():
    # Load the pre-calculated math vectors
    embeddings = np.load("movie_embeddings.npy")
    return embeddings


# --- 2. INITIALIZE APP ---
st.set_page_config(page_title="Semantic Movie Matcher", page_icon="🎬")

st.title("🎬 Semantic Movie Matcher")
st.write(
    "Describe a movie vaguely, and AI will find it based on *meaning*, not keywords. Searching through 5,000 movies instantly!")

# Load resources with a loading spinner
with st.spinner("Loading AI Brain and Movie Database..."):
    model = load_model()
    df = load_data()
    embeddings = load_embeddings()

# --- 3. THE SEARCH ENGINE ---
query = st.text_input("What kind of movie are you looking for?", "A giant shark attacks a beach town")

if st.button("Find Movie"):
    if query:
        with st.spinner("Searching vector space..."):
            # 1. Turn the user's query into a vector
            query_vec = model([query]).numpy()

            # 2. Compare query vector to all 5,000 movie vectors at once
            scores = cosine_similarity(query_vec, embeddings)[0]

            # 3. Get the indices of the top 5 highest scores
            top_indices = scores.argsort()[-5:][::-1]

            st.subheader("Top 5 Recommendations:")

            # 4. Display the results
            for index in top_indices:
                title = df.iloc[index]['Title']
                year = df.iloc[index]['Release Year']
                plot = df.iloc[index]['Plot']
                score = scores[index]

                # Create a neat dropdown for each movie
                with st.expander(f"**{title}** ({year}) - Match: {score:.0%}", expanded=True):
                    st.write(plot)
                    st.progress(float(score))
    else:
        st.warning("Please enter a description first.")

# --- 4. SIDEBAR EXPLANATION (For Recruiters) ---
st.sidebar.header("🧠 How it Works")
st.sidebar.markdown("""
This app demonstrates **Semantic Vector Search**.
1. 5,000 movie plots were pre-processed into 512-dimensional vectors using Google's **Universal Sentence Encoder**.
2. These vectors are stored as a highly optimized NumPy array.
3. When you search, your query is vectorized in real-time.
4. The app uses **Cosine Similarity** to find the mathematical closest match in milliseconds.
""")