# 🎬 Semantic Movie Matcher: High-Performance Vector Search

### 🚀 [Live Demo: Try the App Here](https://semantic-movie-matcher-dfvez9gnezxjyju95zvasq.streamlit.app/)

## 📖 Overview
This project is an AI-powered search engine that uses **Natural Language Processing (NLP)** to find movies based on their "vibe" or plot meaning rather than simple keyword matching. By mapping 5,000+ movie plots into a 512-dimensional vector space, the system can understand that a query for *"outer space battle"* is semantically related to *"intergalactic warfare"* even if the words don't match exactly.

## 💡 The Problem
Traditional search engines rely on **keyword matching**. If a user searches for *"scary movie in space"*, a standard database might fail if the movie description doesn't explicitly contain the words "scary" or "space."

## 🚀 The Solution
I built a **Semantic Search Engine** that understands the *meaning* and *context* of a user's query, not just the keywords. 

Using **Deep Learning (NLP)**, the app vectorizes user input and compares it against thousands of movie plot summaries to find the mathematically closest match in a high-dimensional vector space.

**Example:**
* **Query:** *"A robot protects a boy from the future"*
* **Result:** *"Terminator 2"* (Even if the specific keywords don't match exactly).

## 🛠️ The Tech Stack
* **AI Model:** Google’s **Universal Sentence Encoder (USE)** (Deep Averaging Network)
* **Logic:** Python, NumPy, Scikit-Learn (Cosine Similarity)
* **Machine Learning:** TensorFlow, TensorFlow Hub
* **Data Handling:** Pandas (ETL Pipeline)
* **Frontend:** Streamlit
* **Deployment:** Streamlit Cloud

## 🧠 Key Machine Learning Features

### 1. Semantic Vector Search
Unlike traditional SQL `LIKE` queries, this app uses **dense vector embeddings**. Every movie is represented as a mathematical coordinate. We calculate the **Cosine Similarity** between the user's query and the database to find the closest matches in milliseconds.

### 2. Performance Optimization (Pre-computation)
To ensure the app scales, I implemented a pre-processing pipeline:
* **Step 1:** Cleaned a raw Wikipedia dataset of 34,000+ movies down to a curated sample.
* **Step 2:** Pre-calculated embeddings for all plots and saved them as a `.npy` file.
* **Step 3:** The live app loads these pre-computed vectors, bypassing the need for heavy inference during the user's search session.

### 3. Production Problem Solving (The "Dependency Hell" Fix)
During deployment, I resolved a critical environment conflict involving **Python 3.13** and **setuptools v70+**. By pinning dependencies (`setuptools<70.0.0`) and managing the Python runtime environment, I ensured cross-platform stability—a key requirement for production ML systems.

## Screenshot
<img width="917" height="387" alt="image" src="https://github.com/user-attachments/assets/993a62f2-bb10-4108-b5b4-9a8380382a7b" />


## 📂 Project Structure
```text
├── app.py                # Main Streamlit application
├── requirements.txt      # Pinned dependencies for production
├── cleaned_movies.csv    # Processed movie metadata (Title, Year, Plot)
├── movie_embeddings.npy  # Pre-computed 512-D math vectors
└── assets/               # Screenshots and UI elements
```

## 🧠 How It Works
1.  **Vectorization:** The app uses the **Universal Sentence Encoder** to convert every movie plot into a **512-dimensional vector** (embedding).
2.  **Querying:** When a user types a description, that text is also converted into a 512-dimensional vector.
3.  **Similarity Search:** The app calculates the **Cosine Similarity** between the query vector and every movie vector in the database.
4.  **Ranking:** It returns the movies with the highest similarity scores (closest distance in vector space).

## 📦 How to Run Locally

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/43v3rn88b/Semantic-Movie-Matcher.git
    cd semantic-movie-matcher
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
