# 🎬 Movie Recommender System

A hybrid movie recommendation engine combining **content-based filtering** and **collaborative filtering (SVD)**, built on the MovieLens 100K dataset and deployed as a live web application.

**🔗 Live Demo:** [https://huggingface.co/spaces/Philips20/movie-recommender](https://huggingface.co/spaces/Philips20/movie-recommender)

---

## 📖 Overview

This project implements a hybrid recommendation system that personalizes movie suggestions by combining two complementary approaches:

- **Content-Based Filtering** — recommends movies similar in genre to a movie the user likes, using TF-IDF vectorization and cosine similarity.
- **Collaborative Filtering** — predicts how a specific user would rate unseen movies using **SVD (Singular Value Decomposition)** trained on 100,000 historical ratings from 943 users.

The two approaches are blended into a single **hybrid score**, producing recommendations that are both genre-relevant and personalized to individual user taste.

---

## 🏗️ Architecture

```
┌─────────────────────┐
│  MovieLens 100K      │
│  (100K ratings,      │
│   1,682 movies,      │
│   943 users)         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐       ┌────────────────────────┐
│  Content-Based Model  │       │  Collaborative Model    │
│  TF-IDF on genres →   │       │  SVD (matrix             │
│  Cosine Similarity    │       │  factorization)          │
│  (1682 × 1682 matrix) │       │  RMSE: 0.9355            │
└──────────┬────────────┘       └───────────┬─────────────┘
           │                                 │
           └───────────────┬─────────────────┘
                            ▼
                ┌────────────────────────┐
                │   Hybrid Recommender     │
                │   Score = 0.4 × Content  │
                │   + 0.6 × Predicted      │
                │   Rating                 │
                └───────────┬──────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │   Streamlit Web App      │
                │   (Dockerized)           │
                └───────────┬──────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │  Hugging Face Spaces     │
                │  (Live Deployment)       │
                └────────────────────────┘
```

---

## ✨ Features

- 🔍 **Movie search** — pick from 1,682 movies in the MovieLens catalog
- 👤 **Personalized recommendations** — enter a User ID to get ratings tailored to that user's taste profile
- 📊 **Live model metrics** — RMSE, MAE, and dataset stats displayed in the sidebar
- ⚙️ **Adjustable recommendation count** — slider to get 5–20 recommendations
- 🐳 **Containerized deployment** — runs identically locally and in production via Docker

---

## 📈 Model Performance

| Metric | Value | Method |
|---|---|---|
| **RMSE** | **0.9355** | 5-fold cross-validation |
| **MAE** | 0.7372 | 5-fold cross-validation |
| Users | 943 | — |
| Movies | 1,682 | — |
| Ratings | 100,000 | — |
| Matrix Sparsity | 93.7% | — |

The SVD model was tuned with `n_factors=100`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.02`, and validated across 5 folds with consistent performance (RMSE std: 0.0024).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML / Data | Scikit-learn, Surprise (SVD), Pandas, NumPy |
| Web Framework | Streamlit |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Dataset | [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) |

---

## 🚀 How It Works

1. **Data Preprocessing** — Load and clean MovieLens 100K ratings and movie metadata (genres, titles).
2. **Content-Based Model** — Convert genre tags into TF-IDF vectors and compute pairwise cosine similarity across all 1,682 movies.
3. **Collaborative Filtering** — Train an SVD model on the user-item ratings matrix to learn latent factors representing user preferences and movie characteristics.
4. **Hybrid Scoring** — For a chosen "seed" movie, find genre-similar movies, then re-rank them using the user's predicted rating from the SVD model:

   ```
   hybrid_score = 0.4 × content_similarity + 0.6 × normalized_predicted_rating
   ```

5. **Web App** — A Streamlit interface lets users select a movie and User ID, then displays the top-N hybrid-ranked recommendations.

---

## 💻 Running Locally

```bash
# Clone the repository
git clone https://github.com/hana-20092006/movie-recommender.git
cd movie-recommender

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🐳 Running with Docker

```bash
docker build -t movie-recommender .
docker run -p 7860:7860 movie-recommender
```

The app will be available at `http://localhost:7860`.

---

## 📂 Project Structure

```
movie-recommender/
├── app.py                 # Streamlit application
├── Dockerfile             # Container configuration
├── requirements.txt       # Python dependencies
├── models/
│   ├── cosine_sim.pkl      # Precomputed content-similarity matrix
│   ├── movie_indices.pkl   # Movie title → index mapping
│   ├── movies_clean.csv    # Cleaned movie metadata
│   ├── svd_model.pkl       # Trained SVD collaborative filtering model
│   └── metrics.pkl         # Model evaluation metrics
└── README.md
```

---

## 🔮 Future Improvements

- Add poster images via TMDB API integration
- Incorporate implicit feedback (watch history, clicks)
- Experiment with deep learning approaches (Neural Collaborative Filtering)
- Add A/B testing framework for recommendation strategies
- Migrate to a scalable cloud deployment (AWS SageMaker / EC2) for production-grade serving

---

## 👤 Author

**Hana Maria Philip**
[GitHub](https://github.com/hana-20092006) • [LeetCode](https://leetcode.com/u/Hana_20092006)
