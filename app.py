import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Streamlit decorator = Create this resource once, then reuse it instead of recreating it every time the app reruns
@st.cache_resource
def load_artifacts():
    with open('models/cosine_sim.pkl','rb') as f:
        cosine_sim = pickle.load(f)
    with open('models/movie_indices.pkl','rb') as f:
        movie_indices = pickle.load(f)
    with open('models/svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    with open('models/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    movies = pd.read_csv('models/movies_clean.csv')
    return cosine_sim, movie_indices, svd_model, metrics, movies

cosine_sim, movie_indices, svd_model, metrics, movies = load_artifacts()

# Sidebar 
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("RMSE (5-fold CV)", metrics['rmse'])
    st.metric("MAE", metrics['mae'])
    st.metric("Users", metrics['n_users'])
    st.metric("Movies", metrics['n_items'])
    
    st.divider()
    st.subheader("How it works")
    st.write("""
    This recommender combines two approaches:
    
    **1. Content-Based Filtering**  
    Uses TF-IDF on movie genres + cosine similarity to find similar movies.
    
    **2. Collaborative Filtering (SVD)**  
    Learns user preferences from 100,000 ratings to predict how you'd rate each movie.
    
    **Hybrid Score** = 0.4 × Content Similarity + 0.6 × Predicted Rating
    """)
    
    st.divider()
    st.caption("Built with MovieLens 100K dataset")
    st.caption("Dataset: 943 users, 1,682 movies, 100K ratings")

st.title("🎬 Movie Recommender System")
st.write("Hybrid recommendation engine — Content-based + Collaborative Filtering (SVD)")
st.write("Artifacts loaded successfully!")
st.write(f"Movies in database: {len(movies)}")
st.write(f"Model RMSE: {metrics['rmse']}")

# --- Hybrid recommendation function ---
def hybrid_recommend(user_id, title, n=10, content_weight=0.4):
    if title not in movie_indices:
        return None

    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    results = []
    for movie_idx, content_sim in sim_scores:
        if movie_idx == idx:
            continue

        movie_id = movies.iloc[movie_idx]['movie_id']
        movie_title = movies.iloc[movie_idx]['title']

        pred_rating = svd_model.predict(uid=user_id, iid=movie_id).est
        norm_rating = (pred_rating - 1) / 4

        hybrid_score = content_weight * content_sim + (1 - content_weight) * norm_rating

        results.append({
            'Title': movie_title,
            'Predicted Rating': round(pred_rating, 2),
            'Content Similarity': round(content_sim, 3),
            'Hybrid Score': round(hybrid_score, 3)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('Hybrid Score', ascending=False).head(n)
    return result_df.reset_index(drop=True)


# --- UI ---
st.divider()
st.subheader("Get Recommendations")

col1, col2 = st.columns(2)

with col1:
    selected_movie = st.selectbox(
        "Pick a movie you like:",
        options=sorted(movie_indices.index.tolist())
    )

with col2:
    user_id = st.number_input(
        "Enter your User ID (1–943):",
        min_value=1, max_value=943, value=1, step=1
    )

n_recs = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Get Recommendations", type="primary"):
    with st.spinner("Finding movies for you..."):
        recs = hybrid_recommend(user_id=user_id, title=selected_movie, n=n_recs)
    
    if recs is not None:
        st.success(f"Top {n_recs} recommendations based on '{selected_movie}' for User {user_id}")
        st.dataframe(recs, use_container_width=True, hide_index=True)
    else:
        st.error("Movie not found.")