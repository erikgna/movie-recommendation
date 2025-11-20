import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# -------------------------------------------------------------------------

def load_data():
    """
    Loads the datasets from CSV files.
    """
    print("Loading data...")
    # Load Movies: movieId, title, genres
    movies = pd.read_csv('data/movies.csv')
    
    # Load Ratings: userId, movieId, rating, timestamp
    ratings = pd.read_csv('data/ratings.csv')
    
    # Merge them so we have titles attached to ratings
    # Result: userId, movieId, rating, title, genres
    data = pd.merge(ratings, movies, on='movieId')
    
    print(f"✓ Loaded {len(ratings)} ratings and {len(movies)} movies.")
    return movies, ratings, data

# -------------------------------------------------------------------------
# 2. COLLABORATIVE FILTERING (Item-Based)
# Logic: "People who liked X also liked Y"
# -------------------------------------------------------------------------

def get_collaborative_similarity(ratings_df):
    """
    Creates a correlation matrix between movies based on user ratings.
    
    Returns:
        item_correlation_df: A DataFrame where index/columns are movie titles,
                             and values are the correlation (similarity) between them.
    """
    # 1. Create a Pivot Table (Matrix)
    # Rows = Users, Columns = Movie Titles, Values = Ratings
    user_movie_matrix = ratings_df.pivot_table(index='userId', columns='title', values='rating')
    
    # 2. Handle sparse data
    # We assume if a user hasn't rated a movie, it's neutral or unknown. 
    # For correlation, we usually leave NaNs or fill with 0. 
    # Here, we use Pearson correlation which handles NaNs naturally by ignoring pairs 
    # that don't share a user.
    
    # NOTE: In a real production system with thousands of movies, 
    # we would filter to keep only movies with > 50 ratings to speed this up.
    min_ratings_threshold = 2
    ratings_count = ratings_df.groupby('title')['rating'].count()
    popular_movies = ratings_count[ratings_count >= min_ratings_threshold].index
    
    # Filter matrix to only popular movies
    user_movie_matrix = user_movie_matrix[popular_movies]
    
    print("Calculating collaborative correlation matrix (this might take a moment)...")
    # 3. Calculate Correlation (Pearson)
    # This tells us: If rating for Movie A goes up, does rating for Movie B go up?
    item_correlation_df = user_movie_matrix.corr(method='pearson')
    
    # Fill NaNs (caused by no overlapping users) with 0
    item_correlation_df = item_correlation_df.fillna(0)
    
    return item_correlation_df

# -------------------------------------------------------------------------
# 3. CONTENT-BASED FILTERING
# Logic: "Movies with similar Genres"
# -------------------------------------------------------------------------

def get_content_similarity(movies_df):
    """
    Calculates similarity between movies based on their Genres using TF-IDF.
    
    Returns:
        cosine_sim: A matrix of similarity scores (0 to 1).
        indices: A dictionary mapping movie titles to their matrix index.
    """
    # 1. Clean the Genres format
    # Change "Adventure|Children|Fantasy" -> "Adventure Children Fantasy"
    movies_df['genres_clean'] = movies_df['genres'].str.replace('|', ' ')
    
    # 2. Vectorize the text (Turn genres into numbers)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_clean'])
    
    # 3. Calculate Cosine Similarity
    # How similar is the genre vector of Movie A to Movie B?
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create a reverse map of indices {Title: Index} for easy lookup
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    
    return cosine_sim, indices

# -------------------------------------------------------------------------
# 4. HYBRID RECOMMENDATION SYSTEM
# -------------------------------------------------------------------------

def get_recommendations(user_id, ratings, movies, item_corr_df, content_sim, indices):
    """
    Generates recommendations for a specific user.
    """
    # Step 1: Get movies the user has already rated highly (> 4.0)
    user_ratings = ratings[ratings['userId'] == user_id]
    liked_movies = user_ratings[user_ratings['rating'] >= 4.0]['title'].tolist()
    
    if not liked_movies:
        return ["User has no high-rated movies to base recommendations on."]
    
    print(f"\nUser {user_id} likes: {liked_movies[:3]}...")

    # Dictionary to store potential candidates: {movie_title: total_score}
    candidate_scores = {}
    
    # Step 2: For each movie the user likes, find similar movies
    for movie in liked_movies:
        
        # --- A. Get Collaborative Candidates (from correlation matrix) ---
        if movie in item_corr_df.columns:
            # Get movies correlated with this liked movie
            similar_movies_cf = item_corr_df[movie].dropna()
            # Scale scores by how much the user liked the original movie (optional, here simplified)
            for similar_movie, score in similar_movies_cf.items():
                if score > 0.1: # Filter out weak correlations
                    candidate_scores[similar_movie] = candidate_scores.get(similar_movie, 0) + (score * 1.5) # Weight CF higher

        # --- B. Get Content Candidates (from genres) ---
        if movie in indices:
            idx = indices[movie]
            sim_scores = list(enumerate(content_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Top 10 similar by genre
            for i, score in sim_scores[1:11]:
                similar_movie_title = movies.iloc[i]['title']
                # Add to total score (weight content slightly lower)
                candidate_scores[similar_movie_title] = candidate_scores.get(similar_movie_title, 0) + (score * 1.0)

    # Step 3: Filter and Sort Results
    # Remove movies the user has already seen
    seen_movies = user_ratings['title'].tolist()
    recommendations = []
    
    for movie, score in candidate_scores.items():
        if movie not in seen_movies:
            recommendations.append((movie, score))
            
    # Sort by score descending
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    
    # Return top 10
    return [rec[0] for rec in recommendations[:10]]

# -------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -------------------------------------------------------------------------

def main():
    # 1. Load Data
    movies, ratings, data = load_data()
    
    # 2. Build Models
    print("Building Collaborative Model (Correlations)...")
    item_corr_df = get_collaborative_similarity(data)
    
    print("Building Content Model (Genres)...")
    content_sim, indices = get_content_similarity(movies)
    
    # 3. Get Recommendations for a User
    test_user_id = 1
    print(f"{'='*50}")
    print(f"Generating Recommendations for User ID: {test_user_id}")
    print(f"{'='*50}")
    
    # FIX: Pass 'data' (merged df with titles) instead of 'ratings' (raw df without titles)
    recs = get_recommendations(test_user_id, data, movies, item_corr_df, content_sim, indices)
    
    print(f"\nTop Recommended Movies:")
    for i, movie in enumerate(recs, 1):
        print(f"{i}. {movie}")

if __name__ == '__main__':
    main()