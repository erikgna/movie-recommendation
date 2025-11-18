import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# -------------------------
# Custom Collaborative Filtering using Matrix Factorization SGD
# -------------------------

class MatrixFactorizationSGD:
    """
    Matrix Factorization model using Stochastic Gradient Descent (SGD).
    """
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.min_rating = 1.0
        self.max_rating = 5.0

    def fit(self, ratings_df):
        """
        Trains the MF model on the provided ratings DataFrame.
        """
        print(f"Starting Matrix Factorization (SGD) training for {self.n_epochs} epochs...")
        
        # Map external IDs to internal indices
        self.unique_users = ratings_df['user_id'].unique()
        self.unique_items = ratings_df['item_id'].unique()
        self.user_to_idx = {user: i for i, user in enumerate(self.unique_users)}
        self.item_to_idx = {item: i for i, item in enumerate(self.unique_items)}
        self.n_users = len(self.unique_users)
        self.n_items = len(self.unique_items)

        # Initialize parameters
        self.mu = ratings_df['rating'].mean()
        self.bu = np.zeros(self.n_users, dtype=float)
        self.bi = np.zeros(self.n_items, dtype=float)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # Prepare training data
        train_data = []
        for _, row in ratings_df.iterrows():
            u = self.user_to_idx[row['user_id']]
            i = self.item_to_idx[row['item_id']]
            r = row['rating']
            train_data.append((u, i, r))

        # SGD training loop
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)
            total_error = 0.0

            for u, i, r_ui in train_data:
                r_hat_ui = self.mu + self.bu[u] + self.bi[i] + np.dot(self.P[u, :], self.Q[i, :])
                e_ui = r_ui - r_hat_ui
                total_error += e_ui**2

                # Update biases and latent factors
                self.bu[u] += self.lr * (e_ui - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e_ui - self.reg * self.bi[i])

                Pu_old = self.P[u, :].copy()
                Qi_old = self.Q[i, :].copy()
                self.P[u, :] += self.lr * (e_ui * Qi_old - self.reg * Pu_old)
                self.Q[i, :] += self.lr * (e_ui * Pu_old - self.reg * Qi_old)

            train_rmse = np.sqrt(total_error / len(train_data))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs} - Training RMSE: {train_rmse:.4f}')

        return train_rmse

    def predict(self, user_id, item_id):
        """
        Predicts the rating for a given user-item pair.
        """
        prediction = self.mu

        if user_id in self.user_to_idx:
            u = self.user_to_idx[user_id]
            prediction += self.bu[u]

        if item_id in self.item_to_idx:
            i = self.item_to_idx[item_id]
            prediction += self.bi[i]

        if user_id in self.user_to_idx and item_id in self.item_to_idx:
            prediction += np.dot(self.P[u, :], self.Q[i, :])

        return np.clip(prediction, self.min_rating, self.max_rating)

# -------------------------
# Data loading utilities
# -------------------------

def load_movielens_100k(path='data/ml-100k'):
    """
    Load MovieLens 100k data from CSV files with headers.
    Expects:
    - u.data.csv: userId,movieId,rating,timestamp
    - u.item.csv: movieId,title,genres (genres are pipe-separated)
    """
    ratings_path = os.path.join(path, 'u.data.csv')
    items_path = os.path.join(path, 'u.item.csv')
    
    # Load ratings with header
    try:
        ratings = pd.read_csv(ratings_path, encoding='latin-1')
        # Rename columns to match expected format
        ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        print(f"✓ Loaded {len(ratings)} ratings")
    except Exception as e:
        print(f"Error loading ratings: {e}")
        raise

    # Load items with header
    try:
        items = pd.read_csv(items_path, encoding='latin-1')
        # Rename columns to match expected format
        items.columns = ['item_id', 'title', 'genres']
        
        # Convert pipe-separated genres to space-separated for TF-IDF
        # e.g., "Action|Comedy|Drama" -> "Action Comedy Drama"
        items['genres'] = items['genres'].fillna('').str.replace('|', ' ')
        
        print(f"✓ Loaded {len(items)} movies")
        
    except Exception as e:
        print(f"Error loading items: {e}")
        raise

    return ratings, items

# -------------------------
# Model training functions
# -------------------------

def train_cf_model(ratings_df, n_factors=50, n_epochs=20, random_state=42):
    """Train collaborative filtering model."""
    algo = MatrixFactorizationSGD(
        n_factors=n_factors, 
        n_epochs=n_epochs, 
        lr=0.005,
        reg=0.02,
        random_state=random_state
    )
    train_rmse = algo.fit(ratings_df)
    return algo, train_rmse

def build_content_model(movies_df):
    """Build content-based model using TF-IDF."""
    movies = movies_df.copy()
    movies['content'] = (movies['title'].fillna('') + ' ' + movies['genres'].fillna(''))

    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    return movies.reset_index(drop=True), tfidf, tfidf_matrix

def get_similar_items(item_id, movies, tfidf_matrix, top_n=10):
    """Get top-n similar items to a given item based on content."""
    idx = movies.index[movies['item_id'] == item_id]
    if len(idx) == 0:
        return []
    idx = idx[0]
    
    cosine_similarities = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    related_idx = cosine_similarities.argsort()[::-1]
    related_idx = related_idx[related_idx != idx]
    top_idx = related_idx[:top_n]
    
    return movies.iloc[top_idx][['item_id', 'title', 'genres']].to_dict('records')

# -------------------------
# Hybrid recommender
# -------------------------

def recommend_hybrid(user_id, cf_algo, ratings_df, movies, tfidf_matrix, top_k=10, alpha=0.7, user_history_top_n=10):
    """
    Generate hybrid recommendations combining collaborative and content-based filtering.
    
    Args:
        user_id: User ID to generate recommendations for
        cf_algo: Trained collaborative filtering model
        ratings_df: DataFrame with user ratings
        movies: DataFrame with movie information
        tfidf_matrix: TF-IDF matrix for content similarity
        top_k: Number of recommendations to return
        alpha: Weight for CF score (1-alpha for content score)
        user_history_top_n: Number of top-rated items to use for user profile
    """
    all_items = movies['item_id'].unique()
    seen = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'].unique())

    # Build user profile from top-rated items
    user_ratings = ratings_df[ratings_df['user_id'] == user_id].sort_values('rating', ascending=False)
    top_hist = user_ratings['item_id'].head(user_history_top_n).tolist()

    # Compute content similarity scores
    id_to_idx = {iid: idx for idx, iid in enumerate(movies['item_id'])}
    content_scores = {}

    if len(top_hist) == 0 or not any(iid in id_to_idx for iid in top_hist):
        content_scores = {iid: 0.0 for iid in all_items}
    else:
        hist_idxs = [id_to_idx[iid] for iid in top_hist if iid in id_to_idx]
        hist_matrix = tfidf_matrix[hist_idxs]
        similarities = hist_matrix.dot(tfidf_matrix.T).toarray()
        max_sim = similarities.max(axis=0)
        
        for idx, iid in enumerate(movies['item_id']):
            content_scores[iid] = float(max_sim[idx])

    # Normalize content scores
    vals = np.array(list(content_scores.values()))
    if vals.max() - vals.min() > 1e-9:
        norm_content = {iid: (score - vals.min()) / (vals.max() - vals.min()) 
                       for iid, score in content_scores.items()}
    else:
        norm_content = {iid: 0.0 for iid in content_scores.keys()}

    # Get CF predictions for unseen items
    cf_preds = {}
    for iid in all_items:
        if iid not in seen:
            pred_est = cf_algo.predict(user_id, iid)
            cf_preds[iid] = pred_est

    if not cf_preds:
        return []

    # Normalize CF predictions
    cf_vals = np.array(list(cf_preds.values()))
    cf_min, cf_max = cf_vals.min(), cf_vals.max()
    if cf_max - cf_min > 1e-9:
        norm_cf = {iid: (score - cf_min) / (cf_max - cf_min) for iid, score in cf_preds.items()}
    else:
        norm_cf = {iid: 0.0 for iid in cf_preds.keys()}

    # Compute final hybrid scores
    final_scores = {}
    for iid in cf_preds.keys():
        c = norm_content.get(iid, 0.0)
        f = norm_cf.get(iid, 0.0)
        final_scores[iid] = alpha * f + (1 - alpha) * c

    # Return top-k recommendations
    top_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for iid, score in top_items:
        movie_info = movies.loc[movies['item_id'] == iid]
        if len(movie_info) > 0:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            results.append((iid, score, title, genres))
        
    return results

# -------------------------
# Main execution
# -------------------------

def main():
    """Main function to run the recommendation system."""
    data_path = 'data/ml-100k'
    
    if not os.path.isdir(data_path):
        print(f"Error: Data directory '{data_path}' not found.")
        print("Please ensure you have the MovieLens 100k dataset in this path.")
        return

    try:
        ratings, movies = load_movielens_100k(path=data_path)
        print(f"\n{'='*70}")
        print(f"DATA LOADED SUCCESSFULLY")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n✗ Failed to load data: {e}")
        return

    # Ensure IDs are integers
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['item_id'] = ratings['item_id'].astype(int)
    movies['item_id'] = movies['item_id'].astype(int)

    print('Training collaborative filtering model...')
    cf_algo, train_rmse = train_cf_model(ratings, n_epochs=15, n_factors=30)
    print(f'✓ CF model trained. Final Training RMSE: {train_rmse:.4f}\n')

    print('Building content-based model...')
    movies_idxed, tfidf, tfidf_matrix = build_content_model(movies)
    print('✓ Content model ready.\n')

    # Sample recommendations
    sample_user = ratings['user_id'].unique()[0]
    print(f"{'='*70}")
    print(f"RECOMMENDATIONS FOR USER {sample_user} (Hybrid: 70% CF + 30% Content)")
    print(f"{'='*70}")
    
    recs = recommend_hybrid(sample_user, cf_algo, ratings, movies_idxed, tfidf_matrix, 
                           top_k=10, alpha=0.7)
    
    for rank, (iid, score, title, genres) in enumerate(recs, 1):
        print(f"{rank:2d}. [{iid:4d}] {title[:45]:45s}")
        print(f"     Genres: {genres[:60]}")
        print(f"     Score: {score:.4f}\n")

    # Similar movies example
    sample_item = movies['item_id'].iloc[10]
    sample_row = movies.loc[movies['item_id']==sample_item].iloc[0]
    sample_title = sample_row['title']
    sample_genres = sample_row['genres']
    
    print(f"{'='*70}")
    print(f"MOVIES SIMILAR TO: {sample_title}")
    print(f"Genres: {sample_genres}")
    print(f"{'='*70}\n")
    
    similars = get_similar_items(sample_item, movies_idxed, tfidf_matrix, top_n=5)
    for rank, s in enumerate(similars, 1):
        print(f"{rank}. [{s['item_id']:4d}] {s['title']}")
        print(f"   Genres: {s['genres']}\n")

if __name__ == '__main__':
    main()