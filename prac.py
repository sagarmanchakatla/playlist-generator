import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- 1. Data Loading and Preparation ---

def load_and_preprocess_data(filepath='spotify-data.csv'):
    """Loads and preprocesses the Spotify dataset."""
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

    # Select relevant features (adjust based on your dataset)
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']  # Numerical features

    # Handle missing values (if any)
    data = data.dropna(subset=features)

    # Scale numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    scaled_df = pd.DataFrame(scaled_features, columns=features)
    data = pd.concat([data.drop(columns=features).reset_index(drop=True), scaled_df], axis=1)

    return data

# --- 2. User Input Handling ---

def get_user_input():
    """Gets user input for mood, artist, and genre."""
    mood = input("Enter your desired mood (e.g., happy, sad, energetic): ").lower()
    artist = input("Enter your favorite artist: ").lower()
    genre = input("Enter your favorite genre: ").lower()
    return mood, artist, genre

# --- 3. Filtering and Feature Extraction ---

def filter_and_extract_features(data, mood, artist, genre):
    """Filters data based on user input and extracts relevant features."""

    # Simple mood mapping (expand as needed)
    mood_mappings = {
        'happy': {'valence': 0.7, 'energy': 0.6},
        'sad': {'valence': 0.3, 'energy': 0.4},
        'energetic': {'energy': 0.8, 'danceability': 0.7},
        'calm': {'energy': 0.3, 'acousticness': 0.7},
        'chill': {'energy': 0.4, 'acousticness': 0.6, 'valence': 0.5},
    }

    mood_filters = mood_mappings.get(mood, {})  # Default to empty if mood not found

    # Filter by artist and genre (case-insensitive)
    filtered_data = data[data['track_artist'].str.lower().str.contains(artist) | data['playlist_genre'].str.lower().str.contains(genre)]

    # Further filter by mood-related features
    for feature, threshold in mood_filters.items():
        if feature in filtered_data.columns:
            if feature == 'valence':
                filtered_data = filtered_data[filtered_data['valence'] >= threshold]
            else:
                filtered_data = filtered_data[filtered_data[feature] >= threshold]

    return filtered_data

# --- 4. KNN Recommendation ---

def recommend_songs(filtered_data, data, n_recommendations=5):
    """Recommends songs using KNN based on filtered data."""
    if filtered_data.empty:
        print("No songs found matching your criteria.")
        return

    features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

    knn = NearestNeighbors(n_neighbors=n_recommendations + 1, metric='euclidean') #+1 to remove the song itself if present.
    knn.fit(data[features])

    # Use the first song from the filtered dataset as the query point
    query_point = filtered_data[features].iloc[0].values.reshape(1, -1)

    distances, indices = knn.kneighbors(query_point)
    recommended_songs = data.iloc[indices[0][1:]] # Skip the first one (the query song itself)

    return recommended_songs

# --- 5. Main Execution ---

if __name__ == "__main__":
    data = load_and_preprocess_data()
    if data is not None:
        mood, artist, genre = get_user_input()
        filtered_data = filter_and_extract_features(data, mood, artist, genre)
        recommendations = recommend_songs(filtered_data, data)

        if recommendations is not None and not recommendations.empty:
            print("\nRecommended Songs:")
            for index, row in recommendations.iterrows():
                print(f"- {row['track_name']} by {row['track_artist']}") # corrected line