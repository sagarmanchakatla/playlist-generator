from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import faiss

app = Flask(__name__)

# Load dataset
df = pd.read_csv("spotify-data.csv")

# Select relevant audio features
audio_features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Normalize the audio features
df[audio_features] = (df[audio_features] - df[audio_features].mean()) / df[audio_features].std()

# Convert the DataFrame to a numpy array
data = df[audio_features].to_numpy().astype('float32')

# Build the FAISS index
index = faiss.IndexFlatL2(data.shape[1])  # L2 distance (Euclidean)
index.add(data)

# Function to recommend songs based on mood, genre, and artists
def recommend_songs(mood, favorite_genres, favorite_artists, top_n=10):
    try:
        # Filter songs based on mood, genre, and artists
        filtered_df = df.copy()

        # Filter by mood (assuming mood is a string like 'happy', 'sad', etc.)
        if mood == 'happy':
            filtered_df = filtered_df[(filtered_df['valence'] > 0.6) & (filtered_df['energy'] > 0.6)]
        elif mood == 'sad':
            filtered_df = filtered_df[(filtered_df['valence'] < 0.4) & (filtered_df['energy'] < 0.5)]
        elif mood == 'energetic':
            filtered_df = filtered_df[(filtered_df['valence'] > 0.5) & (filtered_df['energy'] > 0.7)]
        elif mood == 'calm':
            filtered_df = filtered_df[(filtered_df['valence'] > 0.3) & (filtered_df['valence'] < 0.6) & (filtered_df['energy'] < 0.5)]
        else:
            pass  # No mood filtering

        # Filter by favorite genres
        if favorite_genres:
            filtered_df = filtered_df[filtered_df['playlist_genre'].isin(favorite_genres)]

        # Filter by favorite artists
        if favorite_artists:
            filtered_df = filtered_df[filtered_df['track_artist'].isin(favorite_artists)]

        if filtered_df.empty:
            return pd.DataFrame()  # Return empty DataFrame if no songs match the criteria

        # Convert filtered DataFrame to numpy array
        filtered_data = filtered_df[audio_features].to_numpy().astype('float32')

        # Build a FAISS index for the filtered data
        filtered_index = faiss.IndexFlatL2(filtered_data.shape[1])
        filtered_index.add(filtered_data)

        # Randomly select a seed song from the filtered data
        seed_song_index = np.random.choice(filtered_df.index)
        seed_song_vector = filtered_data[filtered_df.index.get_loc(seed_song_index)].reshape(1, -1)

        # Search for similar songs
        distances, indices = filtered_index.search(seed_song_vector, top_n)

        # Get the recommended songs
        recommended_songs = filtered_df.iloc[indices[0]][['track_name', 'track_artist', 'playlist_genre', 'track_popularity']]

        return recommended_songs
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()  # Return empty DataFrame if an error occurs

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to recommend songs
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    mood = data.get('mood')
    favorite_genres = data.get('genres', [])
    favorite_artists = data.get('artists', [])

    if not mood:
        return jsonify({"error": "Please provide a mood."}), 400

    # Generate playlists for each genre
    playlists = {}
    for genre in favorite_genres:
        recommendations = recommend_songs(mood, [genre], favorite_artists)

        if recommendations.empty:
            playlists[genre] = {"error": "No songs found for this genre."}
        else:
            playlists[genre] = recommendations.to_dict(orient='records')

    return jsonify(playlists)

if __name__ == '__main__':
    app.run(debug=True)