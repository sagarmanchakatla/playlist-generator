from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load dataset
df = pd.read_csv("spotify-data.csv")

# Load the saved model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Select relevant audio features
audio_features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Function to recommend songs based on mood, genre, and artists
# def recommend_songs(mood, favorite_genres, favorite_artists, top_n=10):
#     try:
#         # Filter songs based on mood, genre, and artists
#         filtered_df = df.copy()

#         # Filter by mood (assuming mood is a string like 'happy', 'sad', etc.)
#         if mood == 'happy':
#             filtered_df = filtered_df[(filtered_df['valence'] > 0.6) & (filtered_df['energy'] > 0.6)]
#         elif mood == 'sad':
#             filtered_df = filtered_df[(filtered_df['valence'] < 0.4) & (filtered_df['energy'] < 0.5)]
#         elif mood == 'energetic':
#             filtered_df = filtered_df[(filtered_df['valence'] > 0.5) & (filtered_df['energy'] > 0.7)]
#         elif mood == 'calm':
#             filtered_df = filtered_df[(filtered_df['valence'] > 0.3) & (filtered_df['valence'] < 0.6) & (filtered_df['energy'] < 0.5)]
#         else:
#             pass  # No mood filtering

#         # Filter by favorite genres
#         if favorite_genres:
#             filtered_df = filtered_df[filtered_df['playlist_genre'].isin(favorite_genres)]

#         # Filter by favorite artists
#         if favorite_artists:
#             filtered_df = filtered_df[filtered_df['track_artist'].isin(favorite_artists)]

#         if filtered_df.empty:
#             return pd.DataFrame()  # Return empty DataFrame if no songs match the criteria

#         # Normalize the filtered data using the saved scaler
#         filtered_data = scaler.transform(filtered_df[audio_features])

#         # Ensure the input data has valid feature names
#         filtered_df[audio_features] = filtered_data

#         # Predict clusters for the filtered data
#         filtered_df['cluster'] = kmeans.predict(filtered_df[audio_features])

#         # Randomly select a seed song from the filtered data
#         seed_song_index = np.random.choice(filtered_df.index)
#         seed_song_cluster = filtered_df.loc[seed_song_index, 'cluster']

#         # Get songs from the same cluster
#         cluster_songs = filtered_df[filtered_df['cluster'] == seed_song_cluster]
#         if len(cluster_songs) > 0:
#             recommended_songs = cluster_songs.sample(min(top_n, len(cluster_songs)))
#         else:
#             recommended_songs = pd.DataFrame()  # Return empty DataFrame if no songs are found

#         return recommended_songs[['track_name', 'track_artist', 'playlist_genre', 'track_popularity']]
#     except Exception as e:
#         print(f"Error: {e}")
#         return pd.DataFrame()  # Return empty DataFrame if an error occurs


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

        # Normalize the filtered data using the saved scaler
        filtered_data = scaler.transform(filtered_df[audio_features])

        # Ensure the input data has valid feature names
        filtered_df[audio_features] = filtered_data

        # Predict clusters for the filtered data
        filtered_df['cluster'] = kmeans.predict(filtered_df[audio_features])

        # Randomly select a seed song from the filtered data
        seed_song_index = np.random.choice(filtered_df.index)
        seed_song_cluster = filtered_df.loc[seed_song_index, 'cluster']

        # Get songs from the same cluster
        cluster_songs = filtered_df[filtered_df['cluster'] == seed_song_cluster]

        # If the cluster has fewer than top_n songs, add songs from other clusters
        if len(cluster_songs) < top_n:
            remaining_slots = top_n - len(cluster_songs)
            other_clusters = filtered_df[filtered_df['cluster'] != seed_song_cluster]
            if len(other_clusters) > 0:
                additional_songs = other_clusters.sample(min(remaining_slots, len(other_clusters)))
                cluster_songs = pd.concat([cluster_songs, additional_songs])

        # If still fewer than top_n songs, return all available songs
        if len(cluster_songs) > top_n:
            recommended_songs = cluster_songs.sample(top_n)
        else:
            recommended_songs = cluster_songs

        return recommended_songs[['track_name', 'track_artist', 'playlist_genre', 'track_popularity']]
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