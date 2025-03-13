import pandas as pd
import numpy as np
import streamlit as st

# Load dataset
df = pd.read_csv("spotify-data.csv")

def get_mood_criteria(mood):
    mood_map = {
        'happy': {'valence': (0.6, 1.0), 'energy': (0.6, 1.0)},
        'sad': {'valence': (0.0, 0.4), 'energy': (0.0, 0.5)},
        'energetic': {'valence': (0.5, 1.0), 'energy': (0.7, 1.0)},
        'calm': {'valence': (0.3, 0.6), 'energy': (0.0, 0.5)}
    }
    return mood_map.get(mood, {'valence': (0, 1), 'energy': (0, 1)})

def filter_songs(user_mood, favorite_artists, favorite_genre):
    mood_criteria = get_mood_criteria(user_mood)

    filtered_df = df[
        (df['valence'].between(*mood_criteria['valence'])) &
        (df['energy'].between(*mood_criteria['energy']))
    ]

    if favorite_artists:
        filtered_df = filtered_df[filtered_df['track_artist'].str.contains('|'.join(favorite_artists), case=False, na=False)]

    if favorite_genre:
        filtered_df = filtered_df[filtered_df['playlist_genre'].str.contains(favorite_genre, case=False, na=False)]

    return filtered_df

def rank_songs(filtered_df):
    if filtered_df.empty:
        return pd.DataFrame()

    filtered_df['score'] = (
        filtered_df['valence'] * 0.4 +
        filtered_df['energy'] * 0.4 +
        np.log1p(filtered_df['track_popularity']) * 0.2
    )

    # Ensure we have at least 10 unique songs
    ranked_df = filtered_df.sort_values(by='score', ascending=False)
    unique_songs = ranked_df.drop_duplicates(subset='track_name')

    return unique_songs

def generate_playlist(user_mood, favorite_artists, favorite_genre):
    filtered_songs = filter_songs(user_mood, favorite_artists, favorite_genre)
    top_songs = rank_songs(filtered_songs)

    # If not enough songs match the criteria, relax the constraints
    if len(top_songs) < 10:
        # First, try relaxing only the genre constraint
        additional_songs = filter_songs(user_mood, favorite_artists, None)
        additional_songs = rank_songs(additional_songs)
        top_songs = pd.concat([top_songs, additional_songs]).drop_duplicates(subset='track_name').head(10)

    return top_songs[['track_name', 'track_artist', 'playlist_genre', 'track_popularity']]

def generate_multiple_playlists(user_mood, favorite_artists, favorite_genres):
    playlists = {}
    for genre in favorite_genres:
        playlists[genre] = generate_playlist(user_mood, favorite_artists, genre)
    return playlists

# Streamlit UI
st.title("🎵 AI Playlist Generator")

mood = st.selectbox("Select your mood:", ["happy", "sad", "energetic", "calm"])
artists = st.text_input("Enter your favorite artists (comma-separated):")
genres = st.text_input("Enter your favorite genres (comma-separated):")

if st.button("Generate Playlists"):
    favorite_artists = [artist.strip() for artist in artists.split(',')] if artists else []
    favorite_genres = [genre.strip() for genre in genres.split(',')] if genres else []

    playlists = generate_multiple_playlists(mood, favorite_artists, favorite_genres)

    if playlists:
        for genre, playlist in playlists.items():
            st.write(f"Playlist for genre '{genre}':")
            st.dataframe(playlist)
    else:
        st.write("No matching songs found. Try different inputs!")
