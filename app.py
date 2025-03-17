from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load dataset
# df = pd.read_csv("spotify-data.csv")

# df = pd.read_csv('music_records.csv')

df = pd.read_csv('songs_dataset_with_thumbnails.csv')

# YouTube Data API key
# API_KEY = 'AIzaSyBe6hKp5D_VNizwr1BvhDxpbbH4IuJWVZ4'
# API_KEY = 'AIzaSyC7ZQW0R4iJ2rNM-DrHNC01wAWEgHLHlBc'
# API_KEY = 'AIzaSyAcegV_mEWr0_p_vLIymgOkOi8BcV3qLY4'
API_KEY = 'AIzaSyA6_UtbwToESxHuPJLno1ESx_2Huwk5RzQ'
youtube = build('youtube', 'v3', developerKey=API_KEY)

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

    ranked_df = filtered_df.sort_values(by='score', ascending=False)
    unique_songs = ranked_df.drop_duplicates(subset='track_name')

    return unique_songs

def generate_playlist(user_mood, favorite_artists, favorite_genre):
    filtered_songs = filter_songs(user_mood, favorite_artists, favorite_genre)
    top_songs = rank_songs(filtered_songs)

    if len(top_songs) < 10:
        additional_songs = filter_songs(user_mood, favorite_artists, None)
        additional_songs = rank_songs(additional_songs)
        top_songs = pd.concat([top_songs, additional_songs]).drop_duplicates(subset='track_name').head(10)
    print(top_songs.head(10))
    return top_songs[['track_name', 'track_artist', 'playlist_genre', 'track_popularity', 'YouTube URL', 'Thumbnail_URL']]

def generate_multiple_playlists(user_mood, favorite_artists, favorite_genres):
    playlists = {}
    for genre in favorite_genres:
        playlists[genre] = generate_playlist(user_mood, favorite_artists, genre)
    return playlists

def get_youtube_url_and_thumbnail(song_name, artist_name):
    search_response = youtube.search().list(
        q=f"{song_name} {artist_name} official audio",
        part="snippet",
        type="video",
        maxResults=1
    ).execute()

    items = search_response.get('items', [])
    if items:
        video_id = items[0]['id']['videoId']
        thumbnail_url = items[0]['snippet']['thumbnails']['default']['url']
        return f"https://www.youtube.com/watch?v={video_id}", thumbnail_url
    return None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_playlists', methods=['POST'])
def generate_playlists():
    data = request.json
    user_mood = data.get('mood')
    favorite_artists = data.get('artists', [])
    favorite_genres = data.get('genres', [])

    playlists = generate_multiple_playlists(user_mood, favorite_artists, favorite_genres)

    result = {}
    for genre, playlist in playlists.items():
        playlist_with_urls = []
        playlist_thumbnail = None
        for _, song in playlist.iterrows():
            # youtube_url, thumbnail_url = get_youtube_url_and_thumbnail(song['track_name'], song['track_artist'])
            youtube_url, thumbnail_url = song['YouTube URL'], song['Thumbnail_URL']
            playlist_with_urls.append({
                'track_name': song['track_name'],
                'track_artist': song['track_artist'],
                'playlist_genre': song['playlist_genre'],
                'track_popularity': song['track_popularity'],
                'youtube_url': youtube_url,
                'thumbnail_url': thumbnail_url
            })
            if not playlist_thumbnail:
                playlist_thumbnail = thumbnail_url

        
        result[genre] = {
            'playlist_thumbnail': playlist_thumbnail,
            'songs': playlist_with_urls
        }
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
