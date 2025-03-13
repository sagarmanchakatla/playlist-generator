import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('spotify-data.csv')

# Preprocessing
# Handle missing values, encode categorical variables, etc.

# Feature selection
features = ['track_artist', 'playlist_genre', 'danceability', 'energy', 'valence']
X = data[features]
y = data['track_id']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['danceability', 'energy', 'valence']),
        ('cat', OneHotEncoder(), ['track_artist', 'playlist_genre'])
    ])

# Create a model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train the model
model.fit(X_train, y_train)


# Mapping of moods to feature values
mood_mapping = {
    'angry': {'danceability': 0.6, 'energy': 0.9, 'valence': 0.3},
    'disgust': {'danceability': 0.4, 'energy': 0.7, 'valence': 0.2},
    'fear': {'danceability': 0.3, 'energy': 0.8, 'valence': 0.1},
    'happy': {'danceability': 0.8, 'energy': 0.8, 'valence': 0.9},
    'sad': {'danceability': 0.3, 'energy': 0.4, 'valence': 0.2},
    'surprise': {'danceability': 0.7, 'energy': 0.8, 'valence': 0.6},
    'neutral': {'danceability': 0.5, 'energy': 0.5, 'valence': 0.5}
}

# Function to get feature values for a given mood
def get_mood_features(mood):
    return mood_mapping.get(mood, {'danceability': 0.5, 'energy': 0.5, 'valence': 0.5})

# Function to generate playlist
def generate_playlist(mood, favorite_artist, genre, model, data, num_songs=10):
    # Get mood features
    mood_features = get_mood_features(mood)

    # Create a user input dataframe
    user_input = pd.DataFrame({
        'track_artist': [favorite_artist],
        'playlist_genre': [genre],
        'danceability': [mood_features['danceability']],
        'energy': [mood_features['energy']],
        'valence': [mood_features['valence']]
    })

    # Predict songs
    predictions = model.predict(user_input)

    # Ensure unique songs
    unique_songs = set()
    playlist = []
    for song in predictions:
        if song not in unique_songs:
            playlist.append(song)
            unique_songs.add(song)
        if len(playlist) == num_songs:
            break

    return playlist

# Example usage
mood = 'happy'
favorite_artist = 'Drake'
genre = 'pop'
playlist = generate_playlist(mood, favorite_artist, genre, model, data)
print(playlist)