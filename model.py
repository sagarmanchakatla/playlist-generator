import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("spotify-data.csv")

# Select relevant audio features
audio_features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Normalize the audio features
scaler = StandardScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# Train a K-Means clustering model
kmeans = KMeans(n_clusters=10, random_state=42)  # 10 clusters
df['cluster'] = kmeans.fit_predict(df[audio_features])

# Save the model and scaler
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')