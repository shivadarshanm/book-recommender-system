import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

# Load CSVs
books = pd.read_csv('data/BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv('data/BX-Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip')

# Rename columns
books.columns = ['ISBN', 'title', 'author', 'year', 'publisher', 'image_s', 'image_m', 'image_l']
ratings.columns = ['user_id', 'ISBN', 'rating']
users.columns = ['user_id', 'location', 'age']

# Keep only explicit ratings
ratings = ratings[ratings['rating'] > 0]

# 🔁 Loosened: Users with at least 50 ratings
user_counts = ratings['user_id'].value_counts()
active_users = user_counts[user_counts >= 50].index
ratings = ratings[ratings['user_id'].isin(active_users)]

# 🔁 Loosened: Books with at least 20 ratings
book_counts = ratings['ISBN'].value_counts()
popular_books = book_counts[book_counts >= 20].index
ratings = ratings[ratings['ISBN'].isin(popular_books)]

# Merge datasets
merged = pd.merge(ratings, books[['ISBN', 'title', 'image_l']], on='ISBN')
merged.rename(columns={'image_l': 'image_url'}, inplace=True)

# Pivot table
book_pivot = merged.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)

# Train model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(book_pivot.values)

# Book names
book_names = list(book_pivot.index)

# Save artifacts
with open('artifacts/final_rating.pkl', 'wb') as f:
    pickle.dump(merged, f)

with open('artifacts/book_pivot.pkl', 'wb') as f:
    pickle.dump(book_pivot, f)

with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('artifacts/book_names.pkl', 'wb') as f:
    pickle.dump(book_names, f)

print(f"✅ Filtered & saved: book_pivot shape = {book_pivot.shape}")
