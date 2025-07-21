import pickle
import streamlit as st
import numpy as np

st.title('📚 Book Recommender System')
st.markdown("Get personalized book recommendations using machine learning!")

# Load artifacts
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))

def fetch_poster(suggestions):
    poster_urls = []
    for book_id in suggestions[0]:
        book_name = book_pivot.index[book_id]
        try:
            idx = final_rating[final_rating['title'] == book_name].index[0]
            url = final_rating.loc[idx, 'image_url']
        except:
            url = "https://via.placeholder.com/150"
        poster_urls.append(url)
    return poster_urls

def recommend_books(book_name):
    books_list = []
    book_index = np.where(book_pivot.index == book_name)[0][0]

    n_neighbors = min(6, book_pivot.shape[0])
    distances, suggestions = model.kneighbors(
        book_pivot.iloc[book_index, :].values.reshape(1, -1),
        n_neighbors=n_neighbors
    )

    poster_urls = fetch_poster(suggestions)

    for i in range(1, n_neighbors):
        books_list.append(book_pivot.index[suggestions[0][i]])

    return books_list, poster_urls[1:n_neighbors]

# UI Logic
if len(book_names) < 2:
    st.error("⚠️ Not enough data to generate recommendations. Please check your dataset.")
else:
    selected_book = st.selectbox("Search or choose a book you like:", book_names)

    if st.button('Show Recommendations'):
        names, posters = recommend_books(selected_book)
        cols = st.columns(len(names))
        for i, col in enumerate(cols):
            with col:
                st.text(names[i])
                st.image(posters[i])
