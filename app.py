import pickle
import streamlit as st
import numpy as np 
from sklearn.neighbors import NearestNeighbors

st.header("Book Recommendation System")
books = pickle.load(open('webfiles/books.pkl', 'rb'))
book_names = pickle.load(open('webfiles/book_names.pkl', 'rb'))
book_pivot = pickle.load(open('webfiles/pt.pkl', 'rb'))
final_rating = pickle.load(open('webfiles/final_ratings.pkl', 'rb'))
modelknn = pickle.load(open('webfiles/modelknn.pkl', 'rb'))
similarity_scores = pickle.load(open('webfiles/similarity_scores.pkl', 'rb'))

# ****************** For Content based recommendation ***************************
def fetch_poster(suggestion):
    poster_url = []

    for i in suggestion:
        book_id = i[0]
        url = books.iloc[book_id]['Image-URL-M']
        poster_url.append(url)

    return poster_url

def recommend_book(book_name, books):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    suggestion = sorted(list(enumerate(similarity_scores[book_id])), key = lambda x:x[1], reverse = True)[1:6]

    poster_url = fetch_poster(suggestion)
    
    for i in suggestion:
        book_id = i[0]
        rec_books = books.iloc[book_id]['Book-Title']
        books_list.append(rec_books)
    return books_list , poster_url       



selected_books = st.selectbox(
    "select a book from the dropdown for CONTENT based recommendation",
    book_names
)


if st.button('Show Recommendation'):
    recommended_books,poster_url = recommend_book(selected_books, books)
    col1, col2, col3, col4, col5 = st.columns(5)   
    with col1:
        st.text(recommended_books[0])
        st.image(poster_url[0])
    with col2:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col3:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col4:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col5:
        st.text(recommended_books[4])
        st.image(poster_url[4])


# # ************************ For User based recommendation ********************
        
# selected_user = st.selectbox(
#     "select a book from the dropdown for USER based recommendation",
#     final_rating.index
# )

# def fetch_poster(suggestion):
#     poster_url = []

#     for i in suggestion:
#         book_id = i[0]
#         url = books.iloc[book_id]['Image-URL-M']
#         poster_url.append(url)

#     return poster_url

# def recommend_book(user_id, book_pivot):
#     books_list = []
#     # book_id = np.where(book_pivot.index == book_id)[0][0]
#     # suggestion = sorted(list(enumerate(similarity_scores[book_id])), key = lambda x:x[1], reverse = True)[1:6]
#     # modelknn = NearestNeighbors(algorithm='brute')
#     distance, suggestion = modelknn.kneighbors(book_pivot.iloc[user_id,:].values.reshape(1,-1), n_neighbors=6 )

#     print(suggestion)
#     poster_url = fetch_poster(suggestion)
    
#     for i in range(len(suggestion)):
#         rec_books = book_pivot.index[suggestion[i]]
#         books_list.append(rec_books)
#     return books_list , poster_url       

# if st.button('Show User Recommendation'):
#     recommended_books,poster_url = recommend_book(selected_user, book_pivot)
#     col1, col2, col3, col4, col5 = st.columns(5)   
#     with col1:
#         st.text(recommended_books[0])
#         st.image(poster_url[0])
#     with col2:
#         st.text(recommended_books[1])
#         st.image(poster_url[1])
#     with col3:
#         st.text(recommended_books[2])
#         st.image(poster_url[2])
#     with col4:
#         st.text(recommended_books[3])
#         st.image(poster_url[3])
#     with col5:
#         st.text(recommended_books[4])
#         st.image(poster_url[4])
