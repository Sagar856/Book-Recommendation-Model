import pickle
import streamlit as st
import numpy as np 

st.header("Book Recommendation System Using ML ")
books = pickle.load(open('webfiles/books.pkl', 'rb'))
book_names = pickle.load(open('webfiles/book_names.pkl', 'rb'))
book_pivot = pickle.load(open('webfiles/pt.pkl', 'rb'))
final_rating = pickle.load(open('webfiles/final_ratings.pkl', 'rb'))
similarity_scores = pickle.load(open('webfiles/similarity_scores.pkl', 'rb'))


# ****************** For Content based recommendation ***************************

def recommend_book(book_name, books):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    suggestion = sorted(list(enumerate(similarity_scores[book_id])), key = lambda x:x[1], reverse = True)[1:6]
    
    data = []
    for i in suggestion:
        item = []
        temp_df = books[books['Book-Title'] == book_pivot.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Year-Of-Publication'].values.astype(str)))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)   
    return data   


selected_books = st.selectbox(
    "Select a book from the dropdown for recommendation",
    book_pivot.index, placeholder="Choose an option",index=None,
)


if st.button('Show Recommendation'):
    data = recommend_book(selected_books, books)
    col1, col2, col3, col4, col5 = st.columns(5)   
    with col1:
        st.text(data[0][0])
        st.image(data[0][3], width=125)
        st.text('Author:' + '\n' + data[0][1] + '.')
        st.text('Published year:' + '\n' + data[0][2])
    with col2:
        st.text(data[1][0])
        st.image(data[1][3], width=125)
        st.text('Author:' + '\n' + data[1][1] + '.')
        st.text('Published year:' + '\n' + data[1][2])
    with col3:
        st.text(data[2][0])
        st.image(data[2][3], width=125)
        st.text('Author:' + '\n' + data[2][1] + '.')
        st.text('Published year:' + '\n' + data[2][2])
    with col4:
        st.text(data[3][0])
        st.image(data[3][3], width=125)
        st.text('Author:' + '\n' + data[3][1] + '.')
        st.text('Published year:' + '\n' + data[3][2])
    with col5:
        st.text(data[4][0])
        st.image(data[4][3], width=125)
        st.text('Author:' + '\n' + data[4][1] + '.')
        st.text('Published year:' + '\n' + data[4][2])


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
