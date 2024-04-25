import pickle
import streamlit as st
import numpy as np 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

st.title("Book Recommendation System")
books = pickle.load(open('webfiles/books.pkl', 'rb'))
book_pivot = pickle.load(open('webfiles/pt.pkl', 'rb'))
final_rating = pickle.load(open('webfiles/final_ratings.pkl', 'rb'))
users = pickle.load(open('webfiles/users.pkl', 'rb'))
similarity_scores = pickle.load(open('webfiles/similarity_scores.pkl', 'rb'))


# ****************** For Content based recommendation ***************************

st.subheader('Content Based Recommender')
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
    "Select a book from the dropdown for CONTENT BASED recommendation",
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


st.divider()


# # ************************ For User based recommendation ********************
        
st.subheader('User Based Recommender')
selected_user = st.selectbox(
    "select a User ID from the dropdown for USER BASED recommendation",
    users.index
)

def recommendUsingKNN(user_id):
  modelknn = NearestNeighbors(algorithm='brute')
  book_sparse = csr_matrix(book_pivot)
  modelknn.fit(book_sparse)
  distances, suggestions = modelknn.kneighbors(book_pivot.iloc[user_id, :].values.reshape(1, -1))

  for i in range(len(suggestions)):
    item = book_pivot.index[suggestions[i]]

  recommendation = []
  for j in range(len(item)):
    temp_df1 = final_rating[final_rating['Book-Title'] == item[j]].head()
    recommendation.extend(list(temp_df1.drop_duplicates('Book-Title')['Book-Title'].values))
    recommendation.extend(list(temp_df1.drop_duplicates('Book-Title')['Book-Author'].values))
    recommendation.extend(list(temp_df1.drop_duplicates('Book-Title')['Year-Of-Publication'].values.astype(int)))
    recommendation.extend(list(temp_df1.drop_duplicates('Book-Title')['Image-URL-M'].values))
  return recommendation    

if st.button('Show User Recommendation'):
    recommended_books = recommendUsingKNN(selected_user)
    col1, col2, col3, col4, col5 = st.columns(5)   
    with col1:
        st.text(recommended_books[0])
        st.image(recommended_books[3])
        st.text('Author:' + '\n' + recommended_books[1] + '.')
        st.text('Published year:')
        st.text(recommended_books[2])
    with col2:
        st.text(recommended_books[4])
        st.image(recommended_books[7])
        st.text('Author:' + '\n' + recommended_books[5] + '.')
        st.text('Published year:')
        st.text(recommended_books[6])
    with col3:
        st.text(recommended_books[8])
        st.image(recommended_books[11])
        st.text('Author:' + '\n' + recommended_books[9] + '.')
        st.text('Published year:')
        st.text(recommended_books[10])
    with col4:
        st.text(recommended_books[12])
        st.image(recommended_books[15])
        st.text('Author:' + '\n' + recommended_books[13] + '.')
        st.text('Published year:')
        st.text(recommended_books[14])
    with col5:
        st.text(recommended_books[16])
        st.image(recommended_books[19])
        st.text('Author:' + '\n' + recommended_books[17] + '.')
        st.text('Published year:')
        st.text(recommended_books[18])


st.divider()
st.page_link("https://github.com/Sagar856/Book-Recommendation-Model.git",label = 'Click here -- Github source code')
