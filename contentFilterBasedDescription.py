import pandas as panda
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

df1 = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/tmdb_5000_credits.csv')
df2 = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/tmdb_5000_movies.csv')

df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

# print(df2['overview'].head(5))

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)

# Compute the cosine similarity matrix
cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim, data_frame):
    """

    :param title: title of the movies
    :param cosine_sim: cosine similarity matrix
    :param data_frame: data set of movies
    :return: return the list with ten movies which are suitable to target movies
    """
    # Construct a reverse map of indices and movie titles
    indices = panda.Series(data_frame.index, index=data_frame['title']).drop_duplicates()

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return data_frame['title'].iloc[movie_indices]


get_recommendations('The Avengers', cosine_similarity, df2)
# print(recommend_movies)
