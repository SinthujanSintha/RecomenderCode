# Parse the stringified features into their corresponding python objects
# Import CountVectorizer and create the count matrix
from ast import literal_eval
import numpy as np
import pandas as panda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from contentFilterBasedDescription import get_recommendations

df1 = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/tmdb_5000_credits.csv')
df2 = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/tmdb_5000_movies.csv')


df1.columns = ['id', 'tittle', 'cast', 'crew']
df2 = df2.merge(df1, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

df3 = df2[features].head(5)


# print(df3)

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    # Return empty list in case of missing/malformed data
    return []


# Define new director, cast, genres and keywords features that are in a suitable form.
df2['director'] = df2['crew'].apply(get_director)

features2 = ['cast', 'keywords', 'genres']
for feature in features2:
    df2[feature] = df2[feature].apply(get_list)


# Print the new features of the first 3 films
# print(df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Apply clean_data function to your features.
features3 = ['cast', 'keywords', 'director', 'genres']

for feature in features3:
    df2[feature] = df2[feature].apply(clean_data)


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + \
           x['director'] + ' ' + ' '.join(x['genres'])


df2['soup'] = df2.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

cosine_similarity2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping
df2 = df2.reset_index()
indices = panda.Series(df2.index, index=df2['title'])

recommend_movies = get_recommendations('The Avengers', cosine_similarity2, df2)
print(recommend_movies)