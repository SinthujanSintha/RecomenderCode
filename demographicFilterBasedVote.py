import pandas as panda

df1 = panda.read_csv('DataSet/tmdb_5000_credits.csv')
df2 = panda.read_csv('DataSet/tmdb_5000_movies.csv')

df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')
#print(df2.head(5))

C= df2['vote_average'].mean()
#print(C)=6.092171559442016

m= df2['vote_count'].quantile(0.9)
#print(m) = 1838.4000000000015

qualify_movies = df2.copy().loc[df2['vote_count'] >= m]
#print(qualify_movies.shape) = (481, 23)

#Imdb weighted average ratings
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
qualify_movies[ 'score'] = qualify_movies .apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
qualify_movies = qualify_movies.sort_values('score', ascending=False)


#Print the top 15 movies

recomenderMovies = qualify_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
print(recomenderMovies)
