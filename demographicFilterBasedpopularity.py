import pandas as panda

df1 = panda.read_csv('DataSet/tmdb_5000_credits.csv')
df2 = panda.read_csv('DataSet/tmdb_5000_movies.csv')



df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

pop= df2.sort_values('popularity', ascending=False)
print(pop[['popularity','title']].head(5))


