import pandas as panda
import matplotlib.pyplot as plt

df1 = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/tmdb_5000_credits.csv')
df2 = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/tmdb_5000_movies.csv')



df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

pop= df2.sort_values('popularity', ascending=False)
# print(pop[['popularity','title']].head(5))

plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()

