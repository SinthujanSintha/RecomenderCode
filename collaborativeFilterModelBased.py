import pandas as panda
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()
ratings = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/RecomenderCode/DataSet/ratings.csv')
# print(ratings.head())

# ratings= ratings.take(ratings.index[0:25214])
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
svd = SVD()
# print(cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=6,verbose=True))


trainset = data.build_full_trainset()
svd.fit(trainset)
# print(ratings[ratings['userId'] == 1])

print(svd.predict(1, 302))
