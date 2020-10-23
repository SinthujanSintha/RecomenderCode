import pandas as panda
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()
ratings = panda.read_csv('/home/sinthujan/SinthuProgramming/PythonPyCharm/RecomederSytems/CommonRecommender/RecomenderCode/Dataset/ratings.csv')
# print(ratings.head())

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

svd = SVD()
# print(cross_validate(svd, data, measures=['RMSE', 'MAE']))

trainset = data.build_full_trainset()
svd.fit(trainset)
# print(ratings[ratings['userId'] == 1])

print(svd.predict(1, 302, 3))