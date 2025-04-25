import matplotlib.pyplot as plt
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

import autograd
import autograd.numpy as ag_np
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

from surprise import SVD,KNNBasic,KNNWithZScore,KNNWithMeans,SlopeOne,CoClustering
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from surprise import model_selection

from surprise import SVD, SVDpp
from surprise import Dataset, Reader
from surprise.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# ------------------------------------------------------

##LOADING DATA##
reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)

# train_data = Dataset.load_from_file(
#     'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
# # print(train_set[:5])
# train_set = train_data.build_full_trainset()
# raw_ratings = train_set.build_testset()


#MAKING MERGE DATA FRAME
df = pd.read_csv('data_movie_lens_100k/ratings_all_development_set.csv')
user_df = pd.read_csv('data_movie_lens_100k/user_info.csv')
movie_df = pd.read_csv('data_movie_lens_100k/movie_info.csv')

merged_df = pd.merge(df, user_df, on='user_id', how='inner')
merged_df = merged_df.drop(columns=['orig_user_id'])

merged_movie_df = pd.merge(merged_df, movie_df, on='item_id', how='inner')
merged_movie_df = merged_movie_df.drop(columns=["orig_item_id"])
merged_movie_df = merged_movie_df.drop(columns=["title"])

merged_movie_df = merged_movie_df.to_numpy()
print(merged_movie_df.shape)

# uid_iid = merged_movie_df[]
train = merged_movie_df[:, :2]

third_col = merged_movie_df[:, 2:3]               # shape (n, 1)
remaining_cols = np.delete(merged_movie_df, 2, axis=1)  # remove 3rd column

# Concatenate to move third column to the end
train = np.hstack((remaining_cols, third_col))

print("trian: ")
print(train[:,-1])
print(train[:,:-1])

# merged_movie_df_x = merged_movie_df.drop(columns=["rating"])

# rating_col = merged_movie_df.pop("rating").to_numpy()

# rating_col = merged_movie_df["rating"].values
# merged_movie_df = merged_movie_df.drop(columns=["rating"])


# merged_movie_df["rating"] = rating_col

# print(merged_movie_df)


# DATA_PATH = 'data_movie_lens_100k/' # TODO fixme: Path to where dataset csv files live on your system
# if not os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv')):
#     try:
#         DATA_PATH = os.path.join(os.environ.get("HOME", ""),
#                     'courses/cs135-25s-staffonly/proj_src/projB/data_movie_lens_100k/')
#         assert os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv'))
#     except AssertionError:
#         print("Please store path to movie_lens_100k dataset in DATA_PATH")
# assert os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv'))


# df_masked = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")

# df_masked['user_id'] = df_masked['user_id'].astype(str)
# df_masked['item_id'] = df_masked['item_id'].astype(str)
# df_masked['rating'] = 0


# # Create a list of (uid, iid) pairs
# predict_pairs = list(zip(df_masked['user_id'], df_masked['item_id'],df_masked['rating']))


# dev_set = Dataset.load_from_file(
#    os.path.join(DATA_PATH, 'ratings_all_development_set.csv'), reader=reader)
# dev_set_for_fit = dev_set.build_full_trainset()
# dev_set_for_predict = dev_set_for_fit.build_testset()
# dev_set_for_fit.global_mean

# print("Global Mean: ", dev_set_for_fit.global_mean)

# KNN for Surprise
# param_grid = {
#     'k': [30,40,50,60,70],
#     'sim_options': {
#         'name': ['cosine', 'pearson', 'pearson_baseline'],
#         'user_based': [True, False]  # True = user-user, False = item-item
#     },
#     'min_k': [1, 3, 5,10,50],
#     'verbose': [False]
# }
# model = model_selection.search.RandomizedSearchCV(KNNWithMeans, param_grid, n_iter=50, measures=['mae'], refit=True, n_jobs=-1)
# model.fit(dev_set) 
# best_model = model.best_estimator['mae']
# best_model.fit(dev_set_for_fit)

# predictions = [best_model.predict(uid, iid) for (uid, iid, _) in predict_pairs]
# for i in range(25):
#     print(predictions[i][2],predictions[i][3])

# ##KNNBasic
# print(model.best_score)
# print(model.best_params)

# KNN for SKlearn

# parameters = {
#     "n_neighbors": [1,5,30,50,75,100],
#     "weights": ['uniform', 'distance'],
#     "n_jobs": [-1]
# }

# print(uid_iid)

mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
param_grid = {
    'n_neighbors': [5, 10, 20, 30, 50],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
    'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

model = GridSearchCV(
    KNeighborsRegressor(),
    param_grid,
    scoring=mae_scorer,
    cv=5
)


# # Ensure all data types are floats
# merged_movie_df_x = merged_movie_df_x.astype(np.float64)
# rating_col = pd.to_numeric(rating_col, errors='coerce')

# # Fill NaNs if any remain
# merged_movie_df_x = merged_movie_df_x.fillna(0)
# rating_col = rating_col.fillna(rating_col.mean())

# # Validate: check for NaNs or invalid types
# assert not merged_movie_df_x.isnull().values.any(), "NaNs remain in feature matrix"
# assert not rating_col.isnull().values.any(), "NaNs remain in target"
model.fit(train[:, :-1], train[:,-1])
