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

#MAKING MERGE DATA FRAME
df = pd.read_csv('data_movie_lens_100k/ratings_all_development_set.csv')
user_df = pd.read_csv('data_movie_lens_100k/user_info.csv')
movie_df = pd.read_csv('data_movie_lens_100k/movie_info.csv')

merged_df = pd.merge(df, user_df, on='user_id', how='inner')
merged_df = merged_df.drop(columns=['orig_user_id'])

merged_movie_df = pd.merge(merged_df, movie_df, on='item_id', how='inner')
merged_movie_df = merged_movie_df.drop(columns=["orig_item_id"])
merged_movie_df = merged_movie_df.drop(columns=["title"])

# Optional: one-hot encode any object columns before np conversion
categorical_cols = merged_movie_df.select_dtypes(include=['object']).columns
merged_movie_df = pd.get_dummies(merged_movie_df, columns=categorical_cols)

merged_movie_df = merged_movie_df.to_numpy()
print(merged_movie_df.shape)
print(merged_movie_df.dtype)

# uid_iid = merged_movie_df[]
train = merged_movie_df[:, :2]

third_col = merged_movie_df[:, 2:3]               # shape (n, 1)
remaining_cols = np.delete(merged_movie_df, 2, axis=1)  # remove 3rd column

# Concatenate to move third column to the end
train = np.hstack((remaining_cols, third_col))
train = train.astype(np.float64)
print("trian: ")
print(train[:,-1])
print(train[:,:-1])


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
assert not np.isnan(train).any()
assert np.isfinite(train).all()
train[:, -1] = np.nan_to_num(train[:, -1], nan=0.0, posinf=0.0, neginf=0.0)
print(train.dtype)
print("FITTING")
model.fit(train[:100, :-1], train[:100,-1])
