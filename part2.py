
import matplotlib.pyplot as plt
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

import autograd
import autograd.numpy as ag_np

import pandas as pd


import matplotlib.pyplot as plt
import sklearn

from surprise import SVD,SVDpp
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from surprise import model_selection
import matplotlib.pyplot as plt

# ------------------------------------------------------

plt.style.use('seaborn-v0_8')

# Load Data
# train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
# train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)



# ## Load the entire dev set in surprise's format
train_set = Dataset.load_from_file(
    'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
# train_set = train_set.build_full_trainset()

## Fit model like in problem 1
# model = SVD()

param_grid = {
    "n_factors": [1,10,50,100],
    # "lr_all": ag_np.logspace(0.001, 0.1, 10) NOTE this results in a best_score above 1. 
    "lr_all": [0.00000000000001]
    
}
model = model_selection.search.RandomizedSearchCV(SVD, param_grid, n_iter=4, measures=['mae'], refit=True, n_jobs=-1)
model.fit(train_set)

print("Best score:", model.best_score['mae'])
print("Best params:", model.best_params['mae'])


df_test = pd.read_csv(
    'data_movie_lens_100k/ratings_masked_leaderboard_set.csv')

# Fill missing ratings with dummy value
df_test['rating'] = df_test['rating'].fillna(0.0)

# Convert to list of (user, item, rating) tuples
test_triples = list(zip(df_test['user_id'], df_test['item_id'], df_test['rating']))


best_model = model.best_estimator['mae']
# predictions = best_model.test(test_triples)
# Saving x_test positive probabilities to a file

# reader2 = Reader(
#     line_format='user item rating', sep=',',
#     skip_lines=1)
# test_set = Dataset.load_from_file(
#     'data_movie_lens_100k/ratings_masked_leaderboard_set.csv', reader=reader2)

# test_set = test_set.build_full_trainset().build_testset()
yproba1_te = best_model.predict(test_triples)
np.savetxt("predicted_ratings_leaderboard.txt", yproba1_te, fmt="%.6f")

# print("global mean:")
# print(model.trainset.global_mean)
# print("shape of bias_per_item: ")
# print(model.bi.shape)
# print("shape of bias_per_user: ")
# print(model.bu.shape)
# print("shape of U (per user vectors): ")
# print(model.pu.shape)
# print("shape of V (per item vectors): ")
# print(model.qi.shape)
# y_predicted = model.predict(test_tuple[0])

    # print("Error: ", ag_np.mean(ag_np.abs(test_tuple[1] - y_predicted)))
