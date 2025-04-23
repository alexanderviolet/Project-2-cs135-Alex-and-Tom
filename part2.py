
import matplotlib.pyplot as plt
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

import autograd
import autograd.numpy as ag_np
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import sklearn

from surprise import SVD,SVDpp,KNN
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from surprise import model_selection

from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

# ------------------------------------------------------

plt.style.use('seaborn-v0_8')


reader = Reader(
    line_format='user item rating', sep=',',
    rating_scale=(1, 5), skip_lines=1)



##LOADING DATA##
train_set = Dataset.load_from_file(
    'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)



df = pd.read_csv('data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)


##TRAINING##
param_grid = {
    "n_factors": [1,10,50,100],
    "lr_all": [0.1]
}
model = model_selection.search.RandomizedSearchCV(KNN, param_grid, n_iter=4, measures=['mae'], refit=True, n_jobs=-1)
model.fit(train_set)
print("Best score:", model.best_score['mae'])
print("Best params:", model.best_params['mae'])
best_model = model.best_estimator['mae']



# test_set = list(zip(test_df['user_id'], test_df['item_id'], [0.0] * len(test_df)))

yproba1_te = best_model.predict(94,384)
print(yproba1_te)
