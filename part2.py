
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

from surprise import SVD,KNNBasic
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

train_set = Dataset.load_from_file(
    'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
df = pd.read_csv('data_movie_lens_100k/ratings_masked_leaderboard_set.csv')
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

DATA_PATH = 'data_movie_lens_100k/' # TODO fixme: Path to where dataset csv files live on your system
if not os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv')):
    try:
        DATA_PATH = os.path.join(os.environ.get("HOME", ""),
                    'courses/cs135-25s-staffonly/proj_src/projB/data_movie_lens_100k/')
        assert os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv'))
    except AssertionError:
        print("Please store path to movie_lens_100k dataset in DATA_PATH")

assert os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv'))



#NOTE: MOVIE
movie_df = pd.read_csv(os.path.join(DATA_PATH, 'movie_info.csv'))
print("MOVIE LOADED: ", movie_df.shape)

dev_set = Dataset.load_from_file(
   os.path.join(DATA_PATH, 'ratings_all_development_set.csv'), reader=reader)
dev_set_for_fit = dev_set.build_full_trainset()
dev_set_for_predict = dev_set_for_fit.build_testset()
dev_set_for_fit.global_mean

print("Length of prediction set: ",len(dev_set_for_predict))
print(dev_set_for_predict[0])

# out_test_set = Dataset.load_from_file(
#    os.path.join(DATA_PATH, 'ratings_masked_leaderboard_set.csv'), reader=reader)



##TRAINING##
param_grid = {
    "n_factors": [1,2,5,7,10,50,100],
    "lr_all": [0.0005,0.001,0.005,0.05]
    # "lr_all": [0.005]
}
model = model_selection.search.RandomizedSearchCV(SVDpp, param_grid, n_iter=28, measures=['mae'], refit=True, n_jobs=-1)
model.fit(train_set)

print("Best score:", model.best_score['mae'])
print("Best params:", model.best_params['mae'])
best_model = model.best_estimator['mae']

yproba1_te = best_model.test(dev_set_for_predict)

for i in range(15):
    print("predict: ",yproba1_te[i][2],yproba1_te[i][3])
    # print(yproba1_te)


# Convert cv_results_ to a DataFrame
results_df = pd.DataFrame(model.cv_results)

# Show just the params and corresponding mean MAE
print(results_df[['params', 'mean_test_mae']].sort_values(by='mean_test_mae'))

