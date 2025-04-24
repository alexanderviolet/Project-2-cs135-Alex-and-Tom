
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

# train_data = Dataset.load_from_file(
#     'data_movie_lens_100k/ratings_all_development_set.csv', reader=reader)
# # print(train_set[:5])
# train_set = train_data.build_full_trainset()
# raw_ratings = train_set.build_testset()


##MAKING MERGE DATA FRAME
# df = pd.read_csv('data_movie_lens_100k/ratings_all_development_set.csv')
# user_df = pd.read_csv('data_movie_lens_100k/user_info.csv')
# movie_df = pd.read_csv('data_movie_lens_100k/movie_info.csv')
# merged_df = pd.merge(df, user_df, on='user_id', how='inner')
# merged_df = merged_df.drop(columns=['orig_user_id'])
# merged_movie_df = pd.merge(merged_df, movie_df, on='item_id', how='inner')



# user_df = pd.read_csv('data_movie_lens_100k/user_info.csv')'
# print(merged_movie_df[:100])
# data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)



DATA_PATH = 'data_movie_lens_100k/' # TODO fixme: Path to where dataset csv files live on your system
if not os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv')):
    try:
        DATA_PATH = os.path.join(os.environ.get("HOME", ""),
                    'courses/cs135-25s-staffonly/proj_src/projB/data_movie_lens_100k/')
        assert os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv'))
    except AssertionError:
        print("Please store path to movie_lens_100k dataset in DATA_PATH")
assert os.path.exists(os.path.join(DATA_PATH, 'select_movies.csv'))


df_masked = pd.read_csv("data_movie_lens_100k/ratings_masked_leaderboard_set.csv")

df_masked['user_id'] = df_masked['user_id'].astype(str)
df_masked['item_id'] = df_masked['item_id'].astype(str)
df_masked['rating'] = 0


# Create a list of (uid, iid) pairs
predict_pairs = list(zip(df_masked['user_id'], df_masked['item_id'],df_masked['rating']))


dev_set = Dataset.load_from_file(
   os.path.join(DATA_PATH, 'ratings_all_development_set.csv'), reader=reader)
dev_set_for_fit = dev_set.build_full_trainset()
dev_set_for_predict = dev_set_for_fit.build_testset()
dev_set_for_fit.global_mean

print("Global Mean: ", dev_set_for_fit.global_mean)

# print("Length of prediction set: ",len(dev_set_for_predict))
# print(dev_set_for_predict[0])

# out_test_set = Dataset.load_from_file(
#    os.path.join(DATA_PATH, 'ratings_masked_leaderboard_set.csv'), reader=reader)



##TRAINING##---------------------------------------

# param_grid = {
#     # "n_factors": [1,2,5,7,10,50,100],
#     # "lr_all": [0.0005,0.001,0.005,0.05]
#     "n_factors": [50],
#     "lr_all": [0.005]
# }

# param_grid = {
#     # "n_factors": [20, 50, 100],        # Latent dimensions (embedding size)
#     # "lr_all": [0.002, 0.005, 0.01],    # Learning rate for all parameters
#     # "reg_all": [0.02, 0.05, 0.1],      # Regularization term for all parameters
#     # "n_epochs": [10, 20, 30]
#     "n_factors": [100],        # Latent dimensions (embedding size)
#     "lr_all": [0.01],    # Learning rate for all parameters
#     "reg_all": [0.1],      # Regularization term for all parameters
#     "n_epochs": [30]          # Number of SGD iterations
# }
# model = model_selection.search.RandomizedSearchCV(SVDpp, param_grid, n_iter=1, measures=['mae'], refit=True, n_jobs=-1)
# model.fit(dev_set) 
# print("Lowest MAE: ", model.best_score['mae'])
# print("Best params:", model.best_params['mae'])
# best_model = model.best_estimator['mae']
# best_model.fit(dev_set_for_fit)

##TRAIN AND FIT
best_model = SVDpp(n_factors=100, lr_all=0.01, reg_all=0.1, n_epochs=30)
best_model.fit(dev_set_for_fit)


predictions = [best_model.predict(uid, iid) for (uid, iid, _) in predict_pairs]
for i in range(25):
    print(predictions[i][3])
# print("Type is: :",predictions[0].type())
# predictions = np.array(predictions, dtype=float)
ratings_only = [pred[3] for pred in predictions]  # extract just the ratings
ratings_only = np.array(ratings_only, dtype=float)
np.savetxt("predicted_ratings_leaderboard.txt", ratings_only, fmt="%.6f")

# np.savetxt("predicted_ratings_leaderboard.txt", predictions[:][3])


# yproba1_te = best_model.test(dev_set_for_predict)

# print(0 in dev_set_for_fit._raw2inner_id_users)
# print(113 in dev_set_for_fit._raw2inner_id_items)
# for i in range(25):
#     print(yproba1_te[i][2],round(yproba1_te[i][3]))
# print(yproba1_te)


# Convert cv_results_ to a DataFrame
# results_df = pd.DataFrame(model.cv_results)


# Show just the params and corresponding mean MAE
# print(results_df[['params', 'mean_test_mae']].sort_values(by='mean_test_mae'))

# Global Mean:  3.529480398257623
# ('772', '36', 3.0)
# 3.0 3
# 4.0 4
# 3.0 3
# 5.0 4
# 3.0 3
# 2.0 3
# 4.0 4
# 2.0 3
# 4.0 4
# 1.0 2
# 5.0 4
# 3.0 4
# 3.0 4
# 4.0 4
# 3.0 3
# 3.0 3
# 4.0 4
# 3.0 4
# 2.0 2
# 4.0 4
# 4.0 3
# 2.0 2
# 1.0 2
# 2.0 2
# 4.0 3