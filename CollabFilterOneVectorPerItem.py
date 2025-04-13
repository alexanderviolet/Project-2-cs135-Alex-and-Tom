'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.ones(1), 
            b_per_user= ag_np.ones(self.n_factors), # FIX dimensionality
            c_per_item= ag_np.ones(self.n_factors), # FIX dimensionality
            U=0.001 * random_state.randn(self.n_factors), # FIX dimensionality
            V=0.001 * random_state.randn(self.n_factors), # FIX dimensionality
            )
        
        # For testing
        print("param array: ", self.param_dict)


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
           +r_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        N = user_id_N.size
        yhat_N = ag_np.ones(N)
        
        yhat_indexable = yhat_N.shape
        print("userID: ", user_id_N.shape)
        print("itemID: ", item_id_N.shape)
        
        U_user_NF = U[user_id_N]           # shape (N, F)
        V_item_NF = V[item_id_N]
        # Loop through user ids considered and append each prediction to yhat
        # for n, user_id in enumerate(user_id_N):
        #     print("b_per_user at index: ",b_per_user[user_id], "   user id: ", user_id)
        #     # predict for some element yhat_N at i,j (note: we're going off different indexing here)
        #     yhat_indexable[n] = mu + b_per_user[user_id] + c_per_item[item_id_N[n]] + (U[user_id].T * V[item_id_N[n]])
        # dot_N = ag_np.sum(U_user_NF * V_item_NF, axis=1)
        yhat_N = mu + b_per_user + c_per_item + ag_np.sum(U_user_NF * V_item_NF, axis=1)

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        U = param_dict['U']
        V = param_dict['V']
        user_id_N, item_id_N, y_N = data_tuple
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        # y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        # # loss_total = ag_np.mean(ag_np.sum(ag_np.absolute(y_N - yhat_N)))
        loss_total = self.alpha * (ag_np.sum(V ** 2) + ag_np.sum(U ** 2)) + (ag_np.sum(yhat_N - self.param_dict.['mu'] - b_per_user[]))

        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    print("MAIN IS CALLED YURRRR\n")
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, batch_size=10000, step_size=0.1,
        n_factors=2, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)
