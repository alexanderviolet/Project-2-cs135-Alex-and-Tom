import CollabFilterOneVectorPerItem
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD

from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets


# Import plotting libraries
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8') # pretty matplotlib plots


# K value array for training models
K = [2, 10, 50]
colors = ['b', 'g', 'm']
# Load Data
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
all_traces = []
# Train model
for i in K:  
    print("K value: ", i)
    
    model = CollabFilterOneVectorPerItem(
        n_epochs=10, 
        batch_size=1000, 
        step_size=0.1,
        n_factors=i,
        alpha=500)

    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)
    all_traces.append((K, model.trace_epoch, model.trace_rmse_train, model.trace_rmse_valid))
    
for idx, (K, epochs, rmse_train, rmse_valid) in enumerate(all_traces):
    plt.plot(epochs, rmse_train, f'{colors[idx]}.-', label=f'Train RMSE (K={K})')
    plt.plot(epochs, rmse_valid, f'{colors[idx]}--', label=f'Valid RMSE (K={K})')

plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend(loc='lower right')
plt.ylim([0, 3])
plt.title('SGD Training RMSE (α=0)')
plt.show()