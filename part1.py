# import CollabFilterOneVectorPerItem
# from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD

# from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
# from train_valid_test_loader import load_train_valid_test_datasets


# # Import plotting libraries
# import matplotlib
# import matplotlib.pyplot as plt

# plt.style.use('seaborn-v0_8') # pretty matplotlib plots


# # K value array for training models
# K_V = [2, 10, 50]
# colors = ['b', 'g', 'm']
# # Load Data
# train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
# all_traces = []
# # Train model
# for i, K in enumerate([2,10,50]):
#     print("K value: ", K)

#     model = CollabFilterOneVectorPerItem(
#         n_epochs=10, 
#         batch_size=1000, 
#         step_size=0.1,
#         n_factors=K,
#         alpha=0)

#     model.init_parameter_dict(n_users, n_items, train_tuple)

#     # Fit the model with SGD
#     model.fit(train_tuple, valid_tuple)
#     all_traces.append((i, model.trace_epoch, model.trace_rmse_train, model.trace_rmse_valid))

# # model = CollabFilterOneVectorPerItem(
# #     n_epochs=10, 
# #     batch_size=1000, 
# #     step_size=0.1,
# #     n_factors=500,
# #     alpha=0)

# # model.init_parameter_dict(n_users, n_items, train_tuple)

# # # Fit the model with SGD
# # model.fit(train_tuple, valid_tuple)
# # all_traces.append((0, model.trace_epoch, model.trace_rmse_train, model.trace_rmse_valid))
    

    
# # fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)


# # 1) Create one figure with 3 subplots
# fig, axes = plt.subplots(1, len(all_traces), figsize=(15, 4), sharey=True)

# # 2) Loop over each (K, epochs, train, valid) and plot into its own axis
# for idx, (K, epochs, rmse_train, rmse_valid) in enumerate(all_traces):
#     ax = axes[idx]
#     ax.plot(epochs, rmse_train, 'b.-', label='Train RMSE')
#     ax.plot(epochs, rmse_valid, 'b--', label='Valid RMSE')
#     ax.set_title(f'K = {6969}')
#     ax.set_xlabel('Epoch')
#     if idx == 0:
#         ax.set_ylabel('RMSE')
#     ax.set_ylim([1.0, 1.5])
#     ax.legend()
    


# # 3) Tidy up and show
# fig.suptitle('Trace Plots of RMSE vs. Epoch (α = 0)')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # K, epochs, rmse_train, rmse_valid = all_traces[0]

# # # make a single figure
# # plt.figure(figsize=(6,4))

# # # plot train & valid
# # plt.plot(epochs, rmse_train, 'b.-', label='Train RMSE')
# # plt.plot(epochs, rmse_valid, 'b--', label='Valid RMSE')

# # # labels & title
# # plt.title(f'RMSE vs Epoch (K = {K}, α = 0)')
# # plt.xlabel('Epoch')
# # plt.ylabel('RMSE')
# # plt.ylim([1.0, 1.5])
# # plt.legend()

# # plt.tight_layout()
# # plt.show()




import matplotlib.pyplot as plt
from CollabFilterOneVectorPerItem import CollabFilterOneVectorPerItem
from train_valid_test_loader import load_train_valid_test_datasets

plt.style.use('seaborn-v0_8')

# Load Data
train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()

# Hyperparameters
K_values = [2, 10, 50]
colors   = ['b', 'g', 'm']

all_traces = []
for idx, K in enumerate(K_values):
    print("Training K =", K)
    model = CollabFilterOneVectorPerItem(
        n_epochs=1000,
        batch_size=1000,
        step_size=0.5,
        n_factors=K,
        alpha=0
    )
    model.init_parameter_dict(n_users, n_items, train_tuple)
    model.fit(train_tuple, valid_tuple)

    # *** FIX: store K, not idx ***
    all_traces.append((
        K,
        model.trace_epoch,
        model.trace_rmse_train,
        model.trace_rmse_valid
    ))

# Plot 3‑panel figure
fig, axes = plt.subplots(1, len(all_traces), figsize=(15, 4), sharey=True)

for idx, (K, epochs, rmse_train, rmse_valid) in enumerate(all_traces):
    ax = axes[idx]
    c = colors[idx]
    ax.plot(epochs,    rmse_train, f'{c}.-', label='Train RMSE')
    ax.plot(epochs,    rmse_valid, f'{c}--', label='Valid RMSE')
    ax.set_title(f'K = {K}')
    ax.set_xlabel('Epoch')
    if idx == 0:
        ax.set_ylabel('RMSE')
    ax.set_ylim([1.0, 1.5])
    ax.legend()

fig.suptitle('Trace Plots of RMSE vs. Epoch (α = 0)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
