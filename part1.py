
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
        step_size=0.8,
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

fig, axes = plt.subplots(1, len(all_traces), figsize=(15, 4), sharey=True)

for idx, (K, epochs, rmse_train, rmse_valid) in enumerate(all_traces):
    ax = axes[idx]
    c = colors[idx]
    ax.plot(epochs, rmse_train, f'{c}.-', label='Train RMSE')
    ax.plot(epochs, rmse_valid, f'{c}--', label='Valid RMSE')
    ax.set_title(f'K = {K}')
    ax.set_xlabel('Epoch')
    if idx == 0:
        ax.set_ylabel('RMSE')
    ax.set_ylim([0,3])
    ax.legend()

fig.suptitle('Trace Plots of RMSE vs. Epoch (α = 0)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
