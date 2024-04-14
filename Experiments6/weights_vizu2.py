import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

def load_weights(folder_path):
    weights_by_layer_and_step = {}
    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1]))
    for file_name in files:
        step = int(file_name.split('_')[-1])
        with open(os.path.join(folder_path, file_name), 'r') as file:
            for line in file:
                layer, i, j, weight = line.strip().split(',')
                if layer not in weights_by_layer_and_step:
                    weights_by_layer_and_step[layer] = []
                i, j, weight = int(i), int(j), float(weight)
                if len(weights_by_layer_and_step[layer]) <= step:
                    weights_by_layer_and_step[layer].append({})
                if (i, j) not in weights_by_layer_and_step[layer][step]:
                    weights_by_layer_and_step[layer][step][(i, j)] = weight
    return weights_by_layer_and_step



def init():
    for ax, cbar in zip(axs, cbar_list):
        if cbar: cbar.remove()
    cbar_list[:] = [None] * n_layers

def update(step):
    for layer_idx, (layer, steps_weights) in enumerate(weights_by_layer_and_step.items()):
        ax = axs[layer_idx]
        matrix = np.zeros((max_i[layer_idx], max_j[layer_idx]))
        for (i, j), weight in steps_weights[step].items():
            matrix[i, j] = weight
        ax.clear()
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(f'{layer} - Step {step}')
        if cbar_list[layer_idx] is None:  # Create colorbar if it does not exist
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            cbar_list[layer_idx] = cbar
        else:  # Update colorbar limits to match the new data
            cbar_list[layer_idx].mappable.set_clim(vmin=np.min(matrix), vmax=np.max(matrix))
            cbar_list[layer_idx].draw_all()
    plt.tight_layout()

folder_path = './save_weights/weights_2'
weights_by_layer_and_step = load_weights(folder_path)

# Determine max_i and max_j for all layers and steps
max_i = {}
max_j = {}
for layer, steps_weights in weights_by_layer_and_step.items():
    max_i[layer] = max(max((i for i, _ in step.keys())) for step in steps_weights if step) + 1
    max_j[layer] = max(max((j for _, j in step.keys())) for step in steps_weights if step) + 1

# Convert to list for indexed access in the update function
max_i = [max_i[layer] for layer in weights_by_layer_and_step]
max_j = [max_j[layer] for layer in weights_by_layer_and_step]

n_layers = len(weights_by_layer_and_step)
fig, axs = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5))
cbar_list = [None] * n_layers  # Initialize the list with placeholders

n_steps = max(len(steps) for steps in weights_by_layer_and_step.values())
ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, interval=1, repeat=False)

plt.show()

