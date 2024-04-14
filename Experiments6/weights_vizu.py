import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    for ax in axs.ravel():
        ax.clear()

def update(step):
    for layer_idx, (layer, steps_weights) in enumerate(weights_by_layer_and_step.items()):
        matrix = np.zeros((max_i[layer_idx], max_j[layer_idx]))
        for (i, j), weight in steps_weights[step].items():
            matrix[i, j] = weight
        axs[layer_idx].clear()
        axs[layer_idx].imshow(matrix, cmap='viridis', aspect='auto')
        axs[layer_idx].set_title(f'{layer} - Step {step}')
    plt.tight_layout()

folder_path = './save_weights/weights_1'
weights_by_layer_and_step = load_weights(folder_path)


max_i, max_j = [], []
for _, steps_weights in weights_by_layer_and_step.items():
    max_i.append(max(max((i for i, _ in step.keys())) for step in steps_weights if step) + 1)
    max_j.append(max(max((j for _, j in step.keys())) for step in steps_weights if step) + 1)

n_layers = len(weights_by_layer_and_step)
fig, axs = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5))

n_steps = max(len(steps) for steps in weights_by_layer_and_step.values())
ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, interval=1, repeat=False)

plt.show()
