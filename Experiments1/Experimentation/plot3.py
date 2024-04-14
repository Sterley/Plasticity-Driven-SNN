import matplotlib.pyplot as plt
import numpy as np


position_range = np.linspace(-1.2, 0.6, 100) 
velocity_range = np.linspace(-0.07, 0.07, 100)  


goal_position = 0.5


k = 10


metric_grid = np.zeros((len(position_range), len(velocity_range)))

for i, pos in enumerate(position_range):
    for j, vel in enumerate(velocity_range):
        
        metric_grid[i, j] = np.sqrt((goal_position - pos)**2 + (k * vel)**2)


plt.figure(figsize=(10, 6))
cp = plt.contourf(position_range, velocity_range, metric_grid.T, cmap='viridis')
plt.colorbar(cp)
plt.title("Métrique de Distance Euclidienne Pondérée")
plt.xlabel("Position")
plt.ylabel("Vitesse")
plt.grid(True)
plt.show()
