import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = "/home/labady/Documents/PR/results/results/"
first_values = []
max_first_index = 241
max_second_index = 32
for i in range(0, max_first_index, 10): 
    best_fitness = -2
    for j in range(max_second_index):
        file_path = os.path.join(base_path, f"individual_{i}_{j}_result.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            if len(df) >= 2:
                first_value = df.iloc[1, 0]
                if first_value > best_fitness:
                    best_fitness = first_value
    first_values.append(best_fitness)

plt.figure(figsize=(10, 6))
plt.plot(first_values)
plt.title("Evolution of the Fitness Value !")
plt.xlabel("Generation (× 10¹)")
plt.ylabel("Value")
plt.grid(True)
plt.show()
