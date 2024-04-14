import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = "/home/labady/Documents/PR1/results/results/"
first_values = []
max_first_index = 80
max_second_index = 31 
for i in range(max_first_index + 1): 
    for j in range(max_second_index + 1):
        file_path = os.path.join(base_path, f"individual_{i}_{j}_result.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            if len(df) >= 2:
                first_value = df.iloc[1, 0]
                first_values.append(first_value)

plt.figure(figsize=(10, 6))
plt.plot(first_values)
plt.title("Evolution of the Fitness value !")
plt.xlabel("Individual Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
