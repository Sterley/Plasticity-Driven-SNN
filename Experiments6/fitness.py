import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = "/home/labady/Documents/PR5/results/"
first_values = []
nb_generations = 1000
nb_individuals = 96
for i in range(950, nb_generations): 
    if i%10 == 0:
        print(f"Generation : {i}")
    for j in range(nb_individuals):
        file_path = os.path.join(base_path, f"individual_{i}_{j}_result.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            if len(df) >= 2:
                first_value = df.iloc[2, 0]
                if first_value >= 0.1:
                    print(f"individual_{i}_{j}_result.csv", first_value)
                #first_values.append(first_value)

#plt.figure(figsize=(10, 6))
#plt.plot(first_values)
#plt.title("Evolution of the Fitness value !")
#plt.xlabel("File Index")
#plt.ylabel("Value")
#plt.grid(True)
#plt.show()
