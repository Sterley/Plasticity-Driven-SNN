import pandas as pd
import matplotlib.pyplot as plt
import os
import concurrent.futures

base_path = "/home/labady/Documents/PR5/results/"

def process_files(i, max_second_index):
    print(i)
    best_fitness = -2
    for j in range(max_second_index):
        file_path = os.path.join(base_path, f"individual_{i}_{j}_result.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            if len(df) >= 2:
                first_value = df.iloc[2, 0]
                if first_value > best_fitness:
                    best_fitness = first_value
    return best_fitness

max_first_index = 1000
max_second_index = 96

# Utilisation d'un ProcessPoolExecutor pour paralléliser les calculs
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Création d'une liste de tous les travaux à exécuter
    futures = [executor.submit(process_files, i, max_second_index) for i in range(0, max_first_index, 5)]
    # Récupération des résultats dès qu'ils sont terminés
    first_values = [future.result() for future in concurrent.futures.as_completed(futures)]

plt.figure(figsize=(10, 6))
plt.plot(first_values)
plt.title("Evolution of the Fitness Value !")
plt.xlabel("Generation (x 5¹)")
plt.ylabel("Value")
plt.grid(True)
plt.show()
