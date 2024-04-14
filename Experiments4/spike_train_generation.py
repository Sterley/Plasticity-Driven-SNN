import matplotlib.pyplot as plt

biological_time = 0.0  # Temps biologique initial
input_t_offset = 1.0  # Décalage temporel avant le début du train de spike
ISI = 10 # Intervalle entre les spikes (ms)
N_INPUT_SPIKES = 20  # Nombre de spikes dans le train

input_train = [biological_time + input_t_offset + i * ISI for i in range(N_INPUT_SPIKES)]

plt.figure(figsize=(10, 2))
plt.eventplot(input_train, color='black')
plt.title('Train de Spike')
plt.xlabel('Temps (ms)')
plt.yticks([])  # Supprimer les ticks de l'axe des y pour une meilleure clarté
plt.show()
