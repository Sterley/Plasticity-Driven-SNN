import gym
import numpy as np

# Initialisation de l'environnement
env = gym.make('CartPole-v1')

# Variables pour stocker les valeurs maximales et minimales
min_cart_velocity = float('inf')
max_cart_velocity = float('-inf')
min_pole_angular_velocity = float('inf')
max_pole_angular_velocity = float('-inf')

# Nombre d'épisodes pour la collecte de données
num_episodes = 1000000

for episode in range(num_episodes):
    if episode % 100000 == 0:
        print("Episode {} sur {}".format(episode, num_episodes))
    observation = env.reset()
    done = False

    while not done:
        # Ici, nous choisissons une action aléatoire
        action = env.action_space.sample()

        # Exécution d'une étape de l'environnement
        observation, reward, done, info = env.step(action)

        # Observation[1] est la vitesse du chariot, Observation[3] est la vitesse angulaire du pendule
        cart_velocity = observation[1]
        pole_angular_velocity = observation[3]

        # Mise à jour des valeurs maximales et minimales
        min_cart_velocity = min(min_cart_velocity, cart_velocity)
        max_cart_velocity = max(max_cart_velocity, cart_velocity)
        min_pole_angular_velocity = min(min_pole_angular_velocity, pole_angular_velocity)
        max_pole_angular_velocity = max(max_pole_angular_velocity, pole_angular_velocity)

# Affichage des résultats
print("Intervalle de vitesse du chariot : [{}, {}]".format(min_cart_velocity, max_cart_velocity))
print("Intervalle de vitesse angulaire du pendule : [{}, {}]".format(min_pole_angular_velocity, max_pole_angular_velocity))
