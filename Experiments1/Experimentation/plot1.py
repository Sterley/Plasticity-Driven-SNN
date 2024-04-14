import matplotlib.pyplot as plt
import gym
import numpy as np


env = gym.make('MountainCar-v0')


positions = []
velocities = []
times = []
rewards = []
actions = []
cumulative_rewards = [0] 


env.reset()
for t in range(200):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    positions.append(observation[0])
    velocities.append(observation[1])
    times.append(t)
    rewards.append(reward)
    actions.append(action)
    cumulative_rewards.append(cumulative_rewards[-1] + reward)
    if done:
        break


cumulative_rewards.pop()


env.close()


fig, axs = plt.subplots(2, 3, figsize=(18, 10))


sc1 = axs[0, 0].scatter(positions, velocities, c=times, cmap='viridis')
plt.colorbar(sc1, ax=axs[0, 0])
axs[0, 0].set_title("Vitesse - Position (Temps en couleur)")
axs[0, 0].set_xlabel("Position")
axs[0, 0].set_ylabel("Vitesse")


sc2 = axs[0, 1].scatter(positions, velocities, c=cumulative_rewards, cmap='plasma')
plt.colorbar(sc2, ax=axs[0, 1])
axs[0, 1].set_title("Vitesse - Position (Récompense en couleur)")
axs[0, 1].set_xlabel("Position")
axs[0, 1].set_ylabel("Vitesse")


sc3 = axs[0, 2].scatter(positions, velocities, c=actions, cmap='coolwarm')
plt.colorbar(sc3, ax=axs[0, 2])
axs[0, 2].set_title("Vitesse - Position (Action en couleur)")
axs[0, 2].set_xlabel("Position")
axs[0, 2].set_ylabel("Vitesse")


sc4 = axs[1, 0].scatter(positions, times, c=cumulative_rewards, cmap='magma')
plt.colorbar(sc4, ax=axs[1, 0])
axs[1, 0].set_title("Position - Temps (Récompense en couleur)")
axs[1, 0].set_xlabel("Position")
axs[1, 0].set_ylabel("Temps")


sc5 = axs[1, 1].scatter(positions, times, c=actions, cmap='spring')
plt.colorbar(sc5, ax=axs[1, 1])
axs[1, 1].set_title("Position - Temps (Action en couleur)")
axs[1, 1].set_xlabel("Position")
axs[1, 1].set_ylabel("Temps")


sc6 = axs[1, 2].scatter(times, cumulative_rewards, c=actions, cmap='autumn')
plt.colorbar(sc6, ax=axs[1, 2])
axs[1, 2].set_title("Temps - Récompense (Action en couleur)")
axs[1, 2].set_xlabel("Temps")
axs[1, 2].set_ylabel("Récompense")


plt.tight_layout()
plt.show()
