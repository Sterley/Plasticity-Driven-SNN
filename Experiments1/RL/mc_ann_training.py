import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MountainCarTrain:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.replay_buffer = deque(maxlen=20000)
        self.batch_size = 32
        self.episodes = 1000
        self.iterationNum = 201
        self.train_network = DQN(self.state_size, self.action_size).to(device)
        self.target_network = DQN(self.state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.optimizer = optim.Adam(self.train_network.parameters(), lr=self.learning_rate)

    def get_best_action(self, state, test=False):
        if not test:
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.action_size)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(device)
            state = state.unsqueeze(0)  
            start_time = time.time()
            action = self.train_network(state).max(1)[1].item()
            end_time = time.time()
            #print(end_time-start_time, "secondes")
        return action



    def train_from_buffer(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = np.array(states)
        next_states = np.array(next_states)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        q_values = self.train_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        total_reward_ref = -200
        for episode in range(self.episodes):
            state = self.env.reset()
            total_reward = 0
            for t in range(self.iterationNum):
                action = self.get_best_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                self.train_from_buffer()
                if done:
                    break
            self.target_network.load_state_dict(self.train_network.state_dict())
            self.epsilon -= self.epsilon_decay
            print(f'Episode: {episode}, Total reward: {total_reward}, Epsilon: {self.epsilon}')
            if total_reward > total_reward_ref:
                torch.save(self.train_network.state_dict(), f'model_{episode}.pth')
                total_reward_ref = total_reward

    def test_model(self, model_path, num_simulations=100):
        self.train_network.load_state_dict(torch.load(model_path))
        self.train_network.eval()  
        total_steps_list = []
        for _ in range(num_simulations):
            state = self.env.reset()
            steps = 0
            done = False
            while not done:
                action = self.get_best_action(state, test=True)  
                state, _, done, _ = self.env.step(action)
                steps += 1
            total_steps_list.append(steps)
        average_steps = np.mean(total_steps_list)
        print(f"Moyenne des étapes sur {num_simulations} simulations: {average_steps}")


env = gym.make('MountainCar-v0')
trainer = MountainCarTrain(env)
#trainer.train()

model_path = './BestModels/model.pth'
trainer.test_model(model_path)
