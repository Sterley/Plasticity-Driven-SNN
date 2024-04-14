import nest
import numpy as np
import gym
import nest
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


POLL_TIME = 200
N_INPUT_SPIKES = 20
ISI = 10.0
BG_STD = 220.0

class NestNetworkRSTDP():
    def __init__(self, apply_noise=True, num_neurons=60):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        self.apply_noise = apply_noise
        self.num_neurons = num_neurons
        self.mean_reward = np.array([0.0 for _ in range(self.num_neurons)])

        self.input_t_offset = 1
        self.learning_rate = 0.7
        self.stdp_amplitude = 36.0
        self.stdp_tau = 64.0
        self.stdp_saturation = 128
        self.mean_weight = 1300.0
        self.biological_time = 0

        self.create_nodes()
        self.connect_network()

    def create_nodes(self):
        self.input_generators = nest.Create("spike_generator", self.num_neurons)
        self.input_neurons = nest.Create("parrot_neuron", self.num_neurons)
        self.motor_neurons = nest.Create("iaf_psc_exp", 2)
        self.spike_recorders = nest.Create("spike_recorder", 2)
        self.background_generator = nest.Create("noise_generator", 2, params={"std": BG_STD})

    def connect_network(self):
        nest.Connect(self.input_generators, self.input_neurons, {"rule": "one_to_one"})
        nest.Connect(self.motor_neurons, self.spike_recorders, {"rule": "one_to_one"})
        nest.Connect(self.background_generator, self.motor_neurons, {"rule": "one_to_one"})
        nest.Connect(
            self.input_neurons,
            self.motor_neurons,
            {"rule": "all_to_all"},
            {"weight": nest.random.normal(self.mean_weight, 1)},
        )

    def simulate(self, sim_time):
        nest.Simulate(sim_time)
        self.biological_time = nest.GetKernelStatus("biological_time")
            
    def feed_network(self, input_value):
        self.input_index = int(self.encode_values(-5, 5, self.num_neurons, input_value)) - 1
        self.set_input_spiketrain(self.input_index, self.biological_time) 

    def get_all_weights(self):
        x_offset = self.input_neurons[0].get("global_id")
        y_offset = self.motor_neurons[0].get("global_id")
        weight_matrix = np.zeros((self.num_neurons, 2))
        conns = nest.GetConnections(self.input_neurons, self.motor_neurons)
        for conn in conns:
            source, target, weight = conn.get(["source", "target", "weight"]).values()
            weight_matrix[source - x_offset, target - y_offset] = weight
        return weight_matrix

    def set_all_weights(self, weights):
        for i in range(self.num_neurons):
            for j in range(2):
                connection = nest.GetConnections(self.input_neurons[i], self.motor_neurons[j])
                connection.set({"weight": weights[i, j]})

    def get_spike_counts(self):
        events = self.spike_recorders.get("n_events")
        return np.array(events)

    def reset(self):
        self.spike_recorders.set({"n_events": 0})

    def set_input_spiketrain(self, input_cell, biological_time):
        self.target_index = input_cell
        self.input_train = [biological_time + self.input_t_offset + i * ISI for i in range(N_INPUT_SPIKES)]
        self.input_train = [np.round(x, 1) for x in self.input_train]
        for input_neuron in range(self.num_neurons):
            nest.SetStatus(self.input_generators[input_neuron], {"spike_times": []})
        nest.SetStatus(self.input_generators[input_cell], {"spike_times": self.input_train})

    def get_max_activation(self):
        spikes = self.get_spike_counts()
        return int(np.random.choice(np.flatnonzero(spikes == spikes.max())))

    def calculate_reward(self, input_value, action):
        if input_value < 0 and action == 0:
            bare_reward = 1
        elif input_value > 0 and action == 1:
            bare_reward = 1
        else:
            bare_reward = -1
        reward = bare_reward - self.mean_reward[self.target_index]
        self.mean_reward[self.target_index] = float(self.mean_reward[self.target_index] + reward / 2.0)
        return bare_reward, reward

    def encode_values(self, min_range, max_range, bins, value):
        if max_range == value:
            return bins
        x = np.linspace(min_range, max_range, bins)
        hist, edges = np.histogram(x, bins=bins)
        return np.digitize(value, edges)   
    
    def apply_synaptic_plasticity(self, input_value, action):
        bare_reward, reward = self.calculate_reward(input_value, action)
        self.apply_rstdp(reward)
        return bare_reward, reward
    
    def apply_rstdp(self, reward):
        post_events = {}
        offset = self.motor_neurons[0].get("global_id")
        for index, event in enumerate(self.spike_recorders.get("events")):
            post_events[offset + index] = event["times"]
        for connection in nest.GetConnections(self.input_neurons[self.target_index]):
            motor_neuron = connection.get("target")
            motor_spikes = post_events[motor_neuron]
            correlation = self.calculate_stdp(self.input_train, motor_spikes)
            old_weight = connection.get("weight")
            new_weight = old_weight + self.learning_rate * correlation * reward
            connection.set({"weight": new_weight})

    def calculate_stdp(self, pre_spikes, post_spikes, only_causal=True, next_neighbor=True):
        pre_spikes, post_spikes = np.sort(pre_spikes), np.sort(post_spikes)
        facilitation = 0
        depression = 0
        positions = np.searchsorted(pre_spikes, post_spikes)
        last_position = -1
        for spike, position in zip(post_spikes, positions):
            if position == last_position and next_neighbor:
                continue  
            if position > 0:
                before_spike = pre_spikes[position - 1]
                facilitation += self.stdp_amplitude * np.exp(-(spike - before_spike) / self.stdp_tau)
            if position < len(pre_spikes):
                after_spike = pre_spikes[position]
                depression += self.stdp_amplitude * np.exp(-(after_spike - spike) / self.stdp_tau)
            last_position = position
        if only_causal:
            return min(facilitation, self.stdp_saturation)
        else:
            return min(facilitation - depression, self.stdp_saturation)
        
    def set_gym_action(self, action, env):
        observation, reward, done, info = env.step(action) 
        return observation, reward, done, info


def generate_weight_images(weights_evol, temp_folder='temp_images'):
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    filenames = []
    num_input_neurons = 60  
    num_motor_neurons = 2  
    for i, weights in enumerate(weights_evol):
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(weights, cmap='viridis')
        fig.colorbar(cax)
        ax.set_xlabel('Motor Neurons')
        ax.set_ylabel('Input Neurons')
        ax.set_xticks([0, num_motor_neurons-1])  
        ax.set_yticks([0, num_input_neurons-1]) 
        ax.set_xticklabels(['0', f'{num_motor_neurons-1}'])
        ax.set_yticklabels(['0', f'{num_input_neurons-1}'])
        ax.set_title(f'Weight Matrix at Step {i}')
        filename = f'{temp_folder}/step_{i}.png'
        plt.savefig(filename)
        plt.close()
        filenames.append(filename)
    return filenames


def create_gif(filenames, output_path='weights_evolution.gif'):
    with imageio.get_writer(output_path, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)



def simulate(max_runs=1000):  
    env = gym.make('CartPole-v1') 
    network = NestNetworkRSTDP() 
    seed_val = 10
    env.seed(seed_val)
    observation = env.reset()
    angle = observation[2]
    angular_velocity = observation[3]
    last_failed = 0
    mean_expected_reward_evol = []
    reward_evol = []
    real_reward_evol = []
    weights_evol = []
    for i in range(max_runs):
        env.render()
        network.reset()
        input_value = angle + angular_velocity
        network.feed_network(input_value)
        network.simulate(POLL_TIME)
        action = network.get_max_activation()
        observation, reward, done, info = network.set_gym_action(action, env)
        bare_reward, reward = network.apply_synaptic_plasticity(input_value, action)
        weights_evol.append(network.get_all_weights())
        reward_evol.append(bare_reward)
        mean_expected_reward_evol.append(np.mean(network.mean_reward))
        angle = observation[2]
        angular_velocity = observation[3]
        if done:
            print("Failed at step: ", i-last_failed)
            real_reward_evol.append(i-last_failed)
            last_failed = i
            seed_val = 10 + i
            env.seed(seed_val)
            observation = env.reset()
            angle = observation[2]
            angular_velocity = observation[3]
    env.close()
    return reward_evol, mean_expected_reward_evol, real_reward_evol, weights_evol

if __name__ == "__main__":
    reward_evol, mean_expected_reward_evol, real_reward_evol, weights_evol = simulate()
    plt.figure(figsize=(10, 5))
    plt.plot(reward_evol, label='Reward Evolution')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Evolution of Reward Over Time')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(mean_expected_reward_evol, label='Mean Expected Reward Evolution')
    plt.xlabel('Step')
    plt.ylabel('Mean Expected Reward')
    plt.title('Evolution of Mean Expected Reward Over Time')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(real_reward_evol, label='Real Reward Evolution')
    plt.xlabel('Simulation Index')
    plt.ylabel('Total Reward')
    plt.title('Evolution of Total Reward Over Time')
    plt.legend()
    plt.show()
    filenames = generate_weight_images(weights_evol)
    create_gif(filenames)
    print("GIF créé avec succès.")