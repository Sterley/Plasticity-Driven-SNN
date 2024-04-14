import numpy as np
import pandas as pd
import nest
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import gym

class NestNetwork():
    def __init__(self):
        nest.ResetKernel()
        nest.set_verbosity('M_ERROR')
        nest.SetKernelStatus(
            {
                'resolution': 0.2,
                'rng_seed': 1,
            })
        self.inpt_velocity = None
        self.inpt_position = None
        self.activator_velocity = None
        self.activator_position = None
        self.outpt = None
        self.nodes = None
        self.spike_detector = None
        self.create_nodes()


    def create_nodes(self):
        params = {'t_ref': 1.0}
        self.activator_velocity = nest.Create("dc_generator", 30, params={'amplitude': 40000.}) #40000.
        self.activator_position = nest.Create("dc_generator", 30, params={'amplitude': 40000.})
        self.inpt_velocity = nest.Create('iaf_psc_alpha', 30, params=params)
        self.inpt_position = nest.Create('iaf_psc_alpha', 30, params=params)
        self.nodes = nest.Create('iaf_psc_alpha', 5, params=params)
        self.outpt = nest.Create('iaf_psc_alpha', 3, params=params)
        self.spike_detector = nest.Create('spike_recorder', 3)

        self.spike_detector1 = nest.Create('spike_recorder', 5)


    def connect_network(self,weights):
        nest.Connect(self.activator_velocity, self.inpt_velocity, 'one_to_one')
        nest.Connect(self.activator_position, self.inpt_position, 'one_to_one')
        syn_spec_dict = {'weight': weights[0:150].reshape(5, 30)}
        nest.Connect(self.inpt_velocity, self.nodes, 'all_to_all', syn_spec=syn_spec_dict)
        syn_spec_dict = {'weight': weights[150:300].reshape(5, 30)}
        nest.Connect(self.inpt_position, self.nodes, 'all_to_all', syn_spec=syn_spec_dict)
        syn_spec_dict = {'weight': weights[300:315].reshape(3, 5)}
        nest.Connect(self.nodes, self.outpt, 'all_to_all', syn_spec=syn_spec_dict)
        nest.Connect(self.outpt, self.spike_detector, 'one_to_one')

        nest.Connect(self.nodes[0], self.spike_detector1[0])
        nest.Connect(self.nodes[1], self.spike_detector1[1])
        nest.Connect(self.nodes[2], self.spike_detector1[2])
        nest.Connect(self.nodes[3], self.spike_detector1[3])
        nest.Connect(self.nodes[4], self.spike_detector1[4])

    def simulate(self, sim_time=20.0):
        nest.Simulate(sim_time)
            
    def feed_network(self, velocity, position):
        velocity_neuron = int(encode_values(-0.07, 0.07, 30, velocity)) - 1
        position_neuron = int(encode_values(-1.2, 0.6, 30, position)) - 1
        nest.SetStatus(self.activator_velocity[velocity_neuron], [{'amplitude':40000.0}])            
        nest.SetStatus(self.activator_position[position_neuron], [{'amplitude':40000.0}])            

    def reset_activators_and_recorders(self):
        for n in range(0, 30):
            nest.SetStatus(self.activator_velocity[n], [{'amplitude':0.0}])            
            nest.SetStatus(self.activator_position[n], [{'amplitude':0.0}])            

        for n in range(0, 5):
            nest.SetStatus(self.spike_detector1[n], [{'n_events':0}])            
            nest.SetStatus(self.spike_detector1[n], [{'n_events':0}]) 

        for n in range(0, 3):
            nest.SetStatus(self.spike_detector[n], [{'n_events':0}])            
            nest.SetStatus(self.spike_detector[n], [{'n_events':0}])            

            
    def get_action_from_network(self):
        push_left = nest.GetStatus(self.spike_detector[0], 'n_events')
        push_none = nest.GetStatus(self.spike_detector[1], 'n_events')
        push_right = nest.GetStatus(self.spike_detector[2], 'n_events')
        if (push_left > push_right and push_left > push_none):
            return 0
        elif (push_right > push_left and push_right > push_none):
            return 2
        else:
            return 1

    def set_gym_action(self, action, env):
        observation, reward, done, info = env.step(action) 
        return observation


def encode_values(min_range, max_range, bins, value):
    if max_range == value:
        return bins
    x = np.linspace(min_range, max_range, bins)
    hist, edges = np.histogram(x, bins=bins)
    return np.digitize(value, edges)  
  
  
def load_weights(file_path):
    df = pd.read_csv(file_path, header=None)
    weights = df.iloc[0].values 
    return weights






def simulate(weights):
    test_weights = weights * 100 
    test_network = NestNetwork()
    test_network.connect_network(test_weights)  
    env = gym.make('MountainCar-v0')
    env.reset()
    position_range = [-1.2, 0.6]
    velocity_range = [-0.07, 0.07] 
    position = -0.5
    velocity = 0
    env.env.state = np.array([position, velocity])
    spikes_times = [[], []]
    for _ in range(3):
        spikes_times.append([])
    for sim_step in range(1, 110):
        env.render()
        test_network.reset_activators_and_recorders()
        spikes_times[0].append(position)
        spikes_times[1].append(velocity)
        test_network.feed_network(velocity, position)
        test_network.simulate(20.)
        for neuron in range(3):
            spikes = nest.GetStatus(test_network.spike_detector[neuron], 'events')[0]['times']
            spikes_times[neuron+2].extend(spikes)
        action = test_network.get_action_from_network() 
        observation = test_network.set_gym_action(action, env)
        position = observation[0]
        velocity = observation[1]     
        if position >= 0.5:
            break
    return spikes_times






def plot_spike_raster_with_position(spikes_times):
    num_neurons = len(spikes_times) - 2  
    fig, axes = plt.subplots(num_neurons + 2, 1, figsize=(12, 10), sharex=True)
    total_duration = max(max(spikes) for spikes in spikes_times[1:]) if num_neurons > 0 else len(spikes_times[0])


    position_times = np.linspace(0, total_duration, len(spikes_times[0]))
    axes[0].plot(position_times, spikes_times[0], label='Position')
    axes[0].set_title('Évolution de la Position')
    axes[0].set_ylabel('Position')
    axes[0].legend()

    axes[1].plot(position_times, spikes_times[1], label='Velocity')
    axes[1].set_title('Évolution de la Velocity')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()

    # Tracer les raster plots pour chaque neurone
    for i in range(num_neurons):
        axes[i + 2].eventplot(spikes_times[i + 2], lineoffsets=1, linelengths=0.5)
        axes[i + 2].set_title(f'Neurone {i + 1}')
        axes[i + 2].set_ylabel('Spikes')
        axes[i + 2].set_yticks([])

    axes[-1].set_xlabel('Temps (étapes de simulation)')
    plt.tight_layout()
    plt.show()






def plot_spike_frequency_with_position_velocity(spikes_times, window_size=5):
    num_neurons = len(spikes_times) - 2
    fig, axes = plt.subplots(num_neurons + 2, 1, figsize=(12, 12), sharex=True)
    total_duration = max(max(spikes) for spikes in spikes_times[2:]) if num_neurons > 0 else len(spikes_times[0])
    position_times = np.linspace(0, total_duration, len(spikes_times[0]))

    axes[0].plot(position_times, spikes_times[0], label='Position')
    axes[0].set_title('Position Evolution')
    axes[0].set_ylabel('Position')
    axes[0].legend()

    axes[1].plot(position_times, spikes_times[1], label='Velocity')
    axes[1].set_title('Velocity Evolution')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    num_windows = int(total_duration / window_size)
    for i in range(num_neurons):
        spikes = spikes_times[i + 2]
        frequency = [0] * num_windows

        for spike_time in spikes:
            window_index = int(spike_time / window_size)
            if window_index < num_windows:
                frequency[window_index] += 1

        frequency = [f / window_size for f in frequency] 
        axes[i + 2].plot(np.linspace(0, total_duration, num_windows), frequency)
        axes[i + 2].set_title(f'Spikes Frequency - Neuron {i + 1}')
        axes[i + 2].set_ylabel('Frequency (Hz)')

    axes[-1].set_xlabel('Simulation Times')
    plt.tight_layout()
    plt.show()





def generate_random_weights(num_weights):
    random_weights = np.random.uniform(-20, 20, num_weights)
    return random_weights



if __name__ == "__main__":
    weights = load_weights("./results/results/individual_240_29_result.csv")
    random_weights = generate_random_weights(315)
    spikes_times = simulate(weights)
    #plot_spike_raster_with_position(spikes_times)
    plot_spike_frequency_with_position_velocity(spikes_times)

