import numpy as np
import pandas as pd
import gym
import nest

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

    total_fitness = 0
    for i in range(100):
        env.reset()
        position_range = [-1.2, 0.6]
        velocity_range = [-0.07, 0.07] 
        position = np.random.uniform(position_range[0], position_range[1])
        velocity = np.random.uniform(velocity_range[0], velocity_range[1])
        env.env.state = np.array([position, velocity])
        max_position = -2.0
        for sim_step in range(1, 110):
            #env.render()
            test_network.reset_activators_and_recorders()
            test_network.feed_network(velocity, position)
            test_network.simulate(20.)
            action = test_network.get_action_from_network() 
            observation = test_network.set_gym_action(action, env)
            position = observation[0]
            velocity = observation[1]     
            if max_position < position:
                max_position = position
            if position >= 0.5:
                break
        fitness = max_position
        total_fitness += fitness
    return total_fitness/100


if __name__ == "__main__":
    weights = load_weights("individual_240_29_result.csv")
    print(simulate(weights))