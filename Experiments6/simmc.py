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
        self.spike_detector_f = None
        self.inpt_velocity_f = None
        self.inpt_position_f = None
        self.outpt_f = None
        self.nodes_f = None
        self.create_nodes()


    def create_nodes(self):
        params = {'t_ref': 1.0}
        self.activator_velocity = nest.Create("dc_generator", 30, params={'amplitude': 40000.}) #40000.
        self.activator_position = nest.Create("dc_generator", 30, params={'amplitude': 40000.})
        self.inpt_velocity = nest.Create('iaf_psc_alpha', 30, params=params)
        neuron_params = nest.GetStatus(self.inpt_velocity)[0]  # Get parameters of the first neuron
        print(neuron_params)
        self.inpt_position = nest.Create('iaf_psc_alpha', 30, params=params)
        self.nodes = nest.Create('iaf_psc_alpha', 30, params=params)
        self.outpt = nest.Create('iaf_psc_alpha', 3, params=params)
        self.inpt_velocity_f = nest.Create('iaf_psc_alpha', 30, params=params)
        self.inpt_position_f = nest.Create('iaf_psc_alpha', 30, params=params)
        self.nodes_f = nest.Create('iaf_psc_alpha', 30, params=params)
        self.outpt_f = nest.Create('iaf_psc_alpha', 3, params=params)
        self.spike_detector = nest.Create('spike_recorder', 3)
        self.spike_detector_f = nest.Create('spike_recorder', 3)


    def connect_network(self, weights, plasticity_flags):
        nest.CopyModel("stdp_synapse", "custom_stdp_synapse", {
            "Wmax": 3000.0,
        })

        nest.Connect(self.activator_velocity, self.inpt_velocity, 'one_to_one')
        nest.Connect(self.activator_position, self.inpt_position, 'one_to_one')

        start_idx = 0
        for (source, target, source_size, target_size, num_conn) in [
            (self.inpt_velocity, self.nodes, 30, 30, 900),
            (self.inpt_position, self.nodes, 30, 30, 900),
            (self.nodes, self.nodes, 30, 30, 900),
            (self.nodes, self.outpt, 30, 3, 90)]:
            weights_tmp = weights[start_idx:start_idx+num_conn].reshape(source_size, target_size)
            plasticity_flags_tmp = plasticity_flags[start_idx:start_idx+num_conn].reshape(source_size, target_size)
            for i in range(len(source)):
                for j in range(len(target)):
                    syn_spec = {
                        "synapse_model": "custom_stdp_synapse" if plasticity_flags_tmp[i, j]>=0 else "static_synapse",
                        "weight": weights_tmp[i, j]
                    }
                    nest.Connect(source[i], target[j], conn_spec={"rule": "all_to_all", "allow_autapses": False}, syn_spec=syn_spec)
            start_idx += num_conn
            
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
            nest.SetStatus(self.spike_detector_f[n], [{'n_events':0}])  
            nest.SetStatus(self.spike_detector_f[n], [{'n_events':0}])         

            
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
        
    def get_action_from_network_f(self):
        push_left = nest.GetStatus(self.spike_detector_f[0], 'n_events')
        push_none = nest.GetStatus(self.spike_detector_f[1], 'n_events')
        push_right = nest.GetStatus(self.spike_detector_f[2], 'n_events')
        if (push_left > push_right and push_left > push_none):
            return 0
        elif (push_right > push_left and push_right > push_none):
            return 2
        else:
            return 1

    def set_gym_action(self, action, env):
        observation, reward, done, info = env.step(action) 
        return observation
    
    def save_network_weights(self, file_path):
        with open(file_path, 'w') as file:
            start_idx = 0
            for (name, source, target, source_size, target_size, num_conn) in [
                ("inpt_velocity_to_nodes", self.inpt_velocity, self.nodes, 30, 30, 900),
                ("inpt_position_to_nodes", self.inpt_position, self.nodes, 30, 30, 900),
                ("nodes_to_nodes", self.nodes, self.nodes, 30, 30, 900),
                ("nodes_to_outpt", self.nodes, self.outpt, 30, 3, 90)]:
                for i in range(len(source)):
                    for j in range(len(target)):
                            conn_details = nest.GetConnections(source[i], target[j])
                            weight = nest.GetStatus(conn_details, "weight")
                            if len(weight) == 1:
                                weight = weight[0]
                            else:
                                weight = 0
                            file.write(f"{name},{i},{j},{weight}\n")
                start_idx += num_conn

    def freeze(self):
        nest.Connect(self.activator_velocity, self.inpt_velocity_f, 'one_to_one')
        nest.Connect(self.activator_position, self.inpt_position_f, 'one_to_one')
        start_idx = 0
        for (source, target, source_p, target_p, num_conn) in [
            (self.inpt_velocity_f, self.nodes_f, self.inpt_velocity, self.nodes, 900),
            (self.inpt_position_f, self.nodes_f, self.inpt_position, self.nodes, 900),
            (self.nodes_f, self.nodes_f, self.nodes, self.nodes, 900),
            (self.nodes_f, self.outpt_f, self.nodes, self.outpt, 90)]:
            for i in range(len(source)):
                for j in range(len(target)):
                    conn_details = nest.GetConnections(source_p[i], target_p[j])
                    weight = nest.GetStatus(conn_details, "weight")
                    if len(weight) == 1:
                        weight = weight[0]
                    else:
                        weight = 0
                    syn_spec = {
                        "synapse_model": "static_synapse",
                        "weight": weight
                    }
                    nest.Connect(source[i], target[j], conn_spec={"rule": "all_to_all", "allow_autapses": False}, syn_spec=syn_spec)
            start_idx += num_conn
        nest.Connect(self.outpt_f, self.spike_detector_f, 'one_to_one')







def encode_values(min_range, max_range, bins, value):
    if max_range == value:
        return bins
    x = np.linspace(min_range, max_range, bins)
    hist, edges = np.histogram(x, bins=bins)
    return np.digitize(value, edges)  
  

def load_individual(file_path):
    df = pd.read_csv(file_path, header=None)
    weights = df.iloc[0].values 
    plasticity_flags = df.iloc[1].values 
    return weights, plasticity_flags


def simulate(weights, plasticity_flags): 
    stdp_synapses = 0    
    for p in plasticity_flags:
        if p >= 0:
            stdp_synapses += 1
    print(f"Plastic synapses : {stdp_synapses}")
    test_weights = weights * 100 
    test_plasticity_flags = plasticity_flags
    test_network = NestNetwork()
    test_network.connect_network(test_weights, test_plasticity_flags)  
    total_fitness = 0
    for i in range(3):
        continue
        #test_network.save_network_weights(f"./weights_{i}/weights_step_{0}")
        #test_network.save_network_weights(f"./weights_{i}/weights_step_{1}")
        env = gym.make('MountainCar-v0')
        seed_val = 965 * 1000 + 0 * 1000
        env.seed(seed_val)
        observation = env.reset()
        position = observation[0]
        velocity = observation[1]   
        max_position = -1.2
        for sim_step in range(0, 600):
            env.render()
            test_network.reset_activators_and_recorders()
            test_network.feed_network(velocity, position)
            test_network.simulate(20.)
            #test_network.save_network_weights(f"./weights_{i}/weights_step_{sim_step+2}")
            action = test_network.get_action_from_network() 
            observation = test_network.set_gym_action(action, env)
            position = observation[0]
            velocity = observation[1]     
            if max_position < position:
                max_position = position
            if position >= 0.5:
                max_position = position
                print(f"steps : {sim_step}")
                break
        env.close()
        total_fitness += max_position
    fitness = total_fitness/3
    return fitness


if __name__ == "__main__":
    weights, plasticity_flags = load_individual("individual_965_30_result.csv")
    print(simulate(weights, plasticity_flags))