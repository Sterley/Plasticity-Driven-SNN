from datetime import datetime
from l2l.utils.experiment import Experiment
from l2l.optimizees.mc_gym.optimizee_mc import NeuroEvolutionOptimizeeMC, NeuroEvolutionOptimizeeMCParameters
from l2l.optimizers.crossentropy import CrossEntropyParameters, CrossEntropyOptimizer
import os
import numpy as np

class Gaussian:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
        self.random_state = None
    def init_random_state(self, random_state):
        self.random_state = random_state
    def sample(self, pop_size):
        """
        Génère des échantillons de la distribution actuelle avec la forme correcte.
        
        :param pop_size: Nombre d'individus à générer.
        :return: Un tableau NumPy des échantillons générés.
        """
        # La forme attendue est (pop_size, dimension_de_chaque_individu)
        # Nous utilisons self.mean et self.std qui doivent être configurés pour refléter la dimension correcte des individus.
        if len(self.mean.shape) == 1:  # Si mean est unidimensionnel, nous supposons une distribution indépendante pour chaque dimension
            return self.random_state.normal(self.mean, self.std, (pop_size, self.mean.shape[0]))
        else:  # Si mean n'est pas unidimensionnel, il doit déjà être dans la forme correcte
            return self.random_state.normal(self.mean, self.std, (pop_size, len(self.mean)))

    def fit(self, elite_samples, smoothing=0.0):
        if not self.random_state:
            raise ValueError("Random state not initialized")
        new_mean = np.mean(elite_samples, axis=0)
        new_std = np.std(elite_samples, axis=0)
        if smoothing > 0.0:
            self.mean = smoothing * self.mean + (1 - smoothing) * new_mean
            self.std = smoothing * self.std + (1 - smoothing) * new_std
        else:
            self.mean = new_mean
            self.std = new_std
        return {'mean': self.mean, 'std': self.std}
    def get_params(self):
        return {'mean': self.mean, 'std': self.std}


def run_experiment():
    experiment = Experiment(root_dir_path='crossentropy_results')
    jube_params = {"exec": "python3.8"} 
    traj, _ = experiment.prepare_experiment(
        jube_parameter=jube_params, name="CrossEntropy_{}".format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S")))
        
    # Optimizee params
    optimizee_parameters = NeuroEvolutionOptimizeeMCParameters(
        path=experiment.root_dir_path, seed=1, save_n_generation=10, run_headless=True, load_parameter=False)
    optimizee = NeuroEvolutionOptimizeeMC(traj, optimizee_parameters)

    optimizer_seed = 12345678
    optimizer_parameters = CrossEntropyParameters(
        pop_size=50,
        rho=0.2,
        smoothing=0.7,
        temp_decay=0.95,
        n_iteration=100,
        distribution=Gaussian(std=1.0, mean=0.0),
        stop_criterion=0.1,
        seed=optimizer_seed)

    optimizer = CrossEntropyOptimizer(traj,
                                      optimizee_create_individual=optimizee.create_individual,
                                      optimizee_fitness_weights=(1,),
                                      parameters=optimizer_parameters,
                                      optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)

def main():
    run_experiment()

if __name__ == '__main__':
    main()
