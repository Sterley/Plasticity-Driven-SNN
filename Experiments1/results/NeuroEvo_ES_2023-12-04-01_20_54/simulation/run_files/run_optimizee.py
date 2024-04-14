import pickle
import sys
idx = sys.argv[1]
iteration = sys.argv[2]
handle_trajectory = open("/home/labady/Documents/PR/results/NeuroEvo_ES_2023-12-04-01_20_54/simulation/trajectories/trajectory_" + str(idx) + "_" + str(iteration) + ".bin", "rb")
trajectory = pickle.load(handle_trajectory)
handle_trajectory.close()
handle_optimizee = open("/home/labady/Documents/PR/results/NeuroEvo_ES_2023-12-04-01_20_54/simulation/optimizee.bin", "rb")
optimizee = pickle.load(handle_optimizee)
handle_optimizee.close()

res = optimizee.simulate(trajectory)

handle_res = open("/home/labady/Documents/PR/results/NeuroEvo_ES_2023-12-04-01_20_54/simulation/results/results_" + str(idx) + "_" + str(iteration) + ".bin", "wb")
pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)
handle_res.close()

handle_res = open("/home/labady/Documents/PR/results/NeuroEvo_ES_2023-12-04-01_20_54/simulation/ready_files/ready_245_" + str(idx), "wb")
handle_res.close()