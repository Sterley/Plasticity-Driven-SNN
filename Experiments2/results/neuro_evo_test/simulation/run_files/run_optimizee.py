import pickle
import sys
idx = sys.argv[1]
iteration = sys.argv[2]
handle_trajectory = open("/home/labady/Documents/PR1/results/neuro_evo_test/simulation/trajectories/trajectory_" + str(idx) + "_" + str(iteration) + ".bin", "rb")
trajectory = pickle.load(handle_trajectory)
handle_trajectory.close()
handle_optimizee = open("/home/labady/Documents/PR1/results/neuro_evo_test/simulation/optimizee.bin", "rb")
optimizee = pickle.load(handle_optimizee)
handle_optimizee.close()

res = optimizee.simulate(trajectory)

handle_res = open("/home/labady/Documents/PR1/results/neuro_evo_test/simulation/results/results_" + str(idx) + "_" + str(iteration) + ".bin", "wb")
pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)
handle_res.close()

handle_res = open("/home/labady/Documents/PR1/results/neuro_evo_test/simulation/ready_files/ready_0_" + str(idx), "wb")
handle_res.close()