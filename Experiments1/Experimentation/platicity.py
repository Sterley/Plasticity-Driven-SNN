import nest
import matplotlib.pyplot as plt

nest.ResetKernel()
neuron1 = nest.Create("iaf_psc_alpha")
neuron2 = nest.Create("iaf_psc_alpha")
syn_dict = {
    "synapse_model": "stdp_synapse",
    "alpha": 1.0,
    "lambda": 0.01,
    "weight": 500.0,
    "Wmax": 1000.0
}
nest.Connect(neuron1, neuron2, syn_spec=syn_dict)


"""
conn = nest.GetConnections(source=neuron1, target=neuron2)
conn_status = nest.GetStatus(conn)
print("conn_status : ",conn_status)
print("\nstdp_synapse_status : ",nest.GetDefaults("stdp_synapse"))
neuron2_status = nest.GetStatus(neuron2)
print("\nneuron2_status : ",neuron2_status)

"""


dc_gen = nest.Create("dc_generator", params={"amplitude": 1000.0})
nest.Connect(dc_gen, neuron1)
voltmeter1 = nest.Create("voltmeter")
voltmeter2 = nest.Create("voltmeter")
nest.Connect(voltmeter1, neuron1)
nest.Connect(voltmeter2, neuron2)
spike_recorder_0 = nest.Create("spike_recorder")
spike_recorder = nest.Create("spike_recorder")
nest.Connect(neuron1, spike_recorder_0)
nest.Connect(neuron2, spike_recorder)
sim_time = 1000
dt = 1.0           
weights = []    
for t in range(int(sim_time/dt)):
    nest.Simulate(dt)
    conns = nest.GetConnections(source=neuron1, target=neuron2)
    current_weight = nest.GetStatus(conns, keys='weight')[0]
    weights.append(current_weight)
dmm1 = nest.GetStatus(voltmeter1)[0]
dmm2 = nest.GetStatus(voltmeter2)[0]
Vms1 = dmm1["events"]["V_m"]
Vms2 = dmm2["events"]["V_m"]
ts1 = dmm1["events"]["times"]
ts2 = dmm2["events"]["times"]
spikes_0 = nest.GetStatus(spike_recorder_0, "events")[0]
spikes = nest.GetStatus(spike_recorder, "events")[0]
spike_times_0 = spikes_0["times"]
spike_times = spikes["times"]

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(ts1, Vms1, label="Neurone 1")
print("\nVm1", Vms1)
plt.xlabel("Temps (ms)")
plt.ylabel("Potentiel de membrane (mV)")
plt.legend()
for t in spike_times_0:
    plt.axvline(t, ls="--", color="r", lw=1.5)
plt.subplot(312)
plt.plot(ts2, Vms2, label="Neurone 2")
print("\nVm2", Vms2)
plt.xlabel("Temps (ms)")
plt.ylabel("Potentiel de membrane (mV)")
plt.legend()
for t in spike_times:
    plt.axvline(t, ls="--", color="r", lw=1.5)
plt.subplot(313)
plt.plot(range(int(sim_time)), weights, label="Poids synaptique")
print("\nweight", weights)
plt.xlabel("Temps (ms)")
plt.ylabel("Poids")
plt.legend()
plt.tight_layout()
plt.show()

