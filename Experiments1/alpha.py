import numpy as np
import matplotlib.pyplot as plt
import math
import nest 

params = {'t_ref': 1.0}
neuron1 = nest.Create('iaf_psc_alpha', params=params)
neuron2 = nest.Create('iaf_psc_alpha', params=params)
nest.Connect(neuron1, neuron2)
print("\n ------------------------ \n")
print(nest.GetStatus(neuron1))
print("\n ------------------------ \n")
print(nest.GetStatus(nest.GetConnections(source=neuron1, target=neuron2)))
print("\n ------------------------ \n")

e = math.e
tau_syn = 2.0
times = np.linspace(0, 50, 1000)
pre_syn_neurones = [[2, 5], [-1, 10]]

def i_synx(t):
    return (e/tau_syn) * (t* np.exp(-t / tau_syn)) * (t > 0)

def I_syn(times, pre_syn_neurones):
    ret = []
    for tm in times:
        sum = 0
        for nj in pre_syn_neurones:
            w = nj[0]
            tj = nj[1]
            sum += w*i_synx(tm-tj-1)
        ret.append(sum)
    return ret

current = I_syn(times=times, pre_syn_neurones=pre_syn_neurones)
plt.figure(figsize=(8, 4))
plt.plot(times, current, label=f'Current')
plt.title('Post-synaptic current')
plt.xlabel('Times (ms)')
plt.ylabel('Current unit')
plt.legend()
plt.grid(True)
plt.show()



Ie = 0.0 #pA
Cm = 250 #pF
Vm_initial = -65 #mV
El = -70 #mV
tau = 10
membrane_pot = [-65]
for i in range(len(current)-1):
    dVm = (-(membrane_pot[-1] - El)/tau) + ((current[i] + Ie)/Cm)
    membrane_pot.append(membrane_pot[-1] + dVm)


plt.figure(figsize=(8, 4))
plt.plot(times, membrane_pot, label=f'Membrane')
plt.title('Membrane')
plt.xlabel('Times (ms)')
plt.ylabel('Mmebrane unit')
plt.legend()
plt.grid(True)
plt.show()