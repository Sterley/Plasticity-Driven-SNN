import numpy as np
import matplotlib.pyplot as plt


Wmax = 1000.0  
Wmin = 0.0    
w = 500.0       
lambda_ = 0.01
alpha = -1.0
mu_plus = 1.0
mu_minus = 1.0
tau_tr_pre = 20.0  
tau_tr_post = 20.0 

sim_time = 1000
dt = 1.0

pre_trace = 0.0
post_trace = 0.0
weights = []
pre_trace_values = []
post_trace_values = []

for t in np.arange(0, sim_time, dt):
   
    pre_trace *= np.exp(-dt / tau_tr_pre)
    post_trace *= np.exp(-dt / tau_tr_post)

    
    if t % 50 == 0:  
        pre_trace += 1.0
    if t % 100 == 0:  
        post_trace += 1.0

    
    w_pot = Wmax * (w / Wmax + lambda_ * (1 - w / Wmax)**mu_plus * pre_trace)
    w_dep = Wmax * (w / Wmax - alpha * lambda_ * (w / Wmax)**mu_minus * post_trace)
    w = min(max(Wmin, w_dep), w_pot)

   
    weights.append(w)
    pre_trace_values.append(pre_trace)
    post_trace_values.append(post_trace)


plt.figure(figsize=(12, 6))

plt.subplot(311)
plt.plot(np.arange(0, sim_time, dt), pre_trace_values, label='Pre-synaptic Trace')
plt.xlabel('Time (ms)')
plt.ylabel('Pre-synaptic Trace')
plt.legend()

plt.subplot(312)
plt.plot(np.arange(0, sim_time, dt), post_trace_values, label='Post-synaptic Trace')
plt.xlabel('Time (ms)')
plt.ylabel('Post-synaptic Trace')
plt.legend()

plt.subplot(313)
plt.plot(np.arange(0, sim_time, dt), weights, label='Synaptic weight')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.legend()

plt.tight_layout()
plt.show()
