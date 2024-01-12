from brian2 import *
import numpy as np
from matplotlib.pyplot import *
import json
import os

cwd = os.getcwd()

prefs.devices.cpp_standalone.openmp_threads = 8

class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed - self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")


# try to reproduce the Montbrio paper:
# WARNING : the constant of the time is not good

# Parameters
simulation_duration = 100 * second

## Neurons constant
c1 = 1 / volt  # constant of the voltage
tau_m = 1.0 * second  # time constant of the membrane
vt = 100.0 * volt  # infinity -
vr = -vt  # infinity +

# synaptic input
R = 1 * ohm  # resistance of the input current
taue = 1.0e-3 * second  # time constant of the synapse

# distribution of the heterogeneity of the neurons
mean = -4.6
delta = 0.7

# stimulus
stimulus = TimedArray(np.hstack([np.zeros((30000,)), 3 * 1e0 * np.ones((30000,)), np.zeros((40000,))]) * amp, dt=1*ms)

# connection
N = 10 ** 4  # number of neurons
J = 14.5  # synaptic weights

# sliding time window
window = 0.01 * second
dt = 0.001

# Equation of the neurons
reset = '''
v = vr
'''

neurons = NeuronGroup(N, '''dv/dt = (v*v * c1 + I * R)/tau_m : volt
                            I = J / N * ge + n + I_stim: ampere
                            I_stim = stimulus(t): ampere
                            dge/dt = -ge / taue : ampere
                            n : ampere
                            ''',
                      threshold='v>vt', reset=reset,
                      method='rk4', refractory=2 / vt * volt * second
                      )
neurons.n = ((np.random.standard_cauchy(N) + mean) * delta) * amp
# initial condition
#neurons.v = np.random.rand(N) * -10.0 * volt
v_init = -3.0
neurons.v = v_init * volt

# Creation of the connection between the neurons
conn = Synapses(neurons, neurons, on_pre='ge += 1/ taue * second *ampere')
#on_pre defines what happens when a presynaptic spike arrives at a synapse
conn.connect(p=1.0) #connects all neuron pairs with a probability of p=1
conn.delay = taue  # 0.0 * second
#propagation delay from the presynaptic neuron to the synapse (presynaptic delay)

# Recording of the neurons

#to record spikes of NeuronGroup "neurons":
neurons_monitor = SpikeMonitor(neurons) 
#to record time-varying firing rate of the population of neurons:
neurons_monitor_rate = PopulationRateMonitor(neurons) 
#to record variable continuously :
neurons_monitor_voltage = StateMonitor(neurons, 'v', record=True, dt=dt*second) 
neurons_monitor_ge = StateMonitor(neurons[:10], 'ge', record=True)
neurons_monitor_I = StateMonitor(neurons[:10], 'I', record=True) 
neurons_monitor_I_stim = StateMonitor(neurons[:1], 'I_stim', record=True) 

# run the simulation
run(simulation_duration, report=ProgressBar(), report_period=1 * second)

figure(1, figsize=(10, 8))
subplot(5, 1, 1)
plot(neurons_monitor.t / second, neurons_monitor.i, '.', markersize=0.08, color='k')
xlim(0, simulation_duration / second)
title('spike trains')
subplot(5, 1, 2)
for i in range(10):
    plot(neurons_monitor_voltage.t / second, neurons_monitor_voltage[i].v / volt, linewidth=0.1)
title('example of the Voltage of 10 neurons')
subplot(5, 1, 3)
for i in range(10):
    plot(neurons_monitor_ge.t / second, neurons_monitor_ge[i].ge / siemens, linewidth=0.1)
title('example of the conductance currant of 10 neurons')
subplot(5, 1, 4)
plot(neurons_monitor_I.t / second, (neurons_monitor_I[0].I - J / N * neurons_monitor_ge[0].ge - mean * amp) / siemens,
     linewidth=1)
plot(neurons_monitor_I_stim.t / second, (neurons_monitor_I_stim[0].I_stim) / amp, linewidth=1)
title('Input current of stimulus')
subplot(5, 1, 5)
# plot(neurons_monitor_rate.t/ms, neurons_monitor_rate.rate/Hz, linewidth=0.1)
plot(neurons_monitor_rate.t / second, neurons_monitor_rate.smooth_rate(window='flat', width=window),
     linewidth=0.1)
title('smoothes rate of the network')
tight_layout()
savefig('montbrio1.png')
show()

figure(2, figsize=(10, 8))
subplot(4, 1, 1)
plot(neurons_monitor.t / second, neurons_monitor.i, '.', markersize=0.08, color='k')
xlim(0, simulation_duration / second)
title('spike trains')
subplot(4, 1, 2)
firing_rate = neurons_monitor_rate.smooth_rate(window='flat', width=window)
plot(neurons_monitor_rate.t / second, firing_rate,
     linewidth=0.1)
title('smoothes rate of the network')
subplot(4, 1, 3)
mean_v = neurons_monitor_voltage[0].v/N/volt
for i in range(1, N):
    mean_v += neurons_monitor_voltage[i].v/N/volt
plot(neurons_monitor_voltage.t / second, mean_v, linewidth=0.1)
title('Mean voltage')
subplot(4, 1, 4)
plot(neurons_monitor_I_stim.t / second, (neurons_monitor_I_stim[0].I_stim) / amp, linewidth=1)
title('Input current of stimulus')
tight_layout()
savefig('montbrio2.png')
show()



#smoothing and downsampling
#ds_index = range(0, len(neurons_monitor_rate.t), 100) #10000 datapoints
#t_ds = neurons_monitor_rate.t[ds_index]

# figure(3, figsize=(10, 8))
# subplot(4, 1, 1)
# plot(neurons_monitor.t / second, neurons_monitor.i, '.', markersize=0.08, color='k')
# xlim(0, simulation_duration / second)
# title('spike trains')
# subplot(4, 1, 2)
#firing_rate = neurons_monitor_rate.smooth_rate(window='flat', width=window)[ds_index]
# plot(t_ds / second, firing_rate, linewidth=0.5)
# title('smoothes rate of the network')

# subplot(4, 1, 3)
mean_v = neurons_monitor_voltage[0].v/N/volt
for i in range(1, N):
    mean_v += neurons_monitor_voltage[i].v/N/volt
#plot(neurons_monitor_voltage.t / second, mean_v, linewidth=0.5)
#np.isclose(mean_v, np.mean(neurons_monitor_voltage.state('v'), axis=1)) #True
# mean_volt = np.mean(neurons_monitor_voltage.state('v'), axis=1)
# cumsum_vec = numpy.cumsum(numpy.insert(mean_volt, 0, 0)) 
# mean_v_smooth = (cumsum_vec[10:] - cumsum_vec[:-10]) / 10
# from scipy import signal
# mean_v_smooth = signal.resample(mean_v_smooth, 10000)
# plot(t_ds / second, mean_v_smooth, linewidth=0.5)
# title('Mean voltage')

# subplot(4, 1, 4)
# plot(t_ds / second, neurons_monitor_I_stim[0].I_stim[ds_index] / amp, linewidth=1)
# title('Input current of stimulus')
# tight_layout()
# savefig('montbrio3.png')
# show()

I_input = neurons_monitor_I_stim[0].I_stim
out_dict = {}
out_dict['r_init'] = 0.0
out_dict['v_init'] = v_init
out_dict['delta_true'] = delta
out_dict['eta_true'] = mean
out_dict['J_true'] = J
out_dict['nt'] = len(mean_v)
out_dict['dt'] = dt
out_dict['I_input'] = list(I_input / amp)
out_dict['rlim'] = [0.0, 8.0]
out_dict['vlim'] = [-8.0, 8.0]
out_dict['rs'] = list(firing_rate / hertz)
#out_dict['vs'] = list(mean_v_smooth)
out_dict['vs'] = list(mean_v)
out_dict['tsr'] = list(neurons_monitor_rate.t / second)
out_dict['tsv'] = list(neurons_monitor_voltage.t / second)


print(len(out_dict['rs'] ))
print(len(out_dict['vs'] ))

data_folder = 'Res_syntheticData/data_input_files/'
newpath = cwd + '/' + data_folder
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
    
out_filename = 'qif_data'
nd=[]
npz = {'nd': nd}
npz.update(out_dict)
np.savez(data_folder + out_filename +'.R.npz', **npz)
np.savez(data_folder + out_filename +'.npz', **npz)

json_object = json.dumps(out_dict, indent=4)
with open(data_folder + out_filename + ".json", "w") as outfile:
    outfile.write(json_object)