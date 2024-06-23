from brian2 import NeuronGroup
from brian2.units import *

import numpy as np
import sys

class Conductance_LIF:

    def __init__(self, Neuron_type:str):
        self.Mode = Neuron_type
        if Neuron_type == 'Excitatory':
            
            self.Params = {
                'Vrest'     : -60.*mV,
                'Vreset'    : -65.*mV,
                'Vthresh'   : -52.*mV,
                'Mem_tau'   : 100*ms,
                'tau_ge'    : 1.0*ms,
                'tau_gi'    : 2.0*ms,
                'tau_theta' : 1e7*ms,
                'Offset'    : 20.*mV,
                'Ad_plus'   : 0.05*mV
            }

            self.Model = '''
                dv/dt = ((IsynE + IsynI)/nS + (Vrest-v))/(Mem_tau) : volt (unless refractory)
                IsynE = ge * nS * (0*mV - v)            : amp
                IsynI = gi * nS * (-100.*mV - v)        : amp
                dge/dt = -ge/(tau_ge)                   : 1
                dgi/dt = -gi/(tau_gi)                   : 1
                dtheta/dt = -theta/(tau_theta)          : volt
                Vthr = theta - Offset + Vthresh         : volt
            '''

            self.Mem_Thr = '(v>Vthr)'

            self.Mem_reset = '''
                v = Vreset
                theta += Ad_plus
            '''
        
        elif Neuron_type == 'Inhibitory':
            self.Params = {
                'Vrest'      : -60.*mV,
                'Vreset'     : -45.*mV,
                'Vthr'       : -40.*mV,
                'Mem_tau'    : 10*ms,
                'tau_ge'     : 1.0*ms,
                'tau_gi'     : 2.0*ms
            }

            self.Model = '''
                dv/dt = ((IsynE + IsynI)/nS + (Vrest-v))/(Mem_tau) : volt (unless refractory)
                IsynE = ge * nS * (0*mV - v)            : amp
                IsynI = gi * nS * (-085.*mV - v)        : amp
                dge/dt = -ge/(tau_ge)                   : 1
                dgi/dt = -gi/(tau_gi)                   : 1
            '''

            self.Mem_Thr = '(v>Vthr)'
            self.Mem_reset = 'v = Vreset'
        
        else:
            print('Select a proper Neuron Architecture!!')
            sys.exit(0)
            
    def GroupMode(self, Neurons, tag_name):
        if self.Mode == 'Excitatory':
            Exc =  NeuronGroup(N=Neurons, model=self.Model, threshold=self.Mem_Thr, refractory=5*ms, reset=self.Mem_reset, method='euler', name=tag_name, namespace=self.Params)
            Exc.v = self.Params['Vrest']
            Exc.theta = np.ones((Neurons)) * 20.*mV
            return Exc
        
        elif self.Mode == 'Inhibitory':
            Inh = NeuronGroup(N=Neurons, model=self.Model, threshold=self.Mem_Thr, refractory=2*ms, reset=self.Mem_reset, method='euler', name=tag_name, namespace=self.Params)
            Inh.v = self.Params['Vrest']
            return Inh