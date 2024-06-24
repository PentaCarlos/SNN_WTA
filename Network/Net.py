from Network.Neuron import Conductance_LIF
from Network.Learn import Learning
from Network.Tools import GaborKernel, filterGb, norm_Weight
from skimage.measure import block_reduce
from tqdm import tqdm
from brian2 import *

import numpy as np
import sys

class WTA:

    def __init__(self, Monitors:bool=False, Run_Test:bool=False):
        Mdl = {}
        self.n_input = 28*28
        self.n_layer = 100
        
        if Run_Test == True: self.Stdp_Switch = 0
        else: self.Stdp_Switch = 1
        
        # Initialize Predifined Class Models
        Neuron_Exc = Conductance_LIF(Neuron_type='Excitatory')
        Neuron_Inh = Conductance_LIF(Neuron_type='Inhibitory')
        Learn_Rule = Learning(Rule='pair_STDP', Nearest_Neighbor=True)
        
        # Input images as rate encoded Poisson generators
        Mdl['Input'] = PoissonGroup(self.n_input, rates=np.zeros(self.n_input)*Hz, name='Input')

        # Create Excitatory and Inhibitory Neuron Clusters for WTA Architecture
        Mdl['Exc'] = Neuron_Exc.GroupMode(Neurons=self.n_layer, tag_name='Exc')
        Mdl['Inh'] = Neuron_Inh.GroupMode(Neurons=self.n_layer, tag_name='Inh')

        # Synapse 1 (Learning) [Input --> Exc]
        Mdl['Syn1'] = Learn_Rule.ConnSTDP(preConn=Mdl['Input'], postConn=Mdl['Exc'], Stdp_state=self.Stdp_Switch, tag_name='Syn1')
        # Synapse 2 (Static) [Exc --> Inh]
        Mdl['Syn2'] = Learn_Rule.ConnDirect(preConn=Mdl['Exc'], postConn=Mdl['Inh'], pre_event='ge += w', Syn_weight=10.4, tag_name='Syn2')
        # Synapse 3 (Static) [Inh --> Exc]
        Mdl['Syn3'] = Learn_Rule.ConnIndirect(preConn=Mdl['Inh'], postConn=Mdl['Exc'], pre_event='gi += w', Syn_weight=17, tag_name='Syn3')

        # Monitors
        if Monitors:
            Mdl['Exc_Sp'] = SpikeMonitor(Mdl['Exc'], name='Exc_Sp')
            Mdl['Inh_Sp'] = SpikeMonitor(Mdl['Inh'], name='Inh_Sp')
            Mdl['Input_Sp'] = SpikeMonitor(Mdl['Input'], name='Input_Sp')
            Mdl['Exc_rate'] = PopulationRateMonitor(Mdl['Exc'], name='Exc_rate')
            Mdl['Inh_rate'] = PopulationRateMonitor(Mdl['Inh'], name='Inh_rate')
            Mdl['Exc_mem'] = StateMonitor(Mdl['Exc'], ['v', 'IsynE', 'IsynI', 'Vthr'], record=True, name='Exc_mem')
            Mdl['Inh_mem'] = StateMonitor(Mdl['Inh'], ['v', 'IsynE', 'IsynI'], record=True, name='Inh_mem')
            Mdl['Syn1_Mon'] = StateMonitor(Mdl['Syn1'], ['pre', 'post', 'w'], record=Mdl['Syn1'][300:304,15], name='Syn1_Mon')

        # Save Network as Objects for the Class
        self.net = Network(Mdl.values())
        self.net.run(0*second)
    
    def __getitem__(self, key):
        return self.net[key]
    
    def Norm_SynW(self, Norm_w:bool=False):
        if Norm_w: self.net['Syn1'].w = norm_Weight(Syn=self.net['Syn1'], Exc_neurons=self.n_layer)

    def preProcess(self, X_data, preInp=False):
        Gb_kernels = GaborKernel(Gb_phi='Even')

        if preInp:
            X_pre = []
            for idx in tqdm(range(len(X_data)), desc='Filtering'):
                pre_data = filterGb(Img=X_data[idx], kernel=Gb_kernels)
                single_img = [block_reduce(img, (2, 2), np.max) for img in pre_data]
                X_pre.append(single_img)
        else:
            X_pre = X_data
        return X_pre
    
    def RunModel(self, X_single:ndarray=np.zeros(28*28), preInp:bool=False, norm:bool=False, phase:str='Resting'):
        if phase == 'Stimulus':
            if preInp:
                Inp_data = []
                for idx in range(4):
                    img_arr = np.array(X_single[idx]).reshape((14*14)) # Reshaped Reduced Gabor Filtered Img
                    if norm:
                        norm_factor = 1000.
                        Avr_Gb = np.sum(img_arr)
                        img_norm = [(norm_factor*x)/Avr_Gb for x in img_arr]
                        Inp_data.extend(img_norm)
                    else:
                        Inp_data.extend(img_arr)
                
                # Presenting Stimulus
                self.net['Input'].rates = Inp_data*Hz
                self.net.run(0.35*second)
            else:
                # Presenting Stimulus
                self.net['Input'].rates = np.array(X_single).reshape((self.n_input))*Hz
                self.net.run(0.35*second)
        
        elif phase == 'Resting':
            # Resting Phase
            self.net['Input'].rates = X_single*Hz
            self.net.run(0.15*second)
        
        else:
            print('Phase not correctly declared!!')
            sys.exit(0)