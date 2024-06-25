from Network.Neuron import Conductance_LIF
from Network.Learn import WTA_Connection
from Network.Tools import GaborKernel, filterGb, norm_Weight
from skimage.measure import block_reduce
from tqdm import tqdm
from brian2 import *

import numpy as np
import sys

class WTA:

    def __init__(self, Net_setup:dict={'Neurons':100, 'Learning_Rule':'pair_STDP', 'Nearest_Neighbor':True, 'Run_test':False, 'Monitors':False}):
        Mdl = {}
        self.n_input = 28*28
        self.n_layer = Net_setup['Neurons']
        
        # Initialize Predifined Class Models
        Neuron_Exc = Conductance_LIF(Neuron_type='Excitatory')
        Neuron_Inh = Conductance_LIF(Neuron_type='Inhibitory')
        SynConn = WTA_Connection(Rule=Net_setup['Learning_Rule'], Nearest_Neighbor=Net_setup['Nearest_Neighbor'])
        
        # Input images as rate encoded Poisson generators
        Mdl['Input'] = PoissonGroup(self.n_input, rates=np.zeros(self.n_input)*Hz, name='Input')

        # Create Excitatory and Inhibitory Neuron Clusters for WTA Architecture
        Mdl['Exc'] = Neuron_Exc.GroupMode(Neurons=self.n_layer, tag_name='Exc')
        Mdl['Inh'] = Neuron_Inh.GroupMode(Neurons=self.n_layer, tag_name='Inh')

        # Synapse 1 (Learning) [Input --> Exc]
        if Net_setup['Run_test'] == True: self.Stdp_Switch = 0
        else: self.Stdp_Switch = 1
        Mdl['Syn1'] = SynConn.ConnSTDP(preConn=Mdl['Input'], postConn=Mdl['Exc'], Stdp_state=self.Stdp_Switch, tag_name='Syn1')
        # Synapse 2 (Static) [Exc --> Inh]
        Mdl['Syn2'] = SynConn.ConnStatic(preConn=Mdl['Exc'], postConn=Mdl['Inh'], pre_event='ge += w', Cond='j==i', Syn_weight=10.4, tag_name='Syn2')
        # Synapse 3 (Static) [Inh --> Exc]
        Mdl['Syn3'] = SynConn.ConnStatic(preConn=Mdl['Inh'], postConn=Mdl['Exc'], pre_event='gi += w', Cond='j!=i', Syn_weight=17, tag_name='Syn3')

        # Monitors
        if Net_setup['Monitors']:
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
    
    def Init_State(self):
        Mem_Potential = {
            'Exc':self.net['Exc'].v,
            'Inh':self.net['Inh'].v
        }
        Syn_Conductance = {
            'Exc_ge':self.net['Exc'].ge,
            'Exc_gi':self.net['Exc'].gi,
            'Inh_ge':self.net['Inh'].ge,
            'Inh_gi':self.net['Inh'].gi
        }
        Stdp_traces = {
            'Pre_trace':self.net['Syn1'].pre,
            'Post_trace':self.net['Syn1'].post
        }

        self.init_Mem = Mem_Potential
        self.init_Syn_Cond = Syn_Conductance
        self.init_Stdp_traces = Stdp_traces

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
            # self.net['Input'].rates = X_single*Hz
            # self.net.run(0.15*second)
            self.net['Exc'].v = self.init_Mem['Exc']
            self.net['Inh'].v = self.init_Mem['Inh']
            self.net['Exc'].ge = self.init_Syn_Cond['Exc_ge']
            self.net['Exc'].gi = self.init_Syn_Cond['Exc_gi']
            self.net['Inh'].ge = self.init_Syn_Cond['Inh_ge']
            self.net['Inh'].gi = self.init_Syn_Cond['Inh_gi']
            self.net['Syn1'].pre = self.init_Stdp_traces['Pre_trace']
            self.net['Syn1'].post = self.init_Stdp_traces['Post_trace']
        
        else:
            print('Phase not correctly declared!!')
            sys.exit(0)