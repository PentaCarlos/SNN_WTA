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
        self.Setup = Net_setup
        
        # Initialize Predifined Class Models
        Neuron_Exc = Conductance_LIF(Neuron_type='Excitatory')
        Neuron_Inh = Conductance_LIF(Neuron_type='Inhibitory')
        SynConn = WTA_Connection(Rule=Net_setup['Learning_Rule'], Nearest_Neighbor=Net_setup['Nearest_Neighbor'])
        self.Exc_params = Neuron_Exc.Params
        self.Inh_params = Neuron_Inh.Params
        self.Stdp_params = SynConn.Params
        
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
            if self.Setup['Learning_Rule'] == 'pair_STDP': Syn_var = ['pre', 'post', 'w']
            elif self.Setup['Learning_Rule'] == 'Triplet_STDP': Syn_var = ['pre', 'post', 'post2', 'w']
            Mdl['Exc_Sp'] = SpikeMonitor(Mdl['Exc'], name='Exc_Sp')
            Mdl['Inh_Sp'] = SpikeMonitor(Mdl['Inh'], name='Inh_Sp')
            Mdl['Input_Sp'] = SpikeMonitor(Mdl['Input'], name='Input_Sp')
            Mdl['Exc_rate'] = PopulationRateMonitor(Mdl['Exc'], name='Exc_rate')
            Mdl['Inh_rate'] = PopulationRateMonitor(Mdl['Inh'], name='Inh_rate')
            Mdl['Input_rate'] = PopulationRateMonitor(Mdl['Input'], name='Input_rate')
            Mdl['Exc_mem'] = StateMonitor(Mdl['Exc'], ['v', 'IsynE', 'IsynI', 'Vthr'], record=True, name='Exc_mem')
            Mdl['Inh_mem'] = StateMonitor(Mdl['Inh'], ['v', 'IsynE', 'IsynI'], record=True, name='Inh_mem')
            Mdl['Syn1_Mon'] = StateMonitor(Mdl['Syn1'], Syn_var, record=Mdl['Syn1'][300:304,15], name='Syn1_Mon')

        # Save Network as Objects for the Class
        self.net = Network(Mdl.values())
        self.net.run(0*second)
    
    def __getitem__(self, key):
        return self.net[key]
    
    def Norm_SynW(self, Norm_w:bool=False):
        if Norm_w: self.net['Syn1'].w = norm_Weight(Syn=self.net['Syn1'], Exc_neurons=self.n_layer)

    def Norm_Inp(self, Inp:list, Inp_Shape:int, Norm_factor:int, Norm_Inp:bool=False):
        Inp = np.array(Inp).reshape((Inp_Shape))
        if Norm_Inp:
            #norm_factor = 2750.
            Avr = np.sum(Inp)
            return (Norm_factor/Avr)*Inp
        else: return Inp

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
        self.Stab_rate = np.sum(X_pre)/np.sum(X_data)
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
        self.init_Mem = Mem_Potential
        self.init_Syn_Cond = Syn_Conductance

        if self.Setup['Learning_Rule'] == 'pair_STDP':
            Stdp_traces = {
                'Pre_trace':self.net['Syn1'].pre,
                'Post_trace':self.net['Syn1'].post
            }
        if self.Setup['Learning_Rule'] == 'Triplet_STDP':
            Stdp_traces = {
                'Pre_trace':self.net['Syn1'].pre,
                'Post_trace':self.net['Syn1'].post,
                'Post_trace2':self.net['Syn1'].post2
            }
        self.init_Stdp_traces = Stdp_traces
    
    def get_HomeoThr(self):
        v_theta = self.net.get_states(units=False)['Exc']['theta']
        v_thr = np.array([(self.Exc_params['Vthresh']/volt)-0.02]*self.n_layer)
        return np.array([v_theta[idx] + vthr_offset for idx, vthr_offset in enumerate(v_thr)])

    def RunModel(self, X_single:ndarray=np.zeros(28*28), preInp:bool=False, norm:bool=False, phase:str='Resting'):
        if phase == 'Stimulus':
            if preInp:
                Inp_data = []
                for idx in range(4):
                    img_arr = np.array(X_single[idx]).reshape((14*14)) # Reshaped Reduced Gabor Filtered Img
                    Gb_norm = self.Norm_Inp(Inp=img_arr, Inp_Shape=14*14, Norm_factor=1000, Norm_Inp=norm)
                    if not norm: Gb_norm = Gb_norm/self.Stab_rate
                    Inp_data.extend(Gb_norm)
            else:
                Inp_data = self.Norm_Inp(Inp=X_single, Inp_Shape=self.n_input, Norm_factor=2750, Norm_Inp=norm)
            
            # Presenting Stimulus
            self.net['Input'].rates = Inp_data*Hz
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
            if self.Setup['Learning_Rule'] == 'pair_STDP':
                self.net['Syn1'].pre = self.init_Stdp_traces['Pre_trace']
                self.net['Syn1'].post = self.init_Stdp_traces['Post_trace']
            else:
                self.net['Syn1'].pre = self.init_Stdp_traces['Pre_trace']
                self.net['Syn1'].post = self.init_Stdp_traces['Post_trace']
                self.net['Syn1'].post2 = self.init_Stdp_traces['Post_trace2']
        else:
            print('Phase not correctly declared!!')
            sys.exit(0)