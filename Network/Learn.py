from brian2 import Synapses
from brian2.units import *

class WTA_Connection:

    def __init__(self, Rule:str, Nearest_Neighbor:bool):
        self.Mode = Rule
        self.scheme = Nearest_Neighbor

        if Rule == 'pair_STDP':
            self.Stdp = '''
                w : 1
                On : 1 (shared)
                dpre/dt = -pre/(tau_pre)    : 1 (event-driven)
                dpost/dt = -post/(tau_post) : 1 (event-driven)
            '''
            if Nearest_Neighbor:
                self.Params = {
                    'tau_pre'   : 20*ms,
                    'tau_post'  : 20*ms,
                    'pre_rate'  : 0.0001,
                    'post_rate' : 0.01,
                    'Gmax'      : 1.0
                }
                self.pre_event = '''
                    ge += w
                    pre = 1.
                    w = clip(w + On*pre_rate*post, 0, Gmax)
                '''
                self.post_event = '''
                    w = clip(w + On*post_rate*pre, 0, Gmax)
                    post = -1.
                '''
            else:
                self.Params = {
                    'tau_pre'   : 20*ms,
                    'tau_post'  : 20*ms,
                    'Apost'     : -0.0105,
                    'Apre'      : 0.01,
                    'Gmax'      : 0.05
                }
                self.pre_event = '''
                    ge += w
                    pre += Apre
                    w = clip(w + On*post, 0, Gmax)
                '''
                self.post_event = '''
                    post += Apost
                    w = clip(w + On*pre, 0, Gmax)
                '''
        elif Rule == 'Triplet_STDP':
            self.Stdp = '''
                w : 1
                On : 1 (shared)
                post2before                     : 1
                dpre/dt = -pre/(tau_pre)        : 1 (event-driven)
                dpost/dt = -post/(tau_post)     : 1 (event-driven)
                dpost2/dt = -post2/(tau_post2)  : 1 (event-driven)
            '''
            self.Params = {
                'tau_pre'   : 20*ms,
                'tau_post'  : 20*ms,
                'tau_post2' : 40*ms,
                'pre_rate'  : 0.001,
                'post_rate' : 0.1,
                'Gmax'      : 1.0
            }
            self.pre_event = '''
                ge += w
                pre = 1.
                w = clip(w + On*pre_rate*post, 0, Gmax)
            '''
            self.post_event = '''
                post2before = post2
                w = clip(w + On*post_rate*pre*post2before, 0, Gmax)
                post  = 1.
                post2 = 1.
            '''

    def ConnSTDP(self, preConn, postConn, Stdp_state, tag_name):
        Syn_STDP = Synapses(source=preConn, target=postConn, model=self.Stdp, on_pre=self.pre_event, on_post=self.post_event, method='euler', name=tag_name, namespace=self.Params)
        Syn_STDP.connect(True) # All-to-all Connection
        Syn_STDP.On = Stdp_state # Enable STDP
        Syn_STDP.w = 'rand()*Gmax'
        return Syn_STDP
    
    def ConnStatic(self, preConn, postConn, pre_event, Cond, Syn_weight, tag_name):
        Syn_Static = Synapses(source=preConn, target=postConn, model='w : 1', on_pre=pre_event, method='euler', name=tag_name)
        Syn_Static.connect(condition=Cond)
        Syn_Static.w = Syn_weight
        return Syn_Static