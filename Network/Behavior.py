from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from keras.datasets import mnist
from tqdm import tqdm
from Network.Net import WTA, seed
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np
import yaml

def Traces_S1(S1M):
    fig, axs = plt.subplots(3, sharex=True)
    fig.set_size_inches(8, 6)
    axs[0].plot(S1M.t/second, S1M.pre.T)
    axs[0].set_ylabel('Presynaptic Trace')
    axs[1].plot(S1M.t/second, S1M.post.T)
    axs[1].set_ylabel('Postsynaptic Trace')
    axs[2].plot(S1M.t/second, S1M.w.T)
    axs[2].set_ylabel('Synaptic Weight')
    axs[2].set_xlabel('Time [s]')
    plt.tight_layout()

def Traces_S1_Triplet(S1M):
    fig, axs = plt.subplots(4, sharex=True)
    fig.set_size_inches(8, 6)
    axs[0].plot(S1M.t/second, S1M.pre.T)
    axs[0].set_ylabel('Presynaptic Trace')
    axs[1].plot(S1M.t/second, S1M.post.T)
    axs[1].set_ylabel('Postsynaptic Trace')
    axs[2].plot(S1M.t/second, S1M.post2.T)
    axs[2].set_ylabel('Postsynaptic Trace 2')
    axs[3].plot(S1M.t/second, S1M.w.T)
    axs[3].set_ylabel('Synaptic Weight')
    axs[3].set_xlabel('Time [s]')
    plt.tight_layout()

def pre_post_Spikes(preSp, postSp):
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(8, 6)
    axs[0].plot(preSp.t/second, preSp.i, '.g')
    axs[0].set_ylabel('Pixel Index')
    axs[1].plot(postSp.t/second, postSp.i, '.r')
    axs[1].set_ylabel('Neuron Index')
    axs[1].set_xlabel('Time [s]')
    plt.tight_layout()


def NeuronMem(ESM, ISM, neuron=13):
    fig, axs = plt.subplots(3, sharex=True)
    fig.set_size_inches(8, 6)
    cnt = -50000
    axs[0].set_title('Membrane Potential (Neuron '+ str(neuron) +')')
    axs[0].plot(ESM.t[cnt:]/second, ESM.v[neuron][cnt:]/mV, label='Exc', color='r')
    axs[0].plot(ISM.t[cnt:]/second, ISM.v[neuron][cnt:]/mV, label='Inh', color='b')
    axs[0].plot(ESM.t[cnt:]/second, ESM.Vthr[neuron][cnt:]/mV, label='Exc Threshold', color='black')
    axs[0].set_ylabel('Voltage [mV]')
    axs[0].legend()
    axs[1].set_title('Excitatory Layer Postsynaptic Currents')
    axs[1].plot(ESM.t[cnt:]/second, ESM.IsynE[neuron][cnt:]/pA, label='EPSC', color='r')
    axs[1].plot(ESM.t[cnt:]/second, ESM.IsynI[neuron][cnt:]/pA, label='IPSC', color='b')
    axs[1].set_ylabel('Current [pA]')
    axs[1].legend()
    axs[2].set_title('Inhibitory Layer Postsynaptic Currents')
    axs[2].plot(ISM.t[cnt:]/second, ISM.IsynE[neuron][cnt:]/pA, label='EPSC', color='r')
    axs[2].plot(ISM.t[cnt:]/second, ISM.IsynI[neuron][cnt:]/pA, label='IPSC', color='b')
    axs[2].set_ylabel('Current [pA]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    plt.tight_layout()

def LayerRate(ERM, IRM):
    plt.figure(figsize=(8, 6))
    # Config used for smooth rate == (window="flat", width=0.1*ms)
    plt.plot(ERM.t/second, ERM.smooth_rate(width=0.1*ms)/Hz, color='r')
    plt.plot(IRM.t/second, IRM.smooth_rate(width=0.1*ms)/Hz, color='b')
    plt.ylabel('Rate [Hz]')
    plt.xlabel('Time [s]')
    plt.tight_layout()

def LayerSpike(ESP, ISP):
    plt.figure(figsize=(8,6))
    plt.plot(ESP.t/second, ESP.i, '.r')
    plt.plot(ISP.t/second, ISP.i, '.b')
    plt.ylabel('Neuron Index')
    plt.xlabel('Time [s]')
    plt.tight_layout()

def NeuronRate(ESP):
    counter_dic = {i:list(ESP.i).count(i) for i in ESP.i}
    plt.figure(figsize=(8,6))
    for key, value in counter_dic.items():
        #print(key, '->', value/(0.35*30))
        plt.stem(key, value/(0.35*30))
    plt.xlabel('Neuron Index')
    plt.ylabel('Firing Rate [Sp/s]')
    plt.tight_layout()

def InputRate(InpRM):
    plt.figure(figsize=(8, 6))
    # Config used for smooth rate == (window="flat", width=0.1*ms)
    plt.plot(InpRM.t/second, InpRM.smooth_rate(width=0.1*ms)/Hz, color='g')
    plt.xlabel('Time [s]')
    plt.ylabel('Rate [Hz]')
    plt.tight_layout()

def Learn_Analysis(Net, idx:int=0):
    W_diff = np.diff(Net['Syn1_Mon'].w.T[:,idx])
    w_diff_non_zero = np.where(W_diff != 0)
    dw = W_diff[w_diff_non_zero].tolist()

    Post_spikes = Net['Exc_Sp'].spike_trains()[15]/second
    Pre_spikes = Net['Input_Sp'].spike_trains()[300+idx]/second

    time = Net['Syn1_Mon'].t/second
    t_non_zero = time[w_diff_non_zero]
    

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.set_size_inches(8, 12)
    axs[0].eventplot(Post_spikes, linelengths=0.5, color='r')
    axs[0].eventplot(Pre_spikes, linelengths=0.5, lineoffsets=1.5, color='g')
    axs[0].set_yticks([1, 1.5], ['Post', 'Pre'])
    axs[1].scatter(t_non_zero, dw, label='Weight Changes', color='b')
    axs[1].set_ylabel(r'$\Delta W$')
    axs[1].legend()
    axs[1].grid()
    axs[2].plot(time, Net['Syn1_Mon'].pre.T[:,0], color='g')
    axs[2].set_ylabel(r'$Pre_{trace}$')
    axs[2].grid()
    axs[3].plot(time, Net['Syn1_Mon'].w.T[:,0], color='b')
    axs[3].set_ylabel(r'$W(t)$')
    axs[3].set_xlabel('Time [s]')
    axs[3].grid()
    plt.tight_layout()

def Str2bool(Val_arg):
    if Val_arg == "True": return True
    elif Val_arg == "False": return False

if __name__ == "__main__":
    
    # ==================== Argument Initialization ========================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", default=0, type=int, help="Random Seed Initialization")
    parser.add_argument("-gb", "--gabor", default=True, type=Str2bool, help="Preprocess Input data with Gabor Filter")
    parser.add_argument("-n", "--norm", default=True, type=Str2bool, help="Applied Input Normalization after Gabor Filter")
    parser.add_argument("-p", "--plot", default=True, type=Str2bool, help="Show the weight plots after running the training")
    parser.add_argument("-r", "--run", default=True, type=Str2bool, help="Run training")
    args = vars(parser.parse_args())

    # ===================== Params Initialization =========================
    with open('Network/params.yml', 'r') as file:
        net = yaml.safe_load(file)
    file.close()

    init_params = {
        'Random_Seed':args['seed'],
        'Gabor_filter':args['gabor'],
        'Norm':args['norm'],
        'Run_Behavior':args['run']
    }
    Net_init = {
        'Neurons':net['Net'][0],
        'Learning_Rule':net['Net'][1],
        'Nearest_Neighbor':net['Net'][2],
        'Run_test':net['Behavior'][0],
        'Monitors':net['Behavior'][1]
    }
    # ====================== Load MNIST Dataset ==========================
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 4.
    X_test = X_test / 4.

    # =========================== Model ===================================
    seed(init_params['Random_Seed'])
    Mdl = WTA(Net_setup=Net_init)
    if init_params['Run_Behavior']:
        Mdl.Init_State()
        X_pre = Mdl.preProcess(X_data=X_train[:30], preInp=init_params['Gabor_filter'])
        for idx in tqdm(range(len(X_pre)), desc='Loading'):
            Mdl.Norm_SynW(Norm_w=True)

            Mdl.RunModel(X_single=X_pre[idx], preInp=init_params['Gabor_filter'], norm=init_params['Norm'], phase='Stimulus')
            Mdl.RunModel(phase='Resting')
        Mdl.net.store('Mdl_Behavior','Temp/Mdl_Behavior.b2')
    else:
        Mdl.net.restore('Mdl_Behavior','Temp/Mdl_Behavior.b2')
    
    # ==================== Plots of Network Behavior ====================== 
    if net['Net'][1] == 'pair_STDP': Traces_S1(S1M=Mdl.net['Syn1_Mon'])
    elif net['Net'][1] == 'Triplet_STDP': Traces_S1_Triplet(S1M=Mdl.net['Syn1_Mon'])
    NeuronMem(ESM=Mdl.net['Exc_mem'], ISM=Mdl.net['Inh_mem'], neuron=15)
    LayerRate(ERM=Mdl.net['Exc_rate'], IRM=Mdl.net['Inh_rate'])
    LayerSpike(ESP=Mdl.net['Exc_Sp'], ISP=Mdl.net['Inh_Sp'])
    pre_post_Spikes(preSp=Mdl.net['Input_Sp'], postSp=Mdl.net['Exc_Sp'])
    Learn_Analysis(Net=Mdl.net, idx=0)
    NeuronRate(ESP=Mdl.net['Exc_Sp'])
    InputRate(InpRM=Mdl.net['Input_rate'])
    plt.show(block=args['plot'])