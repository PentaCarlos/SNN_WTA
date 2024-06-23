from keras.datasets import mnist
from tqdm import tqdm
from Network.Net import WTA, seed
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np

def Traces_S1(S1M):
    fig, axs = plt.subplots(3, sharex=True)
    fig.set_size_inches(8, 6)
    axs[0].plot(S1M.t/second, S1M.pre.T)
    axs[0].set_ylabel('Presynaptic Trace')
    axs[1].plot(S1M.t/second, S1M.post.T)
    axs[1].set_ylabel('Postsynaptic Trace')
    axs[2].plot(S1M.t/second, S1M.w.T)
    axs[2].set_ylabel('Synaptic Weight')
    plt.tight_layout()

def pre_post_Spikes(preSp, postSp):
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(8, 6)
    axs[0].plot(preSp.t/second, preSp.i, '.g')
    axs[0].set_ylabel('Pixel Index')
    axs[1].plot(postSp.t/second, postSp.i, '.r')
    axs[1].set_ylabel('Neuron Index')
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
    axs[1].set_title('Excitatory Layer')
    axs[1].plot(ESM.t[cnt:]/second, ESM.IsynE[neuron][cnt:]/pA, label='EPSC', color='r')
    axs[1].plot(ESM.t[cnt:]/second, ESM.IsynI[neuron][cnt:]/pA, label='IPSC', color='b')
    axs[1].set_ylabel('Current [pA]')
    axs[1].legend()
    axs[2].set_title('Inhibitory Layer')
    axs[2].plot(ISM.t[cnt:]/second, ISM.IsynE[neuron][cnt:]/pA, label='EPSC', color='r')
    axs[2].plot(ISM.t[cnt:]/second, ISM.IsynI[neuron][cnt:]/pA, label='IPSC', color='b')
    axs[2].set_ylabel('Current [pA]')
    axs[2].set_xlabel('Time [s]')
    axs[2].legend()
    plt.tight_layout()

def LayerRate(ERM, IRM):
    plt.figure(figsize=(8, 6))
    # Config used for smooth rate == (window="flat", width=0.1*ms)
    plt.plot(ERM.t/second, ERM.rate*Hz, color='r')
    plt.plot(IRM.t/second, IRM.rate*Hz, color='b')
    plt.ylabel('Rate [Hz]')
    plt.tight_layout()

def LayerSpike(ESP, ISP):
    plt.figure(figsize=(8,6))
    plt.plot(ESP.t/second, ESP.i, '.r')
    plt.plot(ISP.t/second, ISP.i, '.b')
    plt.ylabel('Neuron Index')
    plt.tight_layout()

if __name__ == "__main__":
    
    # Random SEED
    seed(0)
    Run_behavior = False

    # ====================== Load MNIST Dataset ==========================
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 4.
    X_test = X_test / 4.

    # =========================== Model ===================================
    Mdl = WTA(Monitors=True)
    if Run_behavior:
        X_pre = Mdl.preProcess(X_data=X_train[:30], preInp=True)
        for idx in tqdm(range(len(X_pre)), desc='Loading'):
            Mdl.Norm_SynW(Norm_w=True)

            Mdl.RunModel(X_single=X_pre[idx], preInp=True, norm=True, phase='Stimulus')
            Mdl.RunModel(phase='Resting')
        Mdl.net.store('Mdl_Behavior','Temp/Mdl_Behavior.b2')
    else:
        Mdl.net.restore('Mdl_Behavior','Temp/Mdl_Behavior.b2')
    
    # ==================== Plots of Network Behavior ======================
    
    def get_dW(idx):
        #new_w = [round(val_w, 2) for val_w in Mdl.net['Syn1_Mon'].w.T[:,idx]]
        W_diff = np.diff(Mdl.net['Syn1_Mon'].w.T[:,idx])
        w_diff_non_zero = np.where(W_diff != 0)
        dw = W_diff[w_diff_non_zero].tolist()

        Post_spikes = Mdl.net['Exc_Sp'].spike_trains()[15]/second
        Pre_spikes = Mdl.net['Input_Sp'].spike_trains()[300+idx]/second

        # dt = []
        # time = Mdl.net['Syn1_Mon'].t/second
        # t_non_zero = time[w_diff_non_zero]
        # for i, Sp_time in enumerate(t_non_zero):
        #     t_post = []
        # for i in range(0, len(dw)):
        #     time_stamp = t_non_zero[i]
        #     t_post = (Post_spikes[np.where(Post_spikes <= time_stamp)])[-1]
        #     t_pre = (Pre_spikes[np.where(Pre_spikes <= time_stamp)])[-1]
        #     dt.append(t_post - t_pre)
        

        plt.figure()
        plt.eventplot(Post_spikes, linelengths=0.5, color='g')
        plt.eventplot(Pre_spikes, linelengths=0.5, lineoffsets=1.5, color='r')
        plt.yticks([1, 1.5], ['Post', 'Pre'])
        
        # plt.figure()
        # plt.scatter(dt, dw, label='Weight Changes', color='b')
        # plt.xlim(-0.2, 0.2)
        # plt.legend()
        # plt.grid()
    
    get_dW(idx=0)
    
    plt.show()
    #print(np.array(Mdl.net['Syn1_Mon'].post.T[:,0]).shape)

    # Traces_S1(S1M=Mdl.net['Syn1_Mon'])
    # NeuronMem(ESM=Mdl.net['Exc_mem'], ISM=Mdl.net['Inh_mem'], neuron=13)
    # LayerRate(ERM=Mdl.net['Exc_rate'], IRM=Mdl.net['Inh_rate'])
    # LayerSpike(ESP=Mdl.net['Exc_Sp'], ISP=Mdl.net['Inh_Sp'])
    # pre_post_Spikes(preSp=Mdl.net['Input_Sp'], postSp=Mdl.net['Exc_Sp'])
    # plt.show()