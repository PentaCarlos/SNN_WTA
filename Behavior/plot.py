from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import numpy as np

def plot_mem_potential(gb_time:np.ndarray, pairSTDP:np.ndarray, pairSTDPNN:np.ndarray, TripletSTDP:np.ndarray):
    plt.figure(figsize=(8,6))
    plt.plot(gb_time[-50000:], pairSTDP)
    plt.plot(gb_time[-50000:], pairSTDPNN)
    plt.plot(gb_time[-50000:], TripletSTDP)
    plt.legend(['pairSTDP (All)', 'pairSTDP (NN)', 'TripletSTDP'])
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [mV]')
    plt.xlim(5.5, 6)
    plt.ylim(-70, -40)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Behavior/Mem_potential.png')

def plot_Syn_weight(gb_time:np.ndarray, pairSTDP:np.ndarray, pairSTDPNN:np.ndarray, TripletSTDP:np.ndarray):
    fig, axs = plt.subplots(3, sharex=True)
    fig.set_size_inches(8, 6)
    axs[0].set_title('pair STDP (All-to-all)')
    axs[0].plot(gb_time, pairSTDP)
    axs[0].grid()
    axs[1].set_title('pair STDP (Nearest_neighbor)')
    axs[1].plot(gb_time, pairSTDPNN)
    axs[1].grid()
    axs[2].set_title('Triplet STDP')
    axs[2].plot(gb_time, TripletSTDP)
    axs[2].grid()
    fig.supxlabel('Time [s]')
    fig.supylabel('Synaptic Weight')
    plt.tight_layout()
    plt.savefig('Behavior/Syn_weight.png')

def plot_Inp_code(Inp_time:np.ndarray, Inp:np.ndarray,
                  pairSTDP_time:np.ndarray, pairSTDPNN_time:np.ndarray, TripletSTDP_time:np.ndarray,
                  pairSTDP:np.ndarray, pairSTDPNN:np.ndarray, TripletSTDP:np.ndarray):
    fig = plt.figure(figsize=(12,6), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)
    axs_inp = fig.add_subplot(gs[:, 0])
    axs_inp.set_title('Sparse code from Input after Gabor filter')
    axs_inp.scatter(Inp_time, Inp, color='g', s=6)
    axs_inp.set_ylim(0, 784)
    axs_inp.set_xlim(2, 2.5)

    axs_pairSTDP = fig.add_subplot(gs[0, 1])
    axs_pairSTDP.set_title('Excitatory layer response per Learning Rule')
    axs_pairSTDP.scatter(pairSTDP_time, pairSTDP, color='r', label='pairSTDP (All)', s=6)
    axs_pairSTDP.set_ylim(0, 100)
    axs_pairSTDP.legend()

    axs_pairSTDPNN = fig.add_subplot(gs[1, 1], sharex=axs_pairSTDP)
    axs_pairSTDPNN.scatter(pairSTDPNN_time, pairSTDPNN, color='b', label='pairSTDP (NN)', s=6)
    axs_pairSTDPNN.set_ylim(0, 100)
    axs_pairSTDPNN.legend()

    axs_TripletSTDP = fig.add_subplot(gs[2, 1], sharex=axs_pairSTDP)
    axs_TripletSTDP.scatter(TripletSTDP_time, TripletSTDP, color='m', label='TripletSTDP', s=6)
    axs_TripletSTDP.set_ylim(0, 100)
    axs_TripletSTDP.legend()
    plt.savefig('Behavior/Img_code.png')

plot_mem_potential(gb_time=np.load('Behavior/pairSTDP/All_Interaction/Global_timestep.npy'),
                   pairSTDP=np.load('Behavior/pairSTDP/All_Interaction/Mem_potential.npy'),
                   pairSTDPNN=np.load('Behavior/pairSTDP/Sym_NN/Mem_potential.npy'),
                   TripletSTDP=np.load('Behavior/TripletSTDP/Mem_potential.npy'))

plot_Syn_weight(gb_time=np.load('Behavior/pairSTDP/All_Interaction/Global_timestep.npy'),
                pairSTDP=np.load('Behavior/pairSTDP/All_Interaction/Weight_Change.npy'),
                pairSTDPNN=np.load('Behavior/pairSTDP/Sym_NN/Weight_Change.npy'),
                TripletSTDP=np.load('Behavior/TripletSTDP/Weight_Change.npy'))

plot_Inp_code(Inp_time=np.load('Behavior/pairSTDP/All_Interaction/Inp_timing.npy'),
              Inp=np.load('Behavior/pairSTDP/All_Interaction/Inp_code.npy'),
              pairSTDP_time=np.load('Behavior/pairSTDP/All_Interaction/Exc_timing.npy'),
              pairSTDP=np.load('Behavior/pairSTDP/All_Interaction/Exc_code.npy'),
              pairSTDPNN_time=np.load('Behavior/pairSTDP/Sym_NN/Exc_timing.npy'),
              pairSTDPNN=np.load('Behavior/pairSTDP/Sym_NN/Exc_code.npy'),
              TripletSTDP_time=np.load('Behavior/TripletSTDP/Exc_timing.npy'),
              TripletSTDP=np.load('Behavior/TripletSTDP/Exc_code.npy'))
plt.show(block=True)