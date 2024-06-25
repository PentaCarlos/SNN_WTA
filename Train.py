from keras.datasets import mnist
from tqdm import tqdm
from Network.Net import WTA, seed
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os

def Gabor_Weight_plot(Syn1_weight):
    Weight_m = np.array(Syn1_weight).reshape((784, 100))
    init_val = 0
    pxl_val = 196
    for orientation in range(4):
        dummy_mtx = np.zeros((140, 140))
        neuron = 0
        for i in range(10):
            for j in range(10):
                actual_w = Weight_m[init_val:pxl_val,neuron].reshape((14, 14))
                dummy_mtx[(i*14):(14*(i+1)), (j*14):((14*(j+1)))] = actual_w
                neuron += 1
        plt.figure(figsize=(8,6))
        plt.imshow(dummy_mtx, cmap='hot_r')
        plt.colorbar()
        plt.tight_layout()
        init_val = pxl_val
        pxl_val += 196

def Clean_TempFolder(Flush:bool=False):
    if Flush:
        file_list = glob('Temp/*.b2')
        for temp_file in file_list:
            os.remove(temp_file)

if __name__ == "__main__":
    
    # =========================== Parameters ==============================
    init_params = {
        'Random_Seed':0,
        'Filename':'pairSTDP_NN',
        'Gabor_filter':True,
        'Norm':True,
        'Train_dt':1000,
        'Epoch':5,
        'Run_train':True
    }

    Net_init = {
        'Neurons':100,
        'Learning_Rule':'pair_STDP',
        'Nearest_Neighbor':True,
        'Run_test':False,
        'Monitors':False
    }

    # ====================== Load MNIST Dataset ==========================
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 4.
    X_test = X_test / 4.

    # =========================== Model ===================================
    seed(init_params['Random_Seed'])
    Mdl = WTA(Net_setup=Net_init)
    if init_params['Run_train']:
        Clean_TempFolder(Flush=True)
        X_pre = Mdl.preProcess(X_data=X_train[:init_params['Train_dt']], preInp=init_params['Gabor_filter'])

        print("================== # TRAINING MODEL # ==================")
        for ep in range(init_params['Epoch']):
            for idx in tqdm(range(len(X_pre)), desc='Loading ' + str(ep + 1)):
                Mdl.Norm_SynW(Norm_w=True)

                Mdl.RunModel(X_single=X_pre[idx], preInp=init_params['Gabor_filter'], norm=init_params['Norm'], phase='Stimulus')
                Mdl.RunModel(phase='Resting')
            temp_ep = 'Temp_Ep' + str(ep + 1)
            Mdl.net.store(temp_ep, 'Temp/' + temp_ep + '.b2')
        Mdl.net.store(init_params['Filename'],'Trained_Models/' + init_params['Filename'] + '.b2')
    else:
        Mdl.net.restore(init_params['Filename'],'Trained_Models/' + init_params['Filename'] + '.b2')
    

    # ==================== Plots of Network Behavior ======================
    if Net_init['Neurons'] == 100: Gabor_Weight_plot(Syn1_weight=Mdl['Syn1'].w)
    plt.show()