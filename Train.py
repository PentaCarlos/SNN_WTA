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
    seed(0)
    file_name = 'pairSTDP_40k'
    Neurons_Mdl = 100
    processInp = True
    norm_mdl = True

    train_dt = 40000
    epoch = 1
    Run_train = True

    # ====================== Load MNIST Dataset ==========================
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 4.
    X_test = X_test / 4.

    # =========================== Model ===================================
    Mdl = WTA(Monitors=False)
    if Run_train:
        Clean_TempFolder(Flush=False)
        Mdl.net['Syn1'].On = 1
        X_pre = Mdl.preProcess(X_data=X_train[:train_dt], preInp=processInp)

        print("================== # TRAINING MODEL # ==================")
        for ep in range(epoch):
            for idx in tqdm(range(len(X_pre)), desc='Loading ' + str(ep + 1)):
                Mdl.Norm_SynW(Norm_w=True)

                Mdl.RunModel(X_single=X_pre[idx], preInp=processInp, norm=norm_mdl, phase='Stimulus')
                Mdl.RunModel(phase='Resting')
            temp_ep = 'Temp_Ep' + str(ep + 1)
            Mdl.net.store(temp_ep, 'Temp/' + temp_ep + '.b2')
        Mdl.net.store(file_name,'Trained_Models/' + file_name + '.b2')
    else:
        Mdl.net.restore(file_name,'Trained_Models/' + file_name + '.b2')
    

    # ==================== Plots of Network Behavior ======================
    Gabor_Weight_plot(Syn1_weight=Mdl['Syn1'].w)
    plt.show()