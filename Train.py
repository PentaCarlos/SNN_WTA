from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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
    
    # ==================== Argument Initialization ========================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", default=0, type=int, help="Random Seed Initialization")
    parser.add_argument("-f", "--filename", default="default", type=str, help="Filename of the Model to be saved")
    parser.add_argument("-gb", "--gabor", default=True, type=bool, help="Preprocess Input data with Gabor Filter")
    parser.add_argument("-n", "--norm", default=True, type=bool, help="Applied Input Normalization after Gabor Filter")
    parser.add_argument("-d", "--dataset", default=1000, type=int, help="Length of dataset to train our model")
    parser.add_argument("-e", "--epoch", default=5, type=int, help="Number of epoch to train our model")
    parser.add_argument("-r", "--run", default=True, type=bool, help="Run training")
    args = vars(parser.parse_args())

    # =========================== Parameters ==============================
    init_params = {
        'Random_Seed':args['seed'],
        'Filename':args['filename'],
        'Gabor_filter':args['gabor'],
        'Norm':args['norm'],
        'Train_dt':args['dataset'],
        'Epoch':args['epoch'],
        'Run_train':args['run']
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
        Mdl.Init_State()
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
    # if Net_init['Neurons'] == 100: Gabor_Weight_plot(Syn1_weight=Mdl['Syn1'].w)
    # plt.show()