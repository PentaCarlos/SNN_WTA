from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from keras.datasets import mnist
from tqdm import tqdm
from Network.Net import WTA, seed
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import yaml
import os

def Gabor_Weight_plot(Syn1_weight):
    Weight_m = np.array(Syn1_weight).reshape((784, 100))
    init_val = 0
    pxl_val = 196
    for orientation in [0, 45, 90, 135]:
        dummy_mtx = np.zeros((140, 140))
        neuron = 0
        for i in range(10):
            for j in range(10):
                actual_w = Weight_m[init_val:pxl_val,neuron].reshape((14, 14))
                dummy_mtx[(i*14):(14*(i+1)), (j*14):((14*(j+1)))] = actual_w
                neuron += 1
        plt.figure(figsize=(8,6))
        plt.title('Gabor prefered orientation of ' + str(orientation) + ' Degrees')
        plt.imshow(dummy_mtx, cmap='hot_r')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('Results/Weight/Gb_Angle' + str(orientation) + '.png')
        init_val = pxl_val
        pxl_val += 196

def Clean_TempFolder(Flush:bool=False):
    if Flush:
        file_list = glob('Temp/*.b2')
        for temp_file in file_list:
            os.remove(temp_file)

def Str2bool(Val_arg):
    if Val_arg == "True": return True
    elif Val_arg == "False": return False

if __name__ == "__main__":
    
    # ==================== Argument Initialization ========================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", default=0, type=int, help="Random Seed Initialization")
    parser.add_argument("-f", "--filename", default="default", type=str, help="Filename of the Model to be saved")
    parser.add_argument("-gb", "--gabor", default=True, type=Str2bool, help="Preprocess Input data with Gabor Filter")
    parser.add_argument("-n", "--norm", default=True, type=Str2bool, help="Applied Input Normalization after Gabor Filter")
    parser.add_argument("-d", "--dataset", default=1000, type=int, help="Length of dataset to train our model")
    parser.add_argument("-e", "--epoch", default=5, type=int, help="Number of epoch to train our model")
    parser.add_argument("-p", "--plot", default=False, type=Str2bool, help="Show the weight plots after running the training")
    parser.add_argument("-r", "--run", default=True, type=Str2bool, help="Run training")
    args = vars(parser.parse_args())

    # =========================== Parameters ==============================
    with open('Network/params.yml', 'r') as file:
        net = yaml.safe_load(file)
    file.close()
    
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
        'Neurons':net['Net'][0],
        'Learning_Rule':net['Net'][1],
        'Nearest_Neighbor':net['Net'][2],
        'Run_test':net['Train'][0],
        'Monitors':net['Train'][1]
    }

    # ====================== Load MNIST Dataset ==========================
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 4.
    X_test = X_test / 4.

    # =========================== Model ===================================
    seed(init_params['Random_Seed'])
    Mdl = WTA(Net_setup=Net_init)
    if init_params['Run_train']:
        print("================== # TRAINING MODEL # ==================")
        Clean_TempFolder(Flush=True)
        Mdl.Init_State()
        X_pre = Mdl.preProcess(X_data=X_train[:init_params['Train_dt']], preInp=init_params['Gabor_filter'])

        for ep in range(init_params['Epoch']):
            for idx in tqdm(range(len(X_pre)), desc='Loading ' + str(ep + 1)):
                it_counter = idx + 1
                Mdl.Norm_SynW(Norm_w=True)

                Mdl.RunModel(X_single=X_pre[idx], preInp=init_params['Gabor_filter'], norm=init_params['Norm'], phase='Stimulus')
                Mdl.RunModel(phase='Resting')

                if it_counter <= 5000:
                    if it_counter % 1000 == 0:
                        temp_ep = 'Temp_It_' + str(it_counter)
                        Mdl.net.store(temp_ep, 'Temp/' + temp_ep + '.b2')
                else:
                    if it_counter % 5000 == 0:
                        temp_ep = 'Temp_It_' + str(it_counter)
                        Mdl.net.store(temp_ep, 'Temp/' + temp_ep + '.b2')
        Mdl.net.store(init_params['Filename'],'Trained_Models/' + init_params['Filename'] + '.b2')
        np.save('Temp/Homeo/V_thr', Mdl.get_HomeoThr())
    else:
        Mdl.net.restore(init_params['Filename'],'Trained_Models/' + init_params['Filename'] + '.b2')
    

    # ==================== Plots of Network Behavior ======================
    cycle_plots = args['plot']
    if ((Net_init['Neurons']) == 100 and (args['gabor'] == True)): Gabor_Weight_plot(Syn1_weight=Mdl['Syn1'].w)
    plt.show(block=cycle_plots)