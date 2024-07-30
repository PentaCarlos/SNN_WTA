from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tkinter import filedialog
from keras.datasets import mnist
from Network.Net import WTA
from Network.Tools import assign_Class, Calculate_Correct
from Network.Tools import Gabor_Weight_plot, plot_Weight, WeightDist
from Network.Tools import plot_NonSp, plot_AvrInp, plot_MissClass, plot_ConfMtx
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import sys

def Map_to_Neurons(X_spikes, Y_label):
    Inp_map = np.array(assign_Class(data=X_spikes, Y=Y_label))
    if np.sum(Inp_map) <= 0: 
        print('No Spikes in Excitatory Layer!!')
        sys.exit(0)
    return Inp_map

def NonSpikes(X_train, X_test):
    Sp_train = np.sum(X_train, axis=1)
    Sp_test = np.sum(X_test, axis=1)
    NonSp_train = [idx for idx, Zero in enumerate(Sp_train) if Zero == 0]
    NonSp_test = [idx for idx, Zero in enumerate(Sp_test) if Zero == 0]
    TrueSp_test = [idx for idx, Zero in enumerate(Sp_test) if Zero != 0]
    return NonSp_train, NonSp_test, TrueSp_test

def TrueMiss(X_output, Y_label, NonSp):
    miss_arg = [idx for idx, Img_label in enumerate(X_output) if Img_label != Y_label[idx]]
    True_miss_arg = np.setdiff1d(miss_arg, NonSp)
    return miss_arg, True_miss_arg

def Str2bool(Val_arg):
    if Val_arg == "True": return True
    elif Val_arg == "False": return False

if __name__ == "__main__":

    # ==================== Argument Initialization ========================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--load", default='Train', type=str, help="Selector for generating plots for training or model validation")
    args = vars(parser.parse_args())

    addr_folder = filedialog.askdirectory()
    file_name = []
    for file in os.listdir(addr_folder):
        if file.endswith('.b2'):
            file_name.append(file)
    # print(file_name[-2])

    test_params = np.load(addr_folder[:-5] + '/test_params.npy')
    # print(test_params)

    # =========================== Parameters ==============================
    with open('Network/params.yml', 'r') as file:
        net = yaml.safe_load(file)
    file.close()

    init_params = {
        'Gabor_filter':test_params[0],
        'Norm':test_params[1]
    }

    Net_init = {
        'Neurons':net['Net'][0],
        'Learning_Rule':net['Net'][1],
        'Nearest_Neighbor':net['Net'][2],
        'Pre_Offset':net['Net'][3],
        'Run_test':False,
        'Monitors':False
    }

    # =========================== Model ===================================
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    Mdl = WTA(Net_setup=Net_init)
    Mdl.net.restore(file_name[-2][:-3], addr_folder + '/' + file_name[-2])

    train_data = np.load(addr_folder[:-5] + '/Activity/Train/Temp_It_40000_40000.npy')
    test_data = np.load(addr_folder[:-5] + '/Activity/Test/Temp_It_40000_10000.npy')
    Inp_test = np.load(addr_folder[:-5] + '/Activity/Test/Input/Poisson_Count_10000.npy')

    Test_sum = np.array(np.sum(Inp_test, axis=1))
    Inp_map = Map_to_Neurons(X_spikes=train_data, Y_label=y_train)
    NonSp_train, NonSp_test, TrueSp_test = NonSpikes(X_train=train_data, X_test=test_data)
    correct, result, Class_idx = Calculate_Correct(data=test_data, Y=y_test, Class_Map=Inp_map)
    miss_arg, True_miss_arg = TrueMiss(X_output=result, Y_label=y_test, NonSp=NonSp_test)

    # ==================== Plots of Network Behavior ======================
    print('Gabor:', str(test_params[0]))
    print('Norm:', str(test_params[1]))
    
    WeightDist(Syn=Mdl['Syn1'])
    
    if ((Net_init['Neurons']) == 100 and (init_params['Gabor_filter'] == True)): Gabor_Weight_plot(Syn1_weight=Mdl['Syn1'].w)
    if ((Net_init['Neurons']) == 100 and (init_params['Gabor_filter'] == False)): plot_Weight(Syn1_weight=Mdl['Syn1'].w)

    plot_NonSp(Label_data=y_test[NonSp_test])
    plot_AvrInp(Sp_Inp=Test_sum, Y_data=y_test, NonSp_idx=NonSp_test, Miss_idx=miss_arg, Correct_idx=Class_idx)
    plot_MissClass(y_arr=np.bincount(result[True_miss_arg]))
    plot_ConfMtx(Sp_pred=result, Label_true=y_test, TrueSp_idx=TrueSp_test)
    plt.show(block=True)