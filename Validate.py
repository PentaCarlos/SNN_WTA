from keras.datasets import mnist
from tqdm import tqdm
from Network.Net import WTA, SpikeMonitor, seed
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np
import sys

def get_Spikes(X_data:np.ndarray, init_params:dict, Net_params:dict, presen_stg:str='Train', Run_trial:bool=False):

    seed(init_params['Random_Seed'])

    if presen_stg == 'Train': dir_data = 'Activity/Train/'
    elif presen_stg == 'Test': dir_data = 'Activity/Test/'

    if Run_trial:
        Mdl = WTA(Net_setup=Net_params)
        X_pre = Mdl.preProcess(X_data=X_data, preInp=init_params['Gabor_filter'])
        Train_Sp, Input_Sp = [], []
        Mdl.net.restore(init_params['Filename'], filename='Trained_Models/' + init_params['Filename'] + '.b2')
        for idx in tqdm(range(len(X_pre)), desc='Validating'):
            # Rate Monitor for Counting Spikes
            mon = SpikeMonitor(Mdl.net['Exc'], name='CountSp')
            Mdl.net.add(mon)
            if presen_stg == 'Test':
                monInp = SpikeMonitor(Mdl.net['Input'], name='Count_InpSp')
                Mdl.net.add(monInp)

            Mdl.Norm_SynW(Norm_w=True)
            Mdl.RunModel(X_single=X_pre[idx], preInp=init_params['Gabor_filter'], norm=init_params['Norm'], phase='Stimulus')
            
            Train_Sp.append(np.array(mon.count, dtype=np.int8))
            if presen_stg == 'Test': Input_Sp.append(np.array(monInp.count, dtype=np.int8))
            
            Mdl.RunModel(phase='Resting')
            Mdl.net.remove(Mdl.net['CountSp'])
            if presen_stg == 'Test': Mdl.net.remove(Mdl.net['Count_InpSp'])
        np.save(dir_data + init_params['Filename'] + '_' + str(len(X_data)), Train_Sp)
        if presen_stg == 'Test': np.save('Activity/Test/Input/Poisson_Count_' + str(len(X_data)), Input_Sp)
    else:
        Train_Sp = np.load(dir_data + init_params['Filename'] + '_' + str(len(X_data)) + '.npy')
        print('Loaded Excitatory Spike Data Shape: ' + str(np.array(Train_Sp).shape))

        if presen_stg == 'Test':
            Input_Sp = np.load('Activity/Test/Input/Poisson_Count_' + str(len(X_data)) + '.npy')
            print('Loaded Input Count Data Shape: ' + str(np.array(Input_Sp).shape))
    if presen_stg == 'Test':
        return Train_Sp, Input_Sp
    else:
        return Train_Sp

def assign_Class(data, Y):
    Y = np.array(Y)
    data_test = np.array(data)
    assignments = np.ones(len(data_test[0,:])) * -1 # initialize them as not assigned
    input_nums = np.asarray(Y[:len(data)])
    maximum_rate = [0] * len(data_test[0,:])    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            # Sum found index MNIST value (row) to get the average of spikes (col) per MNIST input
            rate = np.sum(data_test[input_nums == j], axis = 0) / num_inputs
        for i in range(len(data_test[0,:])):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

def Calculate_Correct(data, Y, Class_Map):
    correct, result, correct_class_idx = 0, [], []
    data = np.array(data)
    Class_Map = np.array(Class_Map)
    for idx in range(len(data)):
        max_val_idx = np.argmax(data[idx,:])
        assign_val = int(Class_Map[max_val_idx])
        result.append(assign_val)
        if assign_val == Y[idx]: 
            correct += 1
            correct_class_idx.append(idx)
    return correct, np.array(result), correct_class_idx

if __name__ == "__main__":
    
    # =========================== Parameters ==============================
    
    Validate_params = {
        'Random_Seed':0,
        'Filename':'pairSTDP_NN',
        'Gabor_filter':True,
        'Norm':True,
        'Train_dt':1000,
        'Test_dt':10000,
        'Run_train':False,
        'Run_test':False
    }

    Net_init = {
        'Neurons':100,
        'Learning_Rule':'pair_STDP',
        'Nearest_Neighbor':True,
        'Run_test':True,
        'Monitors':False
    }

    # ====================== Load MNIST Dataset ==========================
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 4.
    X_test = X_test / 4.

    # ================== Get Train Activity Data ==========================
    train_dt = Validate_params['Train_dt']
    Run_train = Validate_params['Run_train']
    train_data = get_Spikes(X_data=X_train[:train_dt], init_params=Validate_params, Net_params=Net_init, presen_stg='Train', Run_trial=Run_train)

    # =================== Assign Labels to Data ===========================
    Inp_map = np.array(assign_Class(data=train_data, Y=y_train))
    if np.sum(Inp_map) <= 0: 
        print('No Spikes in Excitatory Layer!!')
        sys.exit(0)
    
    # =================== Get Test Activity Data ===========================
    test_dt = Validate_params['Test_dt']
    Run_test = Validate_params['Run_test']
    test_data, Inp_test = get_Spikes(X_data=X_test[:test_dt], init_params=Validate_params, Net_params=Net_init, presen_stg='Test', Run_trial=Run_test)

    # =================== Count Img with No Spike ===========================
    Sp_train = np.sum(train_data, axis=1)
    Sp_test = np.sum(test_data, axis=1)
    NonSp_train = [idx for idx, Zero in enumerate(Sp_train) if Zero == 0]
    NonSp_test = [idx for idx, Zero in enumerate(Sp_test) if Zero == 0]
    TrueSp_test = [idx for idx, Zero in enumerate(Sp_test) if Zero != 0]

    # ======================= Print Accuracy Report =========================
    correct, result, Class_idx = Calculate_Correct(data=test_data, Y=y_test, Class_Map=Inp_map)
    accuracy_r = (correct/test_dt) * 100
    print('=============== ' + Validate_params['Filename'] + ' ===============')
    print('-----Excitatory Neuronal Layer Map-----')
    print(Inp_map)
    print('Correct Classified: ' + str(correct))
    print('Missclassified: ' + str(test_dt - correct))
    print('Accuracy: ' + str(accuracy_r) + ' %')
    print('No Spikes Train Pres: ' + str((len(NonSp_train)/train_dt)*100) + ' %')
    print('No Spikes Test Eval: ' + str((len(NonSp_test)/test_dt)*100) + ' %')