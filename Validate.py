from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from keras.datasets import mnist
from tqdm import tqdm
from Network.Net import WTA, SpikeMonitor, seed
from Network.Tools import plot_NonSp, plot_AvrInp, plot_MissClass, plot_ConfMtx
from brian2.units import *

import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys

def get_Spikes(X_data:np.ndarray, init_params:dict, Net_params:dict, presen_stg:str='Train', Run_trial:bool=False):

    seed(init_params['Random_Seed'])

    if presen_stg == 'Train': 
        dir_data = 'Activity/Train/'
        state_title = 'MAPPING'
    elif presen_stg == 'Test': 
        dir_data = 'Activity/Test/'
        state_title = 'VALIDATING'

    if init_params['Load_Temp'] == True: addr_load = 'Temp/'
    else: addr_load = 'Trained_Models/'

    if Run_trial:
        print("================== # "+ state_title +" MODEL # ==================")
        Mdl = WTA(Net_setup=Net_params)
        Mdl.Init_State()
        X_pre = Mdl.preProcess(X_data=X_data, preInp=init_params['Gabor_filter'])
        Train_Sp, Input_Sp = [], []
        for idx in tqdm(range(len(X_pre)), desc='Validating'):
            Mdl.net.restore(init_params['Filename'], filename=addr_load + init_params['Filename'] + '.b2')
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

def Str2bool(Val_arg):
    if Val_arg == "True": return True
    elif Val_arg == "False": return False

if __name__ == "__main__":
    
    # ==================== Argument Initialization ========================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", default=0, type=int, help="Random Seed Initialization")
    parser.add_argument("-f", "--filename", default="default", type=str, help="Filename of the Model to be saved")
    parser.add_argument("-t", "--temp", default=False, type=Str2bool, help="Select to load the brian2 file from the temp directory")
    parser.add_argument("-gb", "--gabor", default=True, type=Str2bool, help="Preprocess Input data with Gabor Filter")
    parser.add_argument("-n", "--norm", default=True, type=Str2bool, help="Applied Input Normalization after Gabor Filter")
    parser.add_argument("-m", "--train_dt", default=1000, type=int, help="Length of dataset to train our model")
    parser.add_argument("-d", "--test_dt", default=10000, type=int, help="Length of dataset to test our model")
    parser.add_argument("-p", "--plot", default=False, type=Str2bool, help="Show the result plots after running the Validation process")
    parser.add_argument("-rm", "--run_train", default=True, type=Str2bool, help="Run a single presentation of the train dataset")
    parser.add_argument("-r", "--run_test", default=True, type=Str2bool, help="Run a single presentation of the test dataset")
    args = vars(parser.parse_args())

    # =========================== Parameters ==============================
    with open('Network/params.yml', 'r') as file:
        net = yaml.safe_load(file)
    file.close()

    Validate_params = {
        'Random_Seed':args['seed'],
        'Filename':args['filename'],
        'Load_Temp':args['temp'],
        'Gabor_filter':args['gabor'],
        'Norm':args['norm'],
        'Train_dt':args['train_dt'],
        'Test_dt':args['test_dt'],
        'Run_train':args['run_train'],
        'Run_test':args['run_test']
    }

    Net_init = {
        'Neurons':net['Net'][0],
        'Learning_Rule':net['Net'][1],
        'Nearest_Neighbor':net['Net'][2],
        'Pre_Offset':net['Net'][3],
        'Run_test':net['Validate'][0],
        'Monitors':net['Validate'][1]
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
    Test_sum = np.array(np.sum(Inp_test, axis=1))
    NonSp_train = [idx for idx, Zero in enumerate(Sp_train) if Zero == 0]
    NonSp_test = [idx for idx, Zero in enumerate(Sp_test) if Zero == 0]
    TrueSp_test = [idx for idx, Zero in enumerate(Sp_test) if Zero != 0]

    # ======================= Print Accuracy Report =========================
    correct, result, Class_idx = Calculate_Correct(data=test_data, Y=y_test, Class_Map=Inp_map)
    accuracy_r = (correct/test_dt) * 100
    miss_arg = [idx for idx, Img_label in enumerate(result) if Img_label != y_test[idx]]
    True_miss_arg = np.setdiff1d(miss_arg, NonSp_test)
    np.save('Results/Accuracy/Res_' + Validate_params['Filename'], [accuracy_r, train_dt, (len(NonSp_train)/train_dt)*100, (len(NonSp_test)/test_dt)*100])

    print('=============== ' + Validate_params['Filename'] + ' ===============')
    print('-----Excitatory Neuronal Layer Map-----')
    print(Inp_map)
    print('Correct Classified: ' + str(correct))
    print('Missclassified: ' + str(test_dt - correct))
    print('Accuracy: ' + str(accuracy_r) + ' %')
    print('No Spikes Train Pres: ' + str((len(NonSp_train)/train_dt)*100) + ' %')
    print('No Spikes Test Eval: ' + str((len(NonSp_test)/test_dt)*100) + ' %')

    # ==================== Plots of Network Behavior ======================
    cycle_plots = args['plot']
    plot_NonSp(Label_data=y_test[NonSp_test], dataset_type='Test')
    plot_AvrInp(Sp_Inp=Test_sum, Y_data=y_test, NonSp_idx=NonSp_test, Miss_idx=miss_arg, Correct_idx=Class_idx)
    plot_MissClass(y_arr=np.bincount(result[True_miss_arg]))
    plot_ConfMtx(Sp_pred=result, Label_true=y_test, TrueSp_idx=TrueSp_test)
    
    # it_lim = 25000
    # acc_it = [np.load('Results/Accuracy/Res_Temp_It_' + str(num) + '.npy')[0] for num in np.arange(1000, it_lim+1000, 1000)]
    # x_arr = np.arange(1, len(acc_it)+1, 1)
    # plt.figure(figsize=(8,6))
    # plt.plot(x_arr*1000, acc_it, color='purple', marker='o')
    # plt.ylabel('Accuracy [%]')
    # plt.xlabel('Iteration')
    # plt.xlim(0, it_lim+1000)
    # plt.ylim(40, 100)
    # plt.legend(['pair-wise STDP'])
    # plt.grid(True)
    # plt.tight_layout()
    plt.show(block=cycle_plots)