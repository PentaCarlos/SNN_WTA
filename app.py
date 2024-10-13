from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from keras.datasets import mnist
from Network.Net import WTA, SpikeMonitor, seed
from Validate import assign_Class
from brian2.utils import *

import sys
import cv2
import yaml
import numpy as np
import gradio as gr

def get_Spikes(X_inp:np.ndarray, init_params:dict, Net_params:dict, Run_trial:bool=False):
    
    seed(init_params['Random_Seed'])

    if Run_trial:
        Mdl = WTA(Net_setup=Net_params)
        Mdl.Init_State()
        X_pre = Mdl.preProcess(X_data=np.array([X_inp]), preInp=init_params['Gabor_filter'])

        Mdl.net.restore(init_params['Filename'], filename= 'Trained_Models/' + init_params['Filename'] + '.b2')
        mon = SpikeMonitor(Mdl.net['Exc'], name='CountSp')
        Mdl.net.add(mon)

        Mdl.Norm_SynW(Norm_w=True)
        Mdl.RunModel(X_single=X_pre[0], preInp=init_params['Gabor_filter'], norm=init_params['Norm'], phase='Stimulus')

        SpCount = np.array(mon.count, dtype=np.int8)

        Mdl.RunModel(phase='Resting')
        Mdl.net.remove(Mdl.net['CountSp'])
        return SpCount
    else:
        Train_Sp = np.load('Activity/Train/' + init_params['Filename'] + '_' + str(init_params['Train_dt']) + '.npy')
        Test_Sp = np.load('Activity/Test/' + init_params['Filename'] + '_' + str(init_params['Test_dt']) + '.npy')
        return Train_Sp, Test_Sp

def Str2bool(Val_arg):
    if Val_arg == "True": return True
    elif Val_arg == "False": return False


if __name__ == "__main__":
    
    # ==================== Argument Initialization ========================
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--seed", default=0, type=int, help="Random Seed Initialization")
    parser.add_argument("-f", "--filename", default="pair_STDP", type=str, help="Filename of the Model to be saved")
    parser.add_argument("-gb", "--gabor", default=True, type=Str2bool, help="Preprocess Input data with Gabor Filter")
    parser.add_argument("-n", "--norm", default=True, type=Str2bool, help="Applied Input Normalization after Gabor Filter")
    parser.add_argument("-m", "--train_dt", default=40000, type=int, help="Length of dataset which it was trained the model")
    parser.add_argument("-d", "--test_dt", default=10000, type=int, help="Length of dataset to test our model")
    parser.add_argument("-r", "--run", default=True, type=Str2bool, help="Run a single trial of the model")
    args = vars(parser.parse_args())

    # =========================== Parameters ==============================
    with open('Network/params.yml', 'r') as file:
        net = yaml.safe_load(file)
    file.close()

    Validate_params = {
        'Random_Seed':args['seed'],
        'Filename':args['filename'],
        'Gabor_filter':args['gabor'],
        'Norm':args['norm'],
        'Train_dt':args['train_dt'],
        'Test_dt':args['test_dt'],
        'Run_Model':args['run'],
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


    TrainSp, TestSp = get_Spikes(X_inp=None, init_params=Validate_params, Net_params=Net_init)
    # print(np.array(TrainSp).shape)
    # print(np.array(TestSp).shape)
    Receptive_Field = np.array(assign_Class(data=TrainSp, Y=y_train))
    if np.sum(Receptive_Field) <= 0: 
        print('No Spikes in Excitatory Layer!!')
        sys.exit(0)
    

    def predict(img):
        img_red = cv2.resize(img['composite'], (28, 28))
        Sp_input = get_Spikes(X_inp=img_red[:,:,3], init_params=Validate_params, Net_params=Net_init, Run_trial=Validate_params['Run_Model'])
        max_firing_rate = np.argmax(Sp_input)
        assign_label = int(Receptive_Field[max_firing_rate])
        return assign_label

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Sketchpad(),
        outputs=gr.Label()
    )

    demo.launch()