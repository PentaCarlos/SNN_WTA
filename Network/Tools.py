from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import cv2

def norm_Weight(Syn, Exc_neurons):
    # Normalize Weights
    Weight_factor = 78.
    len_source = len(Syn.source)
    len_target = len(Syn.target)
    connection = np.zeros((len_source, len_target))
    connection[Syn.i, Syn.j] = Syn.w
    temp_conn = np.copy(connection)
    colSum = np.sum(temp_conn, axis=0)
    colFactor = Weight_factor/colSum
    for j in range(Exc_neurons):
        temp_conn[:,j] *= colFactor[j]
    return temp_conn[Syn.i, Syn.j]

def GaborKernel(Gb_phi='Odd', theta=[0, 45, 90, 135]):
    '''
    ksize : Kernel Window
    sigma : Std deviation of the Gaussian envelope
    theta : Orientation of the filter
    lamda : Wavelength of the sinusoidal factor
    gamma : Spatial aspect ratio (ellipticity) of the filter
    psi   : Phase offset of the filter in degrees.
    '''
    ksize, sigma, lamda, gamma = 11, 2.201, 5.6, 1 # Gabor Filters Parameters
    
    if Gb_phi == 'Odd':
        psi = np.pi/2 # Odd Gabor Filter
    elif Gb_phi == 'Even':
        psi = 0 # Even Gabor Filter
    
    filters = [cv2.getGaborKernel((ksize, ksize), sigma, np.radians(Ang), lamda, gamma, psi) for Ang in theta]
    return filters

def filterGb(Img, kernel):
    return [cv2.filter2D(src=Img, ddepth=0, kernel=k) for k in kernel]

def WeightDist(Syn, Gmax:float=1.0):
    plt.figure(figsize=(8,6))
    plt.plot(Syn.w/Gmax, '.k')
    plt.xlabel('Synaptic Connection')
    plt.ylabel('W / Gmax')
    plt.tight_layout()

    plt.figure(figsize=(8,6))
    plt.hist(Syn.w/Gmax, ec='yellow', color='k')
    plt.xlabel('W / Gmax')
    plt.tight_layout()

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

def plot_Weight(Syn1_weight):
    Weight_m = np.array(Syn1_weight).reshape((784, 100))
    dummy_mtx = np.zeros((280, 280))
    neuron = 0
    for i in range(10):
        for j in range(10):
            actual_w = Weight_m[:,neuron].reshape((28, 28))
            dummy_mtx[(i*28):(28*(i+1)), (j*28):((28*(j+1)))] = actual_w
            neuron += 1
    plt.figure(figsize=(8,6))
    plt.title('2D Receptive Field')
    plt.imshow(dummy_mtx, cmap='hot_r')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('Results/Weight/ReceptiveField.png')

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

def Count_Occurence(X_list, Cond_arr, Cond_key):
    return [x for idx, x in enumerate(X_list) if Cond_arr[idx] == Cond_key]

def get_Avr_per_Digit(Inp_data, Label_data):
    Avr_count = []
    for digit in range(10):
        Occur_arr = Count_Occurence(X_list=Inp_data, Cond_arr=Label_data, Cond_key=digit)
        if len(Occur_arr) < 1: Avr_count.append(0)
        else: Avr_count.append(np.average(Occur_arr))
    return Avr_count

def plot_AvrInp(Sp_Inp, Y_data, NonSp_idx, Miss_idx, Correct_idx, dataset_type='Test'):

    Inp_NonSp = get_Avr_per_Digit(Inp_data=Sp_Inp[NonSp_idx], Label_data=Y_data[NonSp_idx])
    Inp_Total = get_Avr_per_Digit(Inp_data=Sp_Inp, Label_data=Y_data)
    Inp_Missed = get_Avr_per_Digit(Inp_data=Sp_Inp[Miss_idx], Label_data=Y_data[Miss_idx])
    Inp_Correct = get_Avr_per_Digit(Inp_data=Sp_Inp[Correct_idx], Label_data=Y_data[Correct_idx])

    Inp_data = [Inp_NonSp, Inp_Total, Inp_Missed, Inp_Correct]
    Correct_Avr = np.average(Inp_Correct)

    x_arr = np.arange(0, 10, 1)
    plt.figure(figsize=(8, 6))
    plt.title('Average Spikes produced per Digit (' + dataset_type + ')')
    for idx in range(4):
        plt.plot(x_arr, Inp_data[idx], marker='*', markersize=10, linewidth=0)
    plt.axhline(y=Correct_Avr, color='k', linestyle='--')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.ylabel('Spike Count (Average)')
    plt.xlabel('Digit')
    plt.legend(['Non Output Spikes', 'Total', 'Missclasified', 'Correct', 'Correct (Avr)'])
    # plt.ylim(Correct_Avr-50, Correct_Avr+50)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Results/Validate/Avr_InputSp.png')

def plot_NonSp(Label_data, dataset_type='Test'):
    x_arr = np.arange(0, 10, 1)
    miss_occur = np.bincount(Label_data)
    if len(miss_occur) < 10:
        while True:
            miss_occur = np.append(miss_occur, 0)
            if len(miss_occur) >= 10: break
    print('No generated spikes within excitatory layer per label:', miss_occur)
    plt.figure(figsize=(8, 6))
    plt.title('Non Spike Count for '+ dataset_type + ' Dataset')
    plt.bar(x_arr, miss_occur, ec='yellow', color='k')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.ylabel('Frequency')
    plt.xlabel('Digit')
    plt.tight_layout()
    plt.savefig('Results/Validate/NonSp.png')

def plot_MissClass(y_arr:np.ndarray):
    # Plot the Maximum Frequency of the Missclassified Images
    x_arr = np.arange(0, 10, 1)
    plt.figure(figsize=(8, 6))
    plt.title('Missclassified Digits')
    plt.bar(x_arr, y_arr, ec='yellow', color='k')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('Results/Validate/MissClassif.png')
    print('Missclassified images per label:', y_arr)

def plot_ConfMtx(Sp_pred, Label_true, TrueSp_idx):
    Con_mtx = confusion_matrix(y_pred=Sp_pred[TrueSp_idx], y_true=Label_true[TrueSp_idx], normalize='true')
    mtx_dis = ConfusionMatrixDisplay(confusion_matrix=Con_mtx) # Display Confusion Matrix
    mtx_dis.plot(cmap=plt.cm.Blues, include_values=False)
    plt.savefig('Results/Validate/Conf_Mtx.png')

def plot_AccIt(it_lim:int=25000):
    acc_it = [np.load('Results/Accuracy/Res_Temp_It_' + str(num) + '.npy')[0] for num in np.arange(1000, it_lim+1000, 1000)]
    x_arr = np.arange(1, len(acc_it)+1, 1)
    plt.figure(figsize=(8,6))
    plt.plot(x_arr*1000, acc_it, color='purple', marker='o')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Iteration')
    plt.xlim(0, it_lim+1000)
    plt.ylim(40, 100)
    plt.legend(['pair-wise STDP'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Results/Accuracy/Acc_Iteration_LearningRule.png')