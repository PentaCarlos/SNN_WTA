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

def plot_NonSp(Label_data, dataset_type='Test'):
    x_arr = np.arange(0, 10, 1)
    miss_occur = np.bincount(Label_data)
    if len(miss_occur) < 10:
        while True:
            miss_occur = np.append(miss_occur, 0)
            if len(miss_occur) >= 10: break
    print(miss_occur)
    plt.figure(figsize=(8, 6))
    plt.title('Non Spike Count for '+ dataset_type + ' Dataset')
    plt.bar(x_arr, miss_occur, ec='yellow', color='k')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.ylabel('Frequency')
    plt.xlabel('Digit')
    plt.tight_layout()