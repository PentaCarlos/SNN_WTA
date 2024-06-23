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

