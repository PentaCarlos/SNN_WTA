# An SNN-WTA circuit
A spiking neural network (SNN) implementation to solve the digit-MNIST classification problem within a winner-takes-all (WTA) architecture. 

The model is coupled with three bio-inspired learning rules based on the spike timing correlation of pre- and postsynapics neurons with variations in their spike interactions.

A preprocessing stage was added to evaluate their impact and synergy with the learning rules. A Gabor filter for capturing orientation details from the input images and a norm within all the obtained Gabor images.

## Getting Started
To show the default behavior of the model settings,
```
    python -m Network.Behavior
```
A training run can be perform by the following line,
```
    python Train.py
```
Identically, after the training, we can evaluate the results with,
```
    python Validate.py
```

## Changing default settings
We can modify the parameters of the network within the command prompt as flags for the train, validate and behavior files. The flags for these files are listed below:
```
##################### Behavior.py flags ###############################
    -s   -->  "Random Seed Initialization"
    -gb  -->  "Preprocess Input data with Gabor Filter"
    -n   -->  "Applied Input Normalization after Gabor Filter"
    -p   -->  "Show the weight plots after running the training"
    -sv  -->  "Save the data behavior within a npy file"
    -r   -->  "Run Model Behavior"
```
```
##################### Train.py flags ##################################
    -s   -->  "Random Seed Initialization"
    -f   -->  "Filename of the Model to be saved"
    -gb  -->  "Preprocess Input data with Gabor Filter"
    -n   -->  "Applied Input Normalization after Gabor Filter"
    -d   -->  "Length of dataset to train our model"
    -e   -->  "Number of epoch to train our model"
    -t   -->  "Select to load the brian2 file from the temp directory"
    -p   -->  "Show the weight plots after running the training"
    -r   -->   "Run training"
```
```
##################### Validate.py flags ################################
    -s   -->  "Random Seed Initialization"
    -f   -->  "Filename of the Model to be saved"
    -gb  -->  "Preprocess Input data with Gabor Filter"
    -n   -->  "Applied Input Normalization after Gabor Filter"
    -d   -->  "Length of dataset to test our model"
    -m   -->  "Length of dataset to train our model"
    -t   -->  "Select to load the brian2 file from the temp directory"
    -p   -->  "Show the weight plots after running the training"
    -rm  -->  "Run a single presentation of the train dataset"
    -r   -->  "Run a single presentation of the test dataset"
```