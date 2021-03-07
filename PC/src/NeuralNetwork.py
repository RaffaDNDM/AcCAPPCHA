from numpy import loadtxt
import numpy as np
import csv
import os
from termcolor import cprint, colored
import colorama
import utility
import ctypes
import platform
import sys

#Other results are 'Darwin' for Mac and 'Linux' for Linux
if platform.system()=='Windows':
    hllDll = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cudart64_101.dll')
    hllDll1 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cublas64_10.dll')
    hllDll2 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cufft64_10.dll')
    hllDll3 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\curand64_10.dll') 
    hllDll4 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cusolver64_10.dll')
    hllDll5 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cusparse64_10.dll')
    hllDll6 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDNN\\bin\\cudnn64_7.dll')

from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.models import load_model
from sklearn.metrics import accuracy_score
import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class NeuralNetwork:
    #Default files
    DATA_FOLDER = '../dat/dataset/200_less/'
    CSV_DATASET = 'dataset.csv'
    CSV_DICT_LABELS = 'label_dict.csv'
    MODEL_NAME = 'model'

    '''
    Neural network object performs training over a new model
    or test over a trained model

    Args:
        option (int): Num corresponding to type of feature extraction
                      (Index of vector utility.OPTIONS)

        data_folder (str): Path of the folder containing the three subfolders
                           with datasets and label dictionary 
                           (a subfolder for each type of feature extraction)

    Attributes:
        OPTION (int): Num corresponding to type of feature extraction
                      (Index of vector utility.OPTIONS)

        DATA_FOLDER (str): Path of the folder containing the three subfolders
                           with datasets and label dictionary 
                           (a subfolder for each type of feature extraction)

        CSV_DATASET (str): Name of the csv file containing dataset (features+label)

        CSV_DICT_LABELS = Name of the csv file containing labels association
                          (label, index related to label)

        MODEL_NAME (str): Name of subfolder in folder DATA_FOLDER+utility.OPTIONS[OPTION]
                          that will contain trained model

        X_train (numpy.ndarray): Array of training features
        
        Y_train (numpy.ndarray): Array of training labels
        
        X_validation (numpy.ndarray): Array of validation features
        
        Y_validation (numpy.ndarray): Array of validation labels
        
        X_test (numpy.ndarray): Array of test features
        
        Y_test (numpy.ndarray): Array of test labels

        labels (dict): List of used labels

    '''

    def __init__(self, option, data_folder=None):
        self.OPTION = utility.OPTIONS[option]
        #Update folder containing csv files
        if data_folder:
            if os.path.exists(data_folder) and os.path.isdir(data_folder):
                self.DATA_FOLDER = utility.uniform_dir_path(data_folder)
                self.DATA_FOLDER += (utility.OPTIONS[option]+'/')
            else:
                cprint('[NOT EXISTING FOLDER]', 'blue', end=' ')
                print('The dataset directory', end=' ')
                cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
                print("doesn't exist")
                exit(0)

        files = os.listdir(self.DATA_FOLDER)

        if not self.CSV_DICT_LABELS in files:
            cprint('[FOLDER without files]', 'blue', end=' ')
            print('The dataset directory', end=' ')
            cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
            print("doesn't contain required csv files of dataset and labels dictionary")
            exit(0)

        #Create dictionary of labels from csv file
        with open(self.DATA_FOLDER+self.CSV_DICT_LABELS) as fp:
            reader = csv.reader(fp)
            self.labels = [row[0] for row in reader]

    def train(self):
        '''
        Train the model over training data, evaluate accuracy
        and store trained model
        '''
        
        files = os.listdir(self.DATA_FOLDER)

        if not self.CSV_DATASET in files:
            cprint('[FOLDER without files]', 'blue', end=' ')
            print('The dataset directory', end=' ')
            cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
            print("doesn't contain required csv files of dataset and labels dictionary")
            exit(0)

        #Load the dataset
        dataset = loadtxt(self.DATA_FOLDER+self.CSV_DATASET, delimiter=',')
        #Split into input (X) and input/label (y) variables
        self.X_train = dataset[0:len(dataset):2,:-1]
        self.Y_train = dataset[0:len(dataset):2, -1]
        self.X_validation = dataset[1:len(dataset):4,:-1]
        self.Y_validation = dataset[1:len(dataset):4, -1]
        self.X_test = dataset[3:len(dataset):4,:-1]
        self.Y_test = dataset[3:len(dataset):4, -1]
        print(len(self.X_train))
        
        cprint(f'\n\n{len(dataset)}', 'red', end='\n\n')
        
        
        #Define the keras model
        model = Sequential()

        #Map labels into integer values
        self.Y_train = to_categorical(self.Y_train, len(self.labels))
        self.Y_validation = to_categorical(self.Y_validation, len(self.labels))
        self.Y_test = to_categorical(self.Y_test, len(self.labels))
        
        if self.OPTION == 'spectrum':
            #Create the network with spectrogram feature as input
            model.add(Dense(1024, input_dim=len(self.X_train[0]), activation='relu'))
            model.add(Dense(len(self.labels), activation='sigmoid'))
            model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

            history = model.fit(self.X_train, self.Y_train, epochs=30, shuffle=True)
        else:
            #Create the network with touch/touch_hit feature as input
            model.add(Dense(100, input_dim=len(self.X_train[0]), activation='relu'))
            model.add(Dense(len(self.labels), activation='sigmoid'))
            model.compile(loss=CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

            #end = int(len(self.X)/2)
            epochs = 0

            #Train the model (epochs=4 less accuracy)
            if self.OPTION=='touch_hit':
                epochs=20
            else:
                epochs=30

            history = model.fit(self.X_train, self.Y_train, epochs=epochs, shuffle=True)

        # Evaluate the model
        scores = model.evaluate(self.X_train, self.Y_train, verbose=0)
        cprint(f'Training accuracy:', 'green', end='  ')
        print(f'{scores[1]*100} %')

        # Evaluate the model
        scores = model.evaluate(self.X_validation, self.Y_validation, verbose=0)
        cprint(f'Validation accuracy:', 'green', end='  ')
        print(f'{scores[1]*100} %')
        
        # Evaluate the model
        scores = model.evaluate(self.X_test, self.Y_test, verbose=0)
        cprint(f'Test accuracy:', 'green', end='  ')
        print(f'{scores[1]*100} %')

        #Save the model on the File System
        model.save(self.DATA_FOLDER+self.MODEL_NAME)

    def test(self, X):
        '''
        Load model and classify input feature X returning
        the 10 best prediction labels

        Args:
            X (numpy.array): Feature to be predicted

        Returns:
            results (list): list of label strings related to 10
                            most probable predictions
        '''
        files = os.listdir(self.DATA_FOLDER)
        
        if not self.MODEL_NAME in files:
            cprint('[FOLDER without files]', 'blue', end=' ')
            print('The dataset directory', end=' ')
            cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
            print("doesn't contain required folder of trained model")
            exit(0)

        loaded_model = load_model(self.DATA_FOLDER+self.MODEL_NAME)
        #Prediction example
        Y = loaded_model.predict(X)
        Y_indices_sort = np.argsort(Y)
        
        #10 best results (keys with highest probability)
        results = []
        
        for i in range(1, 11):
            results.append(self.labels[Y_indices_sort[0][-i]])

        return results