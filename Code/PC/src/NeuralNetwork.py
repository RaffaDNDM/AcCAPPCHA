from numpy import loadtxt
import csv
import os
from termcolor import cprint
import colorama
from utility import correct_csv_file

import ctypes
hllDll = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cudart64_101.dll')
hllDll1 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cublas64_10.dll')
hllDll2 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cufft64_10.dll')
hllDll3 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\curand64_10.dll') 
hllDll4 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cusolver64_10.dll')
hllDll5 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cusparse64_10.dll')
hllDll6 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDNN\\bin\\cudnn64_7.dll')

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.utils import to_categorical


class NeuralNetwork:
    #Default files
    CSV_TRAINING = '../dat/touch/dataset.csv'
    CSV_DICT_LABELS = '../dat/touch/label_dict.csv'

    def __init__(self, labels_dict=None, training_set=None):
        #If specified a label dictionary (csv) on command line argument
        self.CSV_DICT_LABELS = correct_csv_file(labels_dict, self.CSV_DICT_LABELS)
        #If specified a training set (csv) on command line argument
        self.CSV_TRAINING = correct_csv_file(training_set, self.CSV_TRAINING)
        #Create dictionary of labels from csv file
        with open(self.CSV_DICT_LABELS) as fp:
            reader = csv.reader(fp)
            self.labels = {rows[0]:rows[1] for rows in reader}

        # Load the dataset
        dataset = loadtxt(self.CSV_TRAINING, delimiter=',')
        # Split into input (X) and input/label (y) variables
        self.X_train = dataset[:,0:-1]
        self.Y_train = dataset[:,-1]
        cprint(len(self.X_train), 'red')
        cprint(len(self.X_train[0]), 'red')
        # Define the keras model
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.X_train), output_dim=64))
        # Add a LSTM layer with 128 internal units.
        self.model.add(LSTM(66, activation='relu'))
        # Add a Dense layer with 10 units.
        self.model.add(Dense(len(self.labels), activation='relu'))
        '''
        # 
        self.model.add(Dense(len(self.labels), input_dim=len(self.X_train[0]), activation='relu'))
        self.model.add(Dense(69, activation='relu'))
        self.model.add(Dense(69, activation='relu'))
        self.model.add(Dense(69, activation='relu'))
        self.model.add(Dense(len(self.labels), activation='relu'))
        '''
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.Y_train = to_categorical(self.Y_train, len(self.labels))


    def train(self):
        end_train = int(len(self.X_train)/2)
        #Train the model
        history = self.model.fit(self.X_train[:end_train], self.Y_train[:end_train], epochs=100, batch_size=64)
        # Evaluate the model
        scores = self.model.evaluate(self.X_train, self.Y_train, verbose=0)
        print(f'{self.model.metrics_names[1]}: {scores[1]*100}%')
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

    def test(self):
        # Load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        end_train = int(len(self.X_train)/2)
        #score = loaded_model.evaluate(self.X_train[end_train:], self.Y_train[end_train:], verbose=0)
        #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


def main():
    colorama.init()
    nn = NeuralNetwork()
    nn.train()
    nn.test()


if __name__=='__main__':
    main()