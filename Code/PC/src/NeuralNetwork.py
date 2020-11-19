from csv import reader

class NeuralNetwork:
    #Default files
    OUTPUT_CSV_TRAINING = '../dat/training/dataset.csv'
    OUTPUT_CSV_DICT_LABEL = '../dat/training/label_dict.csv'

    def __init__(self, training_set, labels_dict):
        #If specified a training set (csv) on command line argument
        if training_set:
            self.OUTPUT_CSV_TRAINING = training_set
        #If specified a label dictionary (csv) on command line argument
        if labels_dict:
            self.OUTPUT_CSV_DICT_LABEL = labels_dict
