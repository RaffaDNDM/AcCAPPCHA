import NeuralNetwork as nn
import colorama
import argparse
from termcolor import cprint
import utility

def args_parser():
    '''
    Parser of command line arguments
    
    Returns:
        plot (bool): If True, it plots data already acquired 
                     instead of acquire new audio files
        
        record (bool): If True, it record audio during key logging
        
        extract (bool): If True, it extract features from data already
                        acquired instead of acquire new audio files
        
        file (str): Path of the file that you want to plot or from
                    which you want to extract features
        
        dir (str): Path of the folder where:
                   [-r option]
                     there will be stored audio files acquired
                     from the recorder and the keylogger
                   [-p option and -e option]
                     there are already recored audio on which
                     extraction and plot will be applied

        output (str): Path of the output folder in which graphics
                      will be stored as image
    '''
    #Parser of command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-dir","-d",
                        dest="dir", 
                        help="""Path of the folder that contains the csv files of 
                                dataset and labels dictionary and contains/will
                                contain the json file of model""")

    #Parse command line arguments
    args = parser.parse_args()

    return args.dir


def main():
    #Initialize colored prints
    colorama.init()
    #Parser of command line arguments
    folder = args_parser()

    option = utility.select_option_feature()
    count = 0

    if option==3:
        while count<option:
            net = nn.NeuralNetwork(option,folder)
            cprint(utility.LINE, 'blue')
            print('Number of labels:', end=' ')
            cprint(len(net.labels), 'green')
            cprint(utility.LINE, 'blue')
            cprint(utility.LINE, 'blue')

            for key in net.labels.keys():
                cprint(f'{key}:', 'red', end=' ')
                cprint(f'{net.labels[key]}', 'yellow')

            cprint(utility.LINE, 'blue')
            cprint(utility.LINE, 'blue')
            net.train()            
    else:
        net = nn.NeuralNetwork(option,folder)
        cprint(utility.LINE, 'blue')
        print('Number of labels:', end=' ')
        cprint(len(net.labels), 'green')
        cprint(utility.LINE, 'blue')
        input('Type ENTER to see them in details')
        cprint(utility.LINE, 'blue')

        for key in net.labels.keys():
            cprint(f'{key}:', 'red', end=' ')
            cprint(f'{net.labels[key]}', 'yellow')

        cprint(utility.LINE, 'blue')
        input('Type ENTER to train the model')
        cprint(utility.LINE, 'blue')
        net.train()


if __name__=='__main__':
    main()