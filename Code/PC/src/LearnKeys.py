import NeuralNetwork as nn
import colorama
import argparse

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
    
    #Initialization of needed arguments
    parser.add_argument("-test", 
                        dest="test", 
                        help="""If specified, it performs test on specified
                                dataset using csv model of trained NN""",
                        action='store_true')
    
    parser.add_argument("-dir","-d",
                        dest="dir", 
                        help="""Path of the folder that contains the csv files of 
                                dataset and labels dictionary and contains/will
                                contain the json file of model""")

    #Parse command line arguments
    args = parser.parse_args()

    return args.test, args.dir


def main():
    #Initialize colored prints
    colorama.init()
    #Parser of command line arguments
    test_mode, folder = args_parser()

    net = nn.NeuralNetwork(folder)

    if test_mode:
        net.test()
    else:
        net.train()

if __name__=='__main__':
    main()