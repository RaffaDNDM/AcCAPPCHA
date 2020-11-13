from termcolor import cprint
import argparse
import acquirerecord as ar
import plotwave as pwave

'''
Parser of command line arguments
'''
def args_parser():

    #Parser of command line arguments
    parser = argparse.ArgumentParser()
    
    #Initialization of needed arguments
    parser.add_argument("-plot", "-p", 
                        dest="plot", 
                        help="If specified, it plots data already acquired instead of acquire new audio files",
                        action='store_true')

    #Parse command line arguments
    args = parser.parse_args()

    return args.plot


def main():
    try:
        if args_parser():
            plot_data = pwave.Plot()
            plot_data.plot()
        else:
            acquisition = ar.AcquireAudio()
            acquisition.record()        

        cprint('\nExit from program\n', 'red', attrs=['bold'])
    
    except KeyboardInterrupt:
        cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')


if __name__=='__main__':
    main()