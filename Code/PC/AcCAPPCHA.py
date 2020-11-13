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

        #Initialization of needed arguments
    parser.add_argument("-file", "-f", 
                        dest="file", 
                        help="If specified with -plot option, it refers to the name of the file inside data/ that must be plot")

    #Parse command line arguments
    args = parser.parse_args()

    #ERROR if specified file but not plot option
    file_with_no_plot = not args.plot and args.file
    specified_file_error = args.plot and args.file and not args.file.endswith('.wav')

    if  file_with_no_plot or specified_file_error:
        parser.print_help()
        exit(0)

    return args.plot, args.file


def main():
    plot_option, filename = args_parser()

    if plot_option:
        print(filename)
        plot_data = pwave.Plot(filename)
        plot_data.plot()
    else:
        acquisition = ar.AcquireAudio(100)
        acquisition.record()        

    cprint('\nExit from program\n', 'red', attrs=['bold'])

if __name__=='__main__':
    main()
