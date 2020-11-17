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

    parser.add_argument("-record", "-r", 
                        dest="record", 
                        help="If specified, it record audio during key logging",
                        action='store_true')

    parser.add_argument("-extract", "-e", 
                        dest="extract", 
                        help="If specified, it extract features from data already acquired instead of acquire new audio files",
                        action='store_true')

    parser.add_argument("-file", "-f", 
                        dest="file", 
                        help="If specified with -plot option, it refers to the name of the file inside data/ that must be plot")

    parser.add_argument("-dir", "-d", 
                    dest="dir", 
                    help="If specified with -plot option, it refers to the name of the folder with all the audios to plot together")

    parser.add_argument("-out", "-o", 
                        dest="output", 
                        help="If specified with -plot option, it's the directory in which graphics will be stored as image")

    #Parse command line arguments
    args = parser.parse_args()

    #ERROR if specified file or output but not plot option
    mandatory_options_miss = not args.plot and not args.record and not args.extract 
    no_plot_error = (not args.plot and not args.extract) and (args.file or args.output)
    specified_file_error = args.plot and args.file and not args.file.endswith('.wav')
    incompatible_options = (args.plot and args.record) or (args.extract and args.record) or (args.plot and args.extract)

    if  (no_plot_error or 
         mandatory_options_miss or
         specified_file_error or 
         incompatible_options):
        parser.print_help()
        exit(0)

    return args.plot, args.record, args.extract, args.file, args.dir, args.output


def main():
    plot_option, record_option, extract_option, filename, audio_dir, output = args_parser()

    if plot_option or extract_option:
        if plot_option:
            plot_data = pwave.Plot(False, filename= filename, audio_dir= audio_dir,output_dir=output)
            plot_data.plot()
        elif extract_option:
            plot_data = pwave.Plot(True, filename= filename, audio_dir= audio_dir,output_dir=output)
            plot_data.plot_extract()
        else:
            cprint('[ERROR]', end=' ')
            print('You cannot insert -p and -e option at the same time')

    elif record_option:
        acquisition = ar.AcquireAudio(audio_dir, 1)
        acquisition.record()        


    cprint('\nExit from program\n', 'red', attrs=['bold'])

if __name__=='__main__':
    main()
