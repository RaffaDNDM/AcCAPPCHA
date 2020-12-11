from termcolor import cprint
import argparse
import AcquireAudio as ar
import PlotExtract as pe
import colorama
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
    
    #Initialization of needed arguments
    parser.add_argument("-plot", "-p", 
                        dest="plot", 
                        help="""If specified, it plots data already acquired instead
                                of acquire new audio files""",
                        action='store_true')

    parser.add_argument("-record", "-r", 
                        dest="record", 
                        help="If specified, it record audio during key logging",
                        action='store_true')

    parser.add_argument("-extract", "-e", 
                        dest="extract", 
                        help="""If specified, it extract features from data already
                                acquired instead of acquire new audio files""",
                        action='store_true')

    parser.add_argument("-zoom", "-z", 
                        dest="zoom", 
                        help="""If specified, it extract plot or extract features
                                showing zoomed graphics""",
                        action='store_true')
    
    parser.add_argument("-file", "-f", 
                        dest="file", 
                        help="""Path of the file that you want to plot or from
                                which you want to extract features""")

    parser.add_argument("-dir", "-d", 
                    dest="dir", 
                    help="""Path of the folder where:
                            [-r option]
                              there will be stored audio files acquired
                              from the recorder and the keylogger
                            [-p option and -e option]
                              there are already recored audio on which
                              extraction and plot will be applied""")

    parser.add_argument("-out", "-o", 
                        dest="output", 
                        help="""Path of the output folder in which graphics
                                will be stored as image""")

    #Parse command line arguments
    args = parser.parse_args()

    #ERROR if specified file or output but not plot option
    mandatory_options_miss = not args.plot and not args.record and not args.extract 
    no_plot_error = (not args.plot and not args.extract) and (args.file or args.output)
    specified_file_error = args.plot and args.file and not args.file.endswith('.wav')
    incompatible_options = (args.plot and args.record) or (args.extract and args.record) \
                            or (args.plot and args.extract)

    if  (no_plot_error or 
         mandatory_options_miss or
         specified_file_error or 
         incompatible_options):
        parser.print_help()
        exit(0)

    return args.plot, args.record, args.extract, args.zoom, args.file, args.dir, args.output


def select_option_feature():
    check = True
    option=-1

    while check:
        try:
            cprint(f'Select which type of features you want to use:\n{utility.LINE}', 'blue')
                
            for i in range(0, len(utility.OPTIONS)):
                cprint(f'{i})', 'yellow', end=' ')
                print(f'{utility.OPTIONS[i]}')

            cprint(f'3)', 'yellow', end=' ')
            print(f'All the features')
            cprint(f'{utility.LINE}', 'blue')

            option = int(input())
            if option >= 0 and option <= len(utility.OPTIONS):
                check = False

        except ValueError:
            cprint('[VALUE ERROR]', 'color', end=' ')
            print('Insert a value of them specified in menu')

    return option


def main():
    '''
    Main function initializes the acquisition of audios
    and plot or extraction of features from wav files 
    '''
    colorama.init()
    plot_option, record_option, extract_option, zoom, filename, audio_dir, output = args_parser()
    print(audio_dir)

    #Plot and extraction option
    if plot_option or extract_option:
        #Plot
        if plot_option:
            analysis_data = pe.PlotExtract(filename= filename, audio_dir= audio_dir,output_dir=output)
            analysis_data.plot(zoom)
        #Extraction
        elif extract_option:
            option = utility.select_option_feature()
            analysis_data = pe.PlotExtract(filename= filename, audio_dir= audio_dir, output_dir=output)
            if option == len(utility.OPTIONS):
                for i in range(0, option):
                    analysis_data.extract(i)
            else:
                analysis_data.extract(option)
        #ERROR
        else:
            cprint('[ERROR]', end=' ')
            print('You cannot insert -p and -e option at the same time')

    #Record acquisition
    elif record_option:
        acquisition = ar.AcquireAudio(audio_dir, 1)
        acquisition.record()        

    cprint('\nExit from program\n', 'red', attrs=['bold'])


#If program is not imported by other files
#but run as main program
if __name__=='__main__':
    main()
