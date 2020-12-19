from termcolor import cprint, colored
import os
import sys
import numpy as np
from numpy import loadtxt
from scipy.io import wavfile
import pyaudio
import progressbar
import colorama
from csv import reader, writer
import ctypes
import platform

OPTIONS = ['touch', 'touch_hit', 'spectrum']
LINE = '_____________________________________________________'

#Other results are 'Darwin' for Mac and 'Linux' for Linux
if platform.system()=='Windows':
    hllDll = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cudart64_101.dll')
    hllDll1 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cublas64_10.dll')
    hllDll2 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cufft64_10.dll')
    hllDll3 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\curand64_10.dll') 
    hllDll4 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cusolver64_10.dll')
    hllDll5 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDA\\v10.1\\bin\\cusparse64_10.dll')
    hllDll6 = ctypes.WinDLL('D:\\Programs\\NVIDIA\\CUDNN\\bin\\cudnn64_7.dll')

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D

CHANNELS = 2
RATE = 44100
FORMAT = pyaudio.paInt16

def swap(m, n):
    '''
    Invert order of args

    Args:
        m (int): first number
        n (int): second number

    Returns:
        n (int): second number
        m (int): first number
    '''
    return n, m


def uniform_dir_path(directory):
    '''
    Return directory path with '/' at the end

    Args:
        directory (str): directory path that you want to uniform

    Returns:
        directory (str): modified directory path that ends with '/'
    '''
    if directory.endswith('/') or directory.endswith('\\'):
        return directory
    else:
        return directory+'/'


def key_definition(key):
    '''
    Evaluate which key has been pressed

    Args:
        key (key): pynput key

    Returns:
        key_string (str): string that correspond to the pressed key
    '''
    #Obtain string of key inserted
    try:
        key_string = str(key.char)
    except AttributeError:
        #Special key pressed
        if key == key.alt:
            key_string= 'ALT'
        elif key == key.alt_gr:
            key_string= 'ALT_GR'
        elif key == key.backspace:
            key_string= 'BACKSPACE'
        elif key == key.caps_lock:
            key_string= 'CAPS_LOCK'
        elif key == key.ctrl_l: 
            key_string = 'CTRL'
        elif key == key.ctrl_r:
            key_string= 'CTRL_R'
        #elif key == key.cmd or key.cmd_r or key.cmd_l:
        #    key_string= 'CMD'
        elif key == key.delete:
            key_string= 'DELETE'
        elif key == key.down:
            key_string= 'DOWN'
        #Fn tast disable in Dell PC
        elif key == key.f1:
            key_string= 'F1'
        elif key == key.f2:
            key_string= 'F2'
        elif key == key.f3:
            key_string= 'F3'
        elif key == key.f4:
            key_string= 'F4'
        elif key == key.f5:
            key_string= 'F5'
        elif key == key.f6:
            key_string= 'F6'
        elif key == key.f7:
            key_string= 'F7'
        elif key == key.f8:
            key_string= 'F8'
        elif key == key.f9:
            key_string= 'F9'
        elif key == key.f10:
            key_string= 'F10'
        elif key == key.f11:
            key_string= 'F11'
        elif key == key.f12:
            key_string= 'F12'
        #elif key == key.f13:
        #    key_string= 'F13'
        #elif key == key.f14:
        #    key_string= 'F14'
        #elif key == key.f15:
        #    key_string= 'F15'
        #elif key == key.f16:
        #    key_string= 'F16'
        #elif key == key.f17:
        #    key_string= 'F17'
        #elif key == key.f18:
        #    key_string= 'F18'
        #elif key == key.f19:
        #    key_string= 'F19'
        #elif key == key.f20:
        #    key_string= 'F20'
        elif key == key.end:
            key_string= 'END'
        elif key == key.esc:
            key_string= 'ESC'
        elif key == key.enter:
            key_string= 'ENTER'
        elif key == key.home:
            key_string= 'HOME'
        elif key == key.insert:
            key_string= 'INSERT'
        elif key == key.left:
            key_string= 'LEFT'
        elif key == key.menu:
            key_string= 'MENU'
        elif key == key.num_lock:
            key_string= 'NUM_LOCK'
        elif key == key.page_down:
            key_string= 'PAGE_DOWN'
        elif key == key.page_up:
            key_string= 'PAGE_UP'
        elif key == key.pause:
            key_string= 'PAUSE'
        elif key == key.print_screen:
            key_string= 'PRINT_SCREEN'
        elif key == key.right:
            key_string= 'RIGHT'
        elif key == key.scroll_lock:
            key_string= 'SCROLL_LOCK'
        elif key == key.space:
            key_string = 'SPACE'
        elif key == key.tab:
            key_string= 'TAB'
        elif key == key.up:
            key_string= 'UP'
        elif key == key.shift or key.shift_r or key.shift_l:
            key_string= 'SHIFT'
        else:
            key_string = str(key)

    if key_string=='.':
        key_string ='POINT'
    elif key_string=='/':
        key_string ='SLASH'
    elif key_string=='\\':
        key_string ='BACKSLASH'
    elif key_string=='*':
        key_string ='STAR'
    elif key_string=='+':
        key_string ='PLUS'
    elif key_string=='-':
        key_string ='MINUS'
    elif key_string==',':
        key_string ='COMMA'
    elif key_string=="'":
        key_string ='APOSTROPHE'
    elif key_string==">":
        key_string ='GREATER'
    elif key_string=="<":
        key_string ='LOWER'

    return key_string

def getchar():
    # Returns a single character from standard input
    ch = ''
    if os.name == 'nt': # how it works on windows
        import msvcrt
        ch = msvcrt.getwch()
    else:
        import tty, termios, sys
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		
    if ord(ch) == 3: 
        raise KeyboardInterrupt # handle ctrl+C

    return ch

def correct_csv_file(csv_file, default_file):
    if csv_file:
        if os.path.isfile(csv_file) and csv_file.endswith('.csv'):
            return csv_file
        else:
            cprint('[NOT CSV FILE]', 'blue', end=' ')
            print('The file', end=' ')
            cprint(f'{csv_file}', end=' ')
            print("doesn't exist or hasn't csv extension")
            exit(0)
    else:
        return default_file


def signal_adjustment(fs, signal):
    '''
    Analyse number of channels, compute mean of signal for
    2-channels signals and other useful information
    
    Args:
        fs (float): sampling frequency of signal
        signal (np.array): signal to be analysed

    Raises:

    Returns:
        ts (float): sampling step in time
        time_ms (np.array): sequence of instances of
                            each sample of the signal
        signal (np.array): signal analysed looking
                            to the number of channels
    '''
    #Duration of audio by looking to its number of samples
    N = signal.shape[0]
    secs = N / float(fs)
    #Computation of time instances for the audio
    ts = 1.0/fs
    time_ms = np.arange(0, N)
    #If num of channels = 2, do mean of the signal
    l_audio = len(signal.shape)

    if l_audio == 2:
        signal = np.mean(signal, axis=1)

    return ts, time_ms, signal


def num_samples(fs, seconds):
    '''
    Extract the feature from the signal
    
    Args:
        seconds (float): seconds to be converted in the
                            number of instances, looking to
                            base sampling step in time (ts)

    Raises:

    Returns:
        features: number of time instances related to ts
    
    '''
    return int(seconds*fs)

def results_to_string(results):
    '''
    Color list of results (already ordered by decreasing value
                           of probability of prediction) 
    '''
    res_string = ''

    res_string += colored(f'{results[0]}', 'blue')

    for item in results[1:]:
        res_string += (', ' + colored(f'{item}', 'green'))

    return res_string


SHORT_LINE ='______________________________________________'

def plot_detailed():
    '''
    Create images for each audio file in '../dat/MSI/'+sys.argv[1]
    '''
    subfolders = os.listdir('D:/THESIS/dat/MSI/'+sys.argv[1]+'/')

    for fold in subfolders:
        os.system('python3 .\DatasetAcquisition.py -p -d D:/THESIS/dat/MSI/'+sys.argv[1]+'/'+fold+' -o D:/THESIS/dat/MSI/graphics/detailed/'+fold)
#    for fold in range(5, 10):
#        os.system('python3 .\DatasetAcquisition.py -p -d ../dat/MSI/'+sys.argv[1]+'/'+str(fold)+' -o ../dat/MSI/graphics/detailed/'+str(fold))


def remove_wrong_files_recursive():
    '''
    Remove audio files in specific subfolders of a given folder, 
    that are not in graphics in subfolders of a given folder
    '''

    PATH_IMG = 'D:/THESIS/dat/MSI/graphics/detailed/'
    PATH_WAV = 'D:/THESIS/dat/MSI/audio/'
    subfolders = os.listdir(PATH_IMG)

    for fold in subfolders:
        img_files = os.listdir(PATH_IMG+fold)
        wav_files = os.listdir(PATH_WAV+fold)

        for f in wav_files:
            if not f[:-4]+'.png' in img_files:
                os.remove(PATH_WAV+fold+'/'+f)


def remove_wrong_files_folder():
    '''
    Remove audio files in a given folder,
    that are not in graphics in a given folder
    '''

    img_files = os.listdir('../dat/MSI/graphics/detailed/2/')
    wav_files = os.listdir('../dat/MSI/TEST/2/')

    for f in wav_files:
        if not f[:-4]+'.png' in img_files:
            os.remove('../dat/MSI/TEST/2/'+f)


def rename_files_recursive():
    '''
    Rename files (counter for each subfolder)
    '''
    PATH_WAV = 'D:/THESIS/dat/MSI/audio3/'
    subfolders = os.listdir(PATH_WAV)

    for fold in subfolders:
        wav_files = os.listdir(PATH_WAV+fold)

        count=0
        for f in wav_files:
            os.rename(PATH_WAV+fold+'/'+f, PATH_WAV+fold+'/'+'{:04d}.wav'.format(count))
            count+=1


def same_folder():
    '''
    See if 2 folders have the same content
    '''
    PATH_1 = 'D:/THESIS/dat/MSI/audio/'
    PATH_2 = 'D:/THESIS/dat/MSI/TEST/'
    subfolders_1 = os.listdir(PATH_1)
    subfolders_2 = os.listdir(PATH_2)
    subfolders_1.sort()
    subfolders_2.sort()
    check = True
    
    if subfolders_1==subfolders_2:
        for fold in subfolders_1:
            files_1 = os.listdir(PATH_1+fold)
            files_2 = os.listdir(PATH_2+fold)
            files_1.sort()
            files_2.sort()

            if files_1!=files_2:
                check=False
                break
    else:
        check=False

    return check


def state_dataset():
    '''
    Print number of elements in dataset
    '''
    dataset_size = 0
    list_low = []
    list_high=[]

    PATH = 'D:/THESIS/dat/MSI/audio/'
    folders = os.listdir(PATH)

    for fold in folders:
        length = len(os.listdir(PATH+fold))
        dataset_size+=length

        if length >= 200:
            list_high.append((fold, length))
        else:
            list_low.append((fold, length))

    print_list('LIST OF KEYS with < 200 clicks', list_low)
    print_list('LIST OF KEYS with >= 200 clicks', list_high)

    print('{:^46s}'.format('Summary'))
    print(SHORT_LINE)
    print(f'Number of keys with <200 clicks: {len(list_low)}')
    print(f'Number of keys with >=200 clicks: {len(list_high)}')
    print(f'Number of keys completed: {len(list_high)+len(list_low)}', end='\n\n')
    print(f'Dataset size: {dataset_size}', end='\n')
    print(SHORT_LINE, end='\n\n')


def print_list(title, wav_list):
    print('{:^46s}'.format(title))
    print(SHORT_LINE)
    for (fold,length) in wav_list:
        print('          {:>12s}: {:>3d}'.format(fold, 200-length))
    print(SHORT_LINE, end='\n\n\n')


def merge_subfolders():
    '''
    Merge content of each couple of subfolders
    with names: 'folder' and 'folder_2'
    '''

    PATH = 'D:/THESIS/dat/MSI/audio/'
    subfolders = [x for x in os.listdir(PATH) if x.endswith('_2')]
    print(subfolders)

    for fold in subfolders:
        count=0    
        files_1 = os.listdir(PATH+fold[:-2])
        
        for f in files_1:
            os.rename(PATH+fold[:-2]+'/'+f, PATH+fold[:-2]+'/'+'{:03d}.wav'.format(count))
            count+=1

        files_2 = os.listdir(PATH+fold)
        
        for f in files_2:
            os.rename(PATH+fold+'/'+f, PATH+fold[:-2]+'/'+'{:03d}.wav'.format(count))
            count+=1

        files_2 = os.listdir(PATH+fold)
        if not files_2:
            os.rmdir(PATH+fold)


def add_noise():
    """
    
    """
    OLD_PATH = 'D:/THESIS/dat/MSI/NEW/'
    NEW_PATH = 'D:/THESIS/dat/MSI/NOISE/'

    folder_old = OLD_PATH
    folder_new = NEW_PATH
    bar = progressbar.ProgressBar(maxval=20,
                                  widgets= [progressbar.Bar('=', '[', ']'),
                            				' ',
			                                progressbar.Percentage()])

    subfolders = os.listdir(folder_old)
    subfolders.sort()
    ranges = {0:(0,25), 1:(25, 50), 2:(50,75), 3:(75, 100)}
    range_fold = ranges[int(sys.argv[1])]
    subfolders_process = subfolders[range_fold[0]:range_fold[1]]
    cprint(f'{subfolders_process}', 'green')

    for subfolder in subfolders_process:
        files = os.listdir(folder_old+subfolder+'/')
        length = len(files)
        count = length
        
        cprint(f'\n{subfolder}', 'red')
        if not os.path.exists(folder_new+subfolder) or os.path.isdir(folder_new+subfolder):
            os.mkdir(folder_new+subfolder)

        bar.start()

        for f in files:
            fs, signal = wavfile.read(folder_old+subfolder+'/'+f)
            ts, time_ms, signal = signal_adjustment(fs, signal)

            noise = np.random.randn(len(signal))
            noise_signal = signal + 150.0 * noise
            # Cast back to same data type
            noise_signal = noise_signal.astype(type(signal[0]))

            wavfile.write(folder_new+subfolder+'/'+'{:04d}.wav'.format(count), fs, noise_signal)
            count += 1

            bar.update(((count-length)/length)*20)

        bar.finish()
        print('')


def time_shift():
    """
    
    """
    OLD_PATH = 'D:/THESIS/dat/MSI/'
    NEW_PATH = 'D:/THESIS/dat/MSI/NEW/'
    SECONDS = [0.5*float(x) for x in range(1, 5)]
    cprint('SECONDS:', 'blue', end='  ')
    print(SECONDS)

    folder_old = OLD_PATH+sys.argv[1]+'/'
    folder_new = NEW_PATH
    bar = progressbar.ProgressBar(maxval=20,
                                  widgets= [progressbar.Bar('=', '[', ']'),
                            				' ',
			                                progressbar.Percentage()])

    for subfolder in os.listdir(folder_old):
        files = os.listdir(folder_old+subfolder+'/')
        length = len(files)
        count = length
        
        cprint(f'\n{subfolder}', 'red')
        bar.start()

        for f in files:
            fs, signal = wavfile.read(folder_old+subfolder+'/'+f)
            ts, time_ms, signal = signal_adjustment(fs, signal)

            for sec in SECONDS:
                shift_num_samples = num_samples(fs, sec)
                v = np.zeros(shift_num_samples)
                signal1 = np.concatenate((v, signal))

                wavfile.write(folder_new+subfolder+'/'+'{:04d}.wav'.format(count), fs, signal1)
                count += 1

            bar.update(((count-length)/length)*20)

        bar.finish()


def select_option_feature():
    check = True
    while check:
        try:
            cprint(f'Select which type of features you want to use:\n{LINE}', 'blue')
                
            for i in range(0, len(OPTIONS)):
                cprint(f'{i})', 'yellow', end=' ')
                print(f'{OPTIONS[i]}')

            cprint(f'3)', 'yellow', end=' ')
            print(f'All the features')
            cprint(f'{LINE}', 'blue')

            option = int(input())
            if option >= 0 and option <= len(OPTIONS):
                check = False

        except ValueError:
            cprint('[VALUE ERROR]', 'color', end=' ')
            print('Insert a value of them specified in menu')

    return option


def fusion_csv():
    PATH_1000 = '../dat/1000/'
    PATH_2000 = '../dat/2000/'
    PATH_noise = '../dat/noise/'

    option = select_option_feature()
    csv_name1 = PATH_1000+OPTIONS[option]+'/dataset.csv'
    csv_name2 = PATH_noise+OPTIONS[option]+'/dataset.csv'
    csv_name_out = PATH_2000+OPTIONS[option]+'/dataset.csv'
    csv1 = loadtxt(csv_name1, delimiter=',')
    csv2 = loadtxt(csv_name2, delimiter=',')

    with open(csv_name_out, 'w', newline='') as out_f:
        writer_out = writer(out_f)
        
        # open file in read mode
        with open(csv_name1, 'r') as in1:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(in1)
                # Iterate over each row in the csv using reader object
            for row in csv_reader:
                writer_out.writerow(row)

        # open file in read mode
        with open(csv_name2, 'r') as in2:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(in2)
            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                writer_out.writerow(row)


def remove_useless_rows():
    PATHS = {'../dat/1000_time_less/','../dat/2000_less/','../dat/1000_noise_less/'}

    for path in PATHS:
        for option in range(0,2):
            csv_name_in = path+OPTIONS[option]+'/dataset_updated.csv'
            csv_name_out = path+OPTIONS[option]+'/dataset_final.csv'

            #Create dictionary of labels from csv file
            labels = {}
            with open(path+OPTIONS[option]+'/label_dict.csv') as fp:
                reader_label = reader(fp)
                labels = {float(row[1]):row[0] for row in reader_label}

            with open(csv_name_out, 'w', newline='') as out_f:
                writer_out = writer(out_f)

                # open file in read mode
                with open(csv_name_in, 'r') as in2:
                    # pass the file object to reader() to get the reader object
                    csv_reader = reader(in2)
                    # Iterate over each row in the csv using reader object
                    for row in csv_reader:
                        if float(row[-1]) in labels.keys():
                            row[-1] = float(list(labels.keys()).index(float(row[-1])))
                            writer_out.writerow(row)


def extract_features_from_imgs():
    PATH = f'D:/THESIS/dat/MSI/graphics/{sys.argv[1]}/'
    CSV_DICT_LABELS = f'D:/github/Invisible-CAPPCHA/Code/PC/dat/{sys.argv[2]}/spectrum/label_dict.csv'
    CSV_DATASET = f'D:/github/Invisible-CAPPCHA/Code/PC/dat/{sys.argv[2]}/spectrum/dataset.csv'

    model = VGG16(weights='imagenet', include_top=False)

    labels = {}
    with open(CSV_DICT_LABELS) as fp:
        csv_reader = reader(fp)
        labels = {row[0]:int(row[1]) for row in csv_reader} 
   
    with open(CSV_DATASET, 'w', newline='') as fp:
        csv_writer = writer(fp)
        
        for fold in os.listdir(PATH):
            cprint(f'{fold}', 'red')
            for img_name in os.listdir(PATH+fold+'/'):
                features = extract(model, PATH+fold+'/'+img_name)
                row = np.append(features, labels[fold])
                csv_writer.writerow(row)


def extract(model, path):
    PATH_SQUARE = 'D:/THESIS/dat/MSI/graphics/spectrum_square_less/'        
    img = image.load_img(path, color_mode='rgb', target_size=(224, 224))

    #if not os.path.exists(PATH_SQUARE+fold):
    #   os.mkdir(PATH_SQUARE+fold)

    #img.save(PATH_SQUARE+fold+'/'+img_name)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return GlobalAveragePooling2D()(features)


def create_label_dict():
    LABELS = {  '0':0, '1':1, '2':2, '3':3, '4':4,
                '5':5, '6':6, '7':7, '8':8, '9':9, 
                'a':10, 'b':11, 'c':12, 'd':13, 
                'e':14, 'f':15, 'g':16, 'h':17,
                'i':18, 'j':19, 'k':20, 'l':21,
                'm':22, 'n':23, 'o':24, 'p':25,
                'q':26, 'r':27, 's':28, 't':29,
                'u':30, 'v':31, 'w':32, 'x':33,
                'y':34, 'z':35, 'à':36, 'è':37,
                'ì':38, 'ò':39, 'ù':40 }

    CSV_DICT_LABELS = 'D:/github/Invisible-CAPPCHA/Code/PC/dat/label_dict.csv'

    with open(CSV_DICT_LABELS, 'w', newline='') as fp:
        csv_writer = writer(fp)
        
        for k,v in LABELS.items():
            csv_writer.writerow([k, v])


if __name__=='__main__':
    colorama.init()
    extract_features_from_imgs()
    #create_label_dict()
    pass