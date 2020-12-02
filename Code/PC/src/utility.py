from termcolor import cprint
import os
import sys

OPTIONS = ['touch', 'touch_hit', 'spectrum']
LINE = '_____________________________________________________'

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


SHORT_LINE ='______________________________________________'

def plot_detailed():
    '''
    Create images for each audio file in '../dat/MSI/'+sys.argv[1]
    '''

    subfolders = os.listdir('../dat/MSI/'+sys.argv[1]+'/')

    for fold in subfolders:
        os.system('python3 .\DatasetAcquisition.py -p -d ../dat/MSI/'+sys.argv[1]+'/'+fold+' -o ../dat/MSI/graphics/detailed/'+fold)
#    for fold in range(5, 10):
#        os.system('python3 .\DatasetAcquisition.py -p -d ../dat/MSI/'+sys.argv[1]+'/'+str(fold)+' -o ../dat/MSI/graphics/detailed/'+str(fold))


def remove_wrong_files_recursive():
    '''
    Remove audio files in specific subfolders of a given folder, 
    that are not in graphics in subfolders of a given folder
    '''

    PATH_IMG = '../dat/MSI/graphics/detailed/'
    PATH_WAV = '../dat/MSI/audio/'
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

    PATH_WAV = '../dat/MSI/audio/'
    subfolders = os.listdir(PATH_WAV)

    for fold in subfolders:
        wav_files = os.listdir(PATH_WAV+fold)

        count=0
        for f in wav_files:
            os.rename(PATH_WAV+fold+'/'+f, PATH_WAV+fold+'/'+'{:03d}.wav'.format(count))
            count+=1


def same_folder():
    '''
    See if 2 folders have the same content
    '''

    PATH_1 = '../dat/MSI/TEST/'
    PATH_2 = '../dat/MSI/audio/'
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

    PATH = '../dat/MSI/audio/'
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
        print('          {:>12s}: {:>3d}'.format(fold, length))
    print(SHORT_LINE, end='\n\n\n')


def merge_subfolders():
    '''
    Merge content of each couple of subfolders
    with names: 'folder' and 'folder_2'
    '''

    PATH = '../dat/MSI/audio/'
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


if __name__=='__main__':
    #rename_files_recursive()
    state_dataset()
    #plot_detailed()
    #merge_subfolders()
    pass