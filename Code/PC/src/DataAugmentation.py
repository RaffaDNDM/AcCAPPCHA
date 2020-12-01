import os
import sys

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
    list_0 = []
    list_100=[]
    list_150=[]

    PATH = '../dat/MSI/audio/'
    folders = os.listdir(PATH)

    for fold in folders:
        length = len(os.listdir(PATH+fold))
        dataset_size+=length

        if length >= 100:
                if length < 150:
                        list_100.append((fold, length))
                else:
                        list_150.append((fold, length))
        else:
            list_0.append((fold, length))

    print_list('LIST OF KEYS with < 100 clicks', list_0)
    print_list('LIST OF KEYS with >= 100 and <150 clicks', list_100)
    print_list('LIST OF KEYS with >= 150 clicks', list_150)

    print('{:^46s}'.format('Summary'))
    print(SHORT_LINE)
    print(f'Number of keys with <100 clicks: {len(list_0)}')
    print(f'Number of keys with >=100 and <150 clicks: {len(list_100)}')
    print(f'Number of keys with >=150 clicks: {len(list_150)}')
    print(f'Number of keys completed: {len(list_0)+len(list_100)+len(list_150)}', end='\n\n')
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
    #state_dataset()
    plot_detailed()
    pass