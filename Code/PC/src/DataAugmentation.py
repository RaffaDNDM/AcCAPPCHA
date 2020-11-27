import os
import sys
import utility

#Create images for each audio file in '../dat/MSI/'+sys.argv[1]
subfolders = os.listdir('../dat/MSI/'+sys.argv[1]+'/')

for fold in subfolders:
    os.system('python3 .\DatasetAcquisition.py -p -d ../dat/MSI/'+sys.argv[1]+'/'+fold+' -o ../dat/MSI/graphics/detailed/'+fold)

#Remove audio files in TEST that are not correct
#PATH_IMG = '../dat/MSI/graphics/detailed/'
#PATH_WAV = '../dat/MSI/TEST/'
#subfolders = os.listdir(PATH_IMG)

#for fold in subfolders:
#    img_files = os.listdir(PATH_IMG+fold)
#    wav_files = os.listdir(PATH_WAV+fold)

#    for f in wav_files:
#        if not f[:-4]+'.png' in img_files:
#            os.remove(PATH_WAV+fold+'/'+f)

#img_files = os.listdir('../dat/MSI/graphics/detailed/2/')
#wav_files = os.listdir('../dat/MSI/TEST/2/')

#for f in wav_files:
#    if not f[:-4]+'.png' in img_files:
#        os.remove('../dat/MSI/TEST/2/'+f)
            

#Rename files
#PATH_WAV = '../dat/MSI/TEST/'
#subfolders = os.listdir(PATH_WAV)

#for fold in subfolders:
#    wav_files = os.listdir(PATH_WAV+fold)

#    count=0
#    for f in wav_files:
#        os.rename(PATH_WAV+fold+'/'+f, PATH_WAV+fold+'/'+'{:03d}.wav'.format(count))
#        count+=1

#See if 2 folders have the same content
#PATH_1 = '../dat/MSI/TEST/'
#PATH_2 = '../dat/MSI/audio/'
#subfolders_1 = os.listdir(PATH_1)
#subfolders_2 = os.listdir(PATH_2)
#subfolders_1.sort()
#subfolders_2.sort()
# check = True
#if subfolders_1==subfolders_2:
#    for fold in subfolders_1:
#        files_1 = os.listdir(PATH_1+fold)
#        files_2 = os.listdir(PATH_2+fold)
#        files_1.sort()
#        files_2.sort()

#        if files_1!=files_2:
#            check=False
#            break
#else:
#    check=False

#print(check)


#Print number of elements in dataset
#count_completed=0
#list_100=[]
#list_150=[]
#
#PATH = '../dat/MSI/audio/'
#folders = os.listdir(PATH)
#
#for fold in folders:
#     length = len(os.listdir(PATH+fold))
#     if length >= 100:
#             count_completed+=1
#             if length < 150:
#                     list_100.append((fold, length))
#             else:
#                     list_150.append((fold, length))
#
#print('LIST OF KEYS with >= 100 and <150 clicks')
#print(utility.LINE)
#for (fold,length) in list_100:
#     print(f'{fold}: {length}')
#print(utility.LINE, end='\n\n')

#print('LIST OF KEYS with >= 150 clicks')
#print(utility.LINE)
#for (fold,length) in list_150:
#     print(f'{fold}: {length}')
#print(utility.LINE, end='\n\n')
#
#print(f'Number of keys with >=100 and <150 clicks: {len(list_100)}')
#print(f'Number of keys with >=150 clicks: {len(list_150)}')
#print(f'Number of keys completed: {count_completed}', end='\n\n')


#Merge content of each couple of subfolders
#with names: 'folder' and 'folder_2'
#PATH = '../dat/MSI/audio/'
#subfolders = [x for x in os.listdir(PATH) if x.endswith('_2')]
#print(subfolders)

#for fold in subfolders:
#    count=0
#    
#    files_1 = os.listdir(PATH+fold[:-2])
#    
#    for f in files_1:
#        os.rename(PATH+fold[:-2]+'/'+f, PATH+fold[:-2]+'/'+'{:03d}.wav'.format(count))
#        count+=1
#
#    files_2 = os.listdir(PATH+fold)
#    
#    for f in files_2:
#        os.rename(PATH+fold+'/'+f, PATH+fold[:-2]+'/'+'{:03d}.wav'.format(count))
#        count+=1