import os
import sys

#Create images for each audio file in '../dat/MSI/'+sys.argv[1]
#subfolders = os.listdir('../dat/MSI/'+sys.argv[1]+'/')

#for fold in subfolders:
#    os.system('python3 .\DatasetAcquisition.py -p -d ../dat/MSI/'+sys.argv[1]+'/'+fold+' -o ../dat/MSI/graphics/detailed/'+fold)

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