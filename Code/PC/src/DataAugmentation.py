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