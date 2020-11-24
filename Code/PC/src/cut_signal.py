import os
from scipy.io import wavfile as wave

# PATH must contain 2 subfolders:
#'letter': audio files folder
#'graph': folder with images, one for each wav file
#              contained in several subfolders, each one
#              with the name = number of first seconds 
#              that you want to mantain
#              [e.g. new_i/0.5/ contains images realtive
#               to wav files for which I want only the
#               first 0.5 seconds]
PATH = '../dat/MSI/TEST/'
subfolders = os.listdir(PATH+'graph/')
letter = 'i'

for subfold in subfolders:
    files = os.listdir(PATH+'graph/'+subfold)

    for f in files:
        fs, signal = wave.read(PATH+letter+'/'+f[:-4]+'.wav')
        end = int(float(subfold)*fs)
        wave.write(PATH+letter+'/'+f[:-4]+'.wav', fs, signal[:end])
