import numpy as np
import pyaudio
import wave
import sys
import scipy
import scipy.fftpack as fftpk
from time import sleep
import threading
from pynput import keyboard
import datetime
from termcolor import cprint
import os
import utility

class AcquireAudio:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    DATA_FOLDER = 'dat/audio/'
    WAVE_OUTPUT_FILENAME = None
    CHECK_TYPING = True
    FIRST_KEY = True
    KILLED = False

    def __init__(self, audio_dir, times_key):
        if not(os.path.exists(audio_dir) and os.path.isdir(audio_dir)):
            os.mkdir(self.DATA_FOLDER)

        if audio_dir:
            self.DATA_FOLDER = utility.uniform_dir_path(audio_dir)
    
        self.already_acquired()

        input('Print something to start the acquisition of audio')
        sleep(2)
        self.mutex = threading.Lock()
        self.count = 0
        self.TIMES_KEY_PRESSED = times_key


    def already_acquired(self):    
        #Initialize LETTERS with number of files already in the subfolder ('a', ...)
        subfolders = os.listdir(self.DATA_FOLDER)
        subfolders.sort()

        count=0
        self.LETTERS = {}
        special_chars = {}
        cprint('\nNum of already acquired audio samples for letters', 'blue')
        cprint('_________________________________________________', 'blue')

        for subfolder in subfolders:
            if os.path.isdir(self.DATA_FOLDER+subfolder):
                num_already_acquired = len([x for x in os.listdir(self.DATA_FOLDER+subfolder) if x.endswith('.wav')])
                self.LETTERS[subfolder] = num_already_acquired
                if len(subfolder)==1:
                    cprint(f'{subfolder}', 'green', end=' ', attrs=['bold',])
                    cprint('---->', 'yellow', end=' ')
                    delimiter = '   '

                    if count==3:
                        delimiter = '\n'
                        count = 0
                    else:
                        count = count + 1
                    
                    print('{:2d}'.format(num_already_acquired), end=delimiter)

                else:
                    special_chars[subfolder] = num_already_acquired

        if count==0:
            print('', end='\r')        
        else:
            print('', end='\n')
            count=0

        for subfolder in special_chars:
            cprint(f'{subfolder}', 'green', end=' ', attrs=['bold',])
            cprint('---->', 'yellow', end=' ')
            print('{:2d}'.format(self.LETTERS[subfolder]))

        cprint('_________________________________________________', 'blue')


    def audio_logging(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK)
        
        cprint('\n*** recording ***', 'green', attrs=['bold'])

        frames = []

        while self.CHECK_TYPING:
            if self.KILLED:
                exit(0)

            data = stream.read(self.CHUNK)
            frames.append(data)

        '''
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        '''

        stream.stop_stream()
        stream.close()
        p.terminate()

        cprint('\n*** End recording ***', 'green', attrs=['bold'])

        while not self.WAVE_OUTPUT_FILENAME:
            sleep(1)

        wf = wave.open(self.DATA_FOLDER+self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.mutex.acquire()
        try:
            self.WAVE_OUTPUT_FILENAME = None
            self.CHECK_TYPING = True
            self.count = 0
            self.FIRST_KEY = True
        finally:
            self.mutex.release()


    def press_key(self, key):
        key_string = utility.key_definition(key)

        if self.count < self.TIMES_KEY_PRESSED:
            self.count = self.count + 1
            print('\r'+str(self.count), end='')

            sleep(1)
            self.mutex.acquire()
            try:
                if key_string in self.LETTERS.keys():
                    self.WAVE_OUTPUT_FILENAME = key_string+'/'+str(self.LETTERS[key_string])+'.wav'
                    self.LETTERS[key_string] = self.LETTERS[key_string] + 1
                else:
                    self.LETTERS[key_string] = 0
                    self.WAVE_OUTPUT_FILENAME = key_string+'/'+str(self.LETTERS[key_string])+'.wav'
                    os.mkdir(self.DATA_FOLDER+key_string)

            finally:
                self.mutex.release()

            if self.FIRST_KEY:
                self.mutex.acquire()
                try:
                    self.FIRST_KEY = False
                finally:
                    self.mutex.release()

            if self.count == self.TIMES_KEY_PRESSED:
                self.mutex.acquire()
                try:
                    self.CHECK_TYPING = False
                    exit(0)
                finally:
                    self.mutex.release()
        
            
    def record(self):
        try:
            while True:
                cprint(f'\nType first letter {self.TIMES_KEY_PRESSED} times', 'green', attrs=['bold'])

                audiologger = threading.Thread(target=self.audio_logging)
                audiologger.start()
                with keyboard.Listener(on_press=self.press_key) as listener:
                    #Manage keyboard input
                    listener.join()

                audiologger.join()

        except KeyboardInterrupt:
            self.KILLED = True
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')