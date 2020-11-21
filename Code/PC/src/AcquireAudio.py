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
    DATA_FOLDER = '../dat/audio/'
    WAVE_OUTPUT_FILENAME = None
    CHECK_TYPING = True
    FIRST_KEY = True
    KILLED = False

    '''
    AcquireAudio object acquires key audio from user by
    using also a keylogger to understand pressed key

    Args:
        audio_dir (str): Path of the directory where the program will 
                         create a subfolder for each new pressed key

        time_key (int): Number of pressed keys that acquisition
                        will wait before saving a file 
                        (e.g. times_key=10  key='a', recorded audio
                         will contain 10 'a')
                        [****WARNING****] 
                        the keylogger will check only the value of
                        the first pressed key for each audio and 
                        trusts user for the (times_key - 1) insertions

    Attributes:
        DATA_FOLDER (float): Path of the directory where the program will 
                             create a subfolder for each new pressed key
                             (by default '../dat/audio')

        LETTERS (dict): Dictionary with couples, each one composed by:
                        key (str): name of the subfolder in DATA_FOLDER
                        value (int): num of audio files in the subfolder
                                     with name key in subfolder

        TIMES_KEY_PRESSED (int): Number of pressed keys that acquisition
                                 will wait before saving a file 
                                 (e.g. times_key=10  key='a', recorded audio
                                 will contain 10 'a')

        [****ACQUISITION PARAMETERS****]
        CHUNK (int): (by default 1024)
        FORMAT (pyaudio format): (by default pyaudio.paInt16)
        CHANNELS (int): (by default 2)
        RATE (int): Sampling rate (by default 44100)

        [****COMMUNICATION BETWEEN RECORDER AND KEYLOGGER****]
        WAVE_OUTPUT_FILENAME (str): Name of audio file that will be created
        
        CHECK_TYPING (bool): True if user is typing, False if user has pressed
                             a key TIME_KEY_PRESSED and audio recorder can store
                             the file because the acquisition is completed 
        
        FIRST_KEY (bool): True if the first key of the sequence of TIMES_KEY_PRESSED
                          hasn't yet pressed
        
        KILLED (bool): True when the keylogger detects CTRL+C and so the audio
                       recorder ends the acqisition
        
        mutex (threading.Lock): Mutual exclusion LOCK
        
        count (int): num of elements already inserted of the sequence of 
                     TIME_KEY_PRESSED keys by user
    '''
    def __init__(self, audio_dir, times_key):
        if not(os.path.exists(audio_dir) and os.path.isdir(audio_dir)):
            os.mkdir(self.DATA_FOLDER)

        if audio_dir:
            self.DATA_FOLDER = utility.uniform_dir_path(audio_dir)
    
        self.already_acquired()
        self.LETTERS['APOSTROPHE']=9

        input('Print something to start the acquisition of audio')
        sleep(2)
        self.mutex = threading.Lock()
        self.count = 0
        self.TIMES_KEY_PRESSED = times_key


    def already_acquired(self):    
        '''
        Instantiate LETTERS dictionary looking at the subfolders 
        of DATA_FOLDER path and the number of wav files in them
        '''
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
        '''
        Record the waves from microphone and store the audio
        file after TIMES_KEY_PRESSED times a key is pressed
        '''
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
                #Terminate the audio recorder (detected CTRL+C)
                exit(0)

            data = stream.read(self.CHUNK)
            frames.append(data)

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
        '''
        Record the key pressed by user
        
        Args:
            key (key): pynput key
        '''
        key_string = utility.key_definition(key)

        if self.count < self.TIMES_KEY_PRESSED:
            self.count = self.count + 1

            sleep(1)
            self.mutex.acquire()
            try:
                if key_string in self.LETTERS.keys():
                    self.WAVE_OUTPUT_FILENAME = key_string+'/'+str(self.LETTERS[key_string])+'.wav'
                    self.LETTERS[key_string] = self.LETTERS[key_string] + 1
                    print('\r'+str(self.LETTERS[key_string]), end='')
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
        '''
        Start keylogger and audio recorder
        '''
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
            #Terminate the keylogger (detected CTRL+C)
            self.KILLED = True
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')
