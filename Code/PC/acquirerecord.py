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

class AcquireAudio:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    TIMES_KEY_PRESSED = 10
    DATA_FOLDER = 'data/'
    WAVE_OUTPUT_FILENAME = ''
    CHECK_TYPING = True
    FIRST_KEY = True


    def __init__(self):
        self.mutex = threading.Lock()
        self.count = 0

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

        wf = wave.open(self.DATA_FOLDER+self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.mutex.acquire()
        try:
            self.WAVE_OUTPUT_FILENAME = ''
            self.CHECK_TYPING = True
            self.count = 0
        finally:
            self.mutex.release()


    def press_key(self, key):

        if self.count < self.TIMES_KEY_PRESSED:
            self.count = self.count + 1
            print(self.count)

            if self.count == self.TIMES_KEY_PRESSED:
                self.mutex.acquire()
                try:
                    self.CHECK_TYPING = False
                    exit(0)
                finally:
                    self.mutex.release()
        
        #Obtain string of key inserted
        try:
            key_string = str(key.char)
        except AttributeError:
            #Special key pressed
            if key == key.space: 
                #Otherwise printed 'key space'
                key_string = 'SPACE'
            else:
                key_string = str(key)
        
        sleep(1)
        if self.FIRST_KEY:
            self.mutex.acquire()
            try:
                self.WAVE_OUTPUT_FILENAME = key_string+'.wav'
            finally:
                self.mutex.release()


    def record(self):
        while True:
            cprint('\nType first letter 30 times', 'green', attrs=['bold'])

            audiologger = threading.Thread(target=self.audio_logging)
            audiologger.start()        
            with keyboard.Listener(on_press=self.press_key) as listener:
                #Manage keyboard input
                listener.join()

            audiologger.join()