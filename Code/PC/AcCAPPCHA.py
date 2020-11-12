import matplotlib.pyplot as plt
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

        start_time = datetime.datetime.now()

        while self.CHECK_TYPING:
            data = stream.read(self.CHUNK)
            frames.append(data)

        '''
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        '''

        amplitude = b''.join(frames)

        stream.stop_stream()
        stream.close()
        p.terminate()

        elapsed_time = datetime.datetime.now() - start_time

        cprint('\n*** End recording ***', 'green', attrs=['bold'])

        time = np.linspace(0, int(elapsed_time.total_seconds()*1000) , num=len(frames))

        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        s_top = fig.add_subplot(gs[0, :])
        s1 = fig.add_subplot(gs[1,0])
        s2 = fig.add_subplot(gs[1,1])
        fig.tight_layout(pad=3.0)
        amplitude = np.fromstring(amplitude, np.int16)
        s_top.plot(amplitude)
        fft_amplitude = np.fft.fft(amplitude)
        s1.plot(np.abs(fft_amplitude))
        s2.specgram(np.abs(fft_amplitude))
        plt.show()

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
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
        
        if self.count <5:
            self.count = self.count + 1
        elif self.count == 5:
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
        
        if self.FIRST_KEY:
            self.mutex.acquire()
            try:
                self.WAVE_OUTPUT_FILENAME = key_string+'.wav'
            finally:
                self.mutex.release()



    def key_logging(self):
        cprint('\nType first letter 30 times', 'green', attrs=['bold'])
        
        with keyboard.Listener(on_press=self.press_key) as listener:
                #Manage keyboard input
                listener.join()


    def record(self):
        keylogger = threading.Thread(target=self.key_logging)
        audiologger = threading.Thread(target=self.audio_logging)
        keylogger.start()
        audiologger.start()
        keylogger.join()
        audiologger.join()

def main():
    try:
        acquisition = AcquireAudio()
        acquisition.record()
        cprint('\nExit from program\n', 'red', attrs=['bold'])
    except KeyboardInterrupt:
        cprint('\nClosing the program', 'red', attrs=['bold'])

if __name__=='__main__':
    main()