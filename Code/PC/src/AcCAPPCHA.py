import numpy as np
import pyaudio
import wave
import sys
from scipy.io import wavfile
from time import sleep
import threading
from termcolor import cprint
import os
import utility
import tempfile
import ExtractFeatures as ef
from matplotlib import pyplot as plt
from scipy.stats import mode
from collections import Counter

class AcCAPPCHA:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    def __init__(self):
        self.mutex = threading.Lock()
        self.COMPLETED_INSERT = False
        self.KILLED = False


    def audio(self):
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
        
        #RECORDING
        frames = []

        while not self.COMPLETED_INSERT:
            if self.KILLED:
                #Terminate the audio recorder (detected CTRL+C)
                exit(0)

            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

#        wf = wave.open(tempfile.gettempdir()+'/tmp.wav', 'wb')
        wf = wave.open('../dat/tmp.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        #Reading audio file
        fs, signal = wavfile.read('../dat/tmp.wav')
        #Analysis of audio signal
        #analysis = ef.ExtractFeatures(fs, signal)
        self.plot_wave(fs, signal)

        self.mutex.acquire()
        try:
            self.COMPLETED_INSERT = False
        
        finally:
            self.mutex.release()


    def password(self):
        sleep(1)
        cprint('Insert the password', 'red')
        
        try:
            password = input('')
            
            if password == 'EXIT':
                self.KILLED = True
                exit(0)
        
        except (EOFError, KeyboardInterrupt):
            self.KILLED = True
            exit(0)

        self.mutex.acquire()
        try:
            self.COMPLETED_INSERT = True
        
        finally:
            self.mutex.release()


    def plot_wave(self, fs, signal):
        '''
        Plot the audio signal with name filename with
        highlighted peaks and FFT of push and hit peaks
        
        Args:
            filename (str): name of wav file
            analysis (ExtractFeatures): ExtractFeatures related to the
                                        wav file with name filename
        '''

        #Time sample step
        ts, time_ms, signal = utility.signal_adjustment(fs, signal)

		#Initialize the figure
        fig = plt.figure('AUDIO')

        peak_indices = self.analyse(signal)
        #Plot of press peak and hit peak with the signal
        plt.plot(time_ms*ts, signal[time_ms], color='blue')
        plt.plot(peak_indices*ts, signal[peak_indices], color='red')
        plt.tick_params(axis='both', which='major', labelsize=6)
        plt.show()


    def run(self):
        '''
        Start keylogger and audio recorder
        '''
        try:
            while not self.KILLED:
                audio_logger = threading.Thread(target=self.audio)
                password_logger = threading.Thread(target=self.password)
                audio_logger.start()
                password_logger.start()
                audio_logger.join()
                password_logger.join()

        except KeyboardInterrupt:
            #Terminate the keylogger (detected CTRL+C)
            self.KILLED = True
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')
            exit(0)


    def analyse(self, signal):
        count = Counter(signal)
        #mode_value = mode(signal)
        mode_value =count.most_common(1)[0][0]
        input('')
        threshold = 50.0 * abs(float(mode_value))
        return np.argwhere(signal > threshold)


if __name__=='__main__':
    captcha = AcCAPPCHA()
    captcha.run()