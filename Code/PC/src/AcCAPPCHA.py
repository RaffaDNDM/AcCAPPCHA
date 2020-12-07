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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, use
from scipy.stats import mode
from collections import Counter

class AcCAPPCHA:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SECONDS_NOISE = 2.0
    OUTPUT_IMG = '../dat/audio_from_user.png'
    COLORS = ['g', 'r', 'c', 'm', 'y']

    def __init__(self):
        self.mutex = threading.Lock()
        self.COMPLETED_INSERT = False
        self.KILLED = False
        self.CHAR_RANGE = utility.num_samples(self.RATE, 0.16)
        self.ENTER_RANGE = utility.num_samples(self.RATE, 0.2)

    def noise_evaluation(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK)
        
        #RECORDING
        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.SECONDS_NOISE)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

#        wf = wave.open(tempfile.gettempdir()+'/tmp.wav', 'wb')
        wf = wave.open('../dat/noise.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        #Reading audio file
        fs, signal = wavfile.read('../dat/noise.wav')
        #Analysis of audio signal
        #analysis = ef.ExtractFeatures(fs, signal)
        self.noise = np.max(np.abs(signal))


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
        print(len(signal))

        self.mutex.acquire()
        try:
            self.COMPLETED_INSERT = False    
            self.signal = signal[:-self.ENTER_RANGE]
            self.plot_wave(fs, self.signal)
        finally:
            self.mutex.release()


    def password(self):
        sleep(1)
        self.noise_evaluation()

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
        print(peak_indices.shape)
        #Plot of press peak and hit peak with the signal
        plt.plot(time_ms*ts, signal[time_ms], color='blue')
        
        char_times = [[peak_indices[0][0],], ]
        print(peak_indices[0])
        start = 0
        for i in peak_indices[1:]:
            if (i[0]-char_times[start][0])>self.CHAR_RANGE:
                char_times.append([i[0],])
                start += 1
            else:
                char_times[start].append(i[0])
                plt.plot(i[0]*ts, signal[i[0]], color=self.COLORS[start%len(self.COLORS)], marker='x')    
        
        plt.tick_params(axis='both', which='major', labelsize=6)
        fig.savefig(self.OUTPUT_IMG)

        count = 0
        for list_time in char_times:
            analysis = ef.ExtractFeatures(self.RATE, signal[list_time[0]:list_time[-1]])
            analysis.extract(original_signal=signal, index=list_time[0])
            #Evaluation of press peak and hit peaks
            features = analysis.extract()
            touch_feature = features['touch']
            hit_feature = features['hit']
            
            fig = plt.figure('CHARACTER')
            gs = fig.add_gridspec(2, 2)
            s_top = fig.add_subplot(gs[0, :])
            s1 = fig.add_subplot(gs[1,0])
            s2 = fig.add_subplot(gs[1,1])
            fig.tight_layout(pad=3.0)

            #Plot of press peak and hit peak with the signal
            s_top.plot(time_ms*ts, signal[time_ms], color='blue')
            s_top.plot(touch_feature.peak*ts, signal[touch_feature.peak], color='red')
            s_top.plot(hit_feature.peak*ts, signal[hit_feature.peak], color='green')
            s_top.set_title('Amplitude')
            s_top.set_xlabel('Time(ms)')
            s_top.tick_params(axis='both', which='major', labelsize=6)

            #Plot FFT double-sided transform of PRESS peak
            s1.plot(touch_feature.freqs, touch_feature.fft_signal, color='red')
            s1.set_xlabel('Frequency (Hz)')
            s1.set_ylabel('FFT of PRESS PEAK')
            s1.tick_params(axis='both', which='major', labelsize=6)
            s1.set_xscale('log')
            s1.set_yscale('log')

            #Plot FFT single-sided transform of HIT peak
            s2.plot(hit_feature.freqs, hit_feature.fft_signal, color='green')
            s2.set_xlabel('Frequency(Hz)')
            s2.set_ylabel('FFT of HIT PEAK')
            s2.tick_params(axis='both', which='major', labelsize=6)
            s2.set_xscale('log')
            s2.set_yscale('log')
            fig.savefig(self.OUTPUT_IMG[:-4]+f'{count}.png')
            count+=1
        

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
        input('')
        #threshold = 50.0 * np.mean(np.array(mode_values))
        threshold = self.noise
        return np.argwhere(signal > threshold)


if __name__=='__main__':
    captcha = AcCAPPCHA()
    captcha.run()