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
import NeuralNetwork as nn
from tensorflow.keras.applications.vgg16 import VGG16
import colorama


class AcCAPPCHA:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SECONDS_NOISE = 2.0
    DATASET_FOLDER = '../dat/1000_time_less/'
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
            self.KILLED = True
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
        self.ts, self.time_ms, self.signal = utility.signal_adjustment(fs, signal)

		#Initialize the figure
        fig = plt.figure('AUDIO')

        peak_indices = self.analyse(signal)
        #Plot of press peak and hit peak with the signal
        plt.plot(self.time_ms*self.ts, signal[self.time_ms], color='blue')
        
        self.char_times = [[peak_indices[0][0],], ]
        
        start = 0
        for i in peak_indices[1:]:
            if (i[0]-self.char_times[start][0])>self.CHAR_RANGE:
                self.char_times.append([i[0],])
                start += 1
            else:
                self.char_times[start].append(i[0])
                plt.plot(i[0]*self.ts, self.signal[i[0]], color=self.COLORS[start%len(self.COLORS)], marker='x')    
        
        plt.tick_params(axis='both', which='major', labelsize=6)
        fig.savefig(self.OUTPUT_IMG)

        option = utility.select_option_feature()
        net = nn.NeuralNetwork(option, self.DATASET_FOLDER)
                
        count = 0
        model = VGG16(weights='imagenet', include_top=False)
                
        for list_time in self.char_times:
            #Evaluation of press peak and hit peaks
            analysis = ef.ExtractFeatures(self.RATE, self.signal[list_time[0]:list_time[-1]])            
            fig = plt.figure('CHARACTER'+str(count))
            fig.tight_layout(pad=3.0)
            
            if utility.OPTIONS[option]=='touch':
                features = analysis.extract(original_signal=self.signal, index=list_time[0])
                touch_feature = features['touch']
                hit_feature = features['hit']
                touch_X = touch_feature.fft_signal.reshape((1, 66))
                cprint(touch_X.shape, 'red')
                print(f'{net.test(touch_X)}', end='')

                gs = fig.add_gridspec(2, 2)
                s_top = fig.add_subplot(gs[0, :])
                s1 = fig.add_subplot(gs[1,0])
                s2 = fig.add_subplot(gs[1,1])
            
                #Plot of press peak and hit peak with the signal
                s_top.plot(self.time_ms*self.ts, self.signal[self.time_ms], color='blue')
                s_top.plot(touch_feature.peak*self.ts, self.signal[touch_feature.peak], color='red')
                s_top.plot(hit_feature.peak*self.ts, self.signal[hit_feature.peak], color='green')
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
                #plt.show()

            elif utility.OPTIONS[option]=='touch_hit':
                features = analysis.extract(original_signal=self.signal, index=list_time[0])
                touch_feature = features['touch']
                hit_feature = features['hit']            
                touch_X = touch_feature.fft_signal
                hit_X = hit_feature.fft_signal
                touch_hit_X = np.concatenate((touch_X, hit_X)).reshape((1, 132))
                #cprint(touch_hit_X.shape, 'red')
                print(f'{count} ---> {net.test(touch_hit_X)}')

                gs = fig.add_gridspec(2, 2)
                s_top = fig.add_subplot(gs[0, :])
                s1 = fig.add_subplot(gs[1,0])
                s2 = fig.add_subplot(gs[1,1])
            
                #Plot of press peak and hit peak with the signal
                s_top.plot(self.time_ms*self.ts, self.signal[self.time_ms], color='blue')
                s_top.plot(touch_feature.peak*self.ts, self.signal[touch_feature.peak], color='red')
                s_top.plot(hit_feature.peak*self.ts, self.signal[hit_feature.peak], color='green')
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
                #plt.show()

            else:
                #Extraction of features
                fig, ax = plt.subplots(1)
                fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
                features = analysis.extract(original_signal=self.signal, index=list_time[0], spectrum=True)
        
                img_feature = np.concatenate((self.signal[features[0]], self.signal[features[1]]))
                spectrum, freqs, t, img_array = ax.specgram(img_feature, NFFT=len(features[0]), Fs=analysis.fs)
                
                if not os.path.exists(self.DATASET_FOLDER+f'spectrum_img'):
                    os.mkdir(self.DATASET_FOLDER+f'spectrum_img')

                fig.savefig(self.DATASET_FOLDER+f'spectrum_img/{count}.jpg', dpi=300)
                plt.close(fig)
			
                gs = fig.add_gridspec(2, 1)
                s_top = fig.add_subplot(gs[0, 0])
                s_bottom = fig.add_subplot(gs[1,0])

                #Plot of press peak and hit peak with the signal
                s_top.plot(self.time_ms*self.ts, self.signal[self.time_ms], color='blue')
                s_top.plot(features[0]*self.ts, self.signal[features[0]], color='red')
                s_top.plot(features[1]*self.ts, self.signal[features[1]], color='green')
                s_top.set_title('Amplitude')
                s_top.set_xlabel('Time(ms)')
                s_top.tick_params(axis='both', which='major', labelsize=6)

                s_bottom.specgram(img_feature, NFFT=len(features[0]), Fs=analysis.fs)

                features = utility.extract(model, self.DATASET_FOLDER+f'spectrum_img/{count}.jpg')
                fig.savefig(f'Char{count}.png')
                print(f'{net.test(features)}', end='')

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
            sleep(2)
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')
            exit(0)


    def analyse(self, signal):
        return np.argwhere(signal > self.noise)


if __name__=='__main__':
    colorama.init()
    captcha = AcCAPPCHA()
    captcha.run()