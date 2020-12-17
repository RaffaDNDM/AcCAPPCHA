import numpy as np
import pyaudio
import wave
import sys
from scipy.io import wavfile
from time import sleep
import threading
from termcolor import cprint, colored
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
import time
from pynput import keyboard
import sys
import argparse


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

        #wf = wave.open(tempfile.gettempdir()+'/tmp.wav', 'wb')
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

    def audio(self, folder, option):
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
            self.verify(folder, option)
            self.KILLED = True
        finally:
            self.mutex.release()

    def password_from_user(self):
        sleep(1)
        cprint('Insert the password', 'red')
        
        try:
            char_user = sys.stdin.read(1)
            self.TIMES.append(int(round(time.time() * 1000)))

            while char_user != '\n':
                self.password.append(char_user)
                char_user = sys.stdin.read(1)
                self.TIMES.append(int(round(time.time() * 1000)))

            self.TIMES = self.TIMES[:-1]

            if self.password == 'EXIT':
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

    def verify(self, folder, option):
        '''
        Plot the audio signal with name filename with
        highlighted peaks and FFT of push and hit peaks
        
        Args:
            filename (str): name of wav file
            analysis (ExtractFeatures): ExtractFeatures related to the
                                        wav file with name filename
        '''

        #Time sample step
        self.ts, self.time_ms, self.signal = utility.signal_adjustment(self.RATE, self.signal)

		#Initialize the figure
        fig = plt.figure('AUDIO')

        peak_indices = self.analyse(self.signal)
        #Plot of press peak and hit peak with the signal
        plt.plot(self.time_ms*self.ts, self.signal[self.time_ms], color='blue')
        
        char_times = [[peak_indices[0][0],], ]
        
        start = 0
        for i in peak_indices[1:]:
            if (i[0]-char_times[start][0])>self.CHAR_RANGE:
                char_times.append([i[0],])
                start += 1
            else:
                char_times[start].append(i[0])
                plt.plot(i[0]*self.ts, self.signal[i[0]], color=self.COLORS[start%len(self.COLORS)], marker='x')    
        
        plt.tick_params(axis='both', which='major', labelsize=6)
        fig.savefig(self.OUTPUT_IMG)
        self.predict_keys(char_times, folder, option)

    def predict_keys(self, char_times, folder, option):
        net = nn.NeuralNetwork(option, folder)
                
        count = 0
        model = VGG16(weights='imagenet', include_top=False)
                
        for list_time in char_times:
            #Evaluation of press peak and hit peaks
            analysis = ef.ExtractFeatures(self.RATE, self.signal[list_time[0]:list_time[-1]])            
            fig = plt.figure('CHARACTER'+str(count))
            fig.tight_layout(pad=3.0)
            
            if utility.OPTIONS[option]=='touch':
                features = analysis.extract(original_signal=self.signal, index=list_time[0])
                
                if features is None:
                    print(f'{count} ---> EMPTY SEQUENCE')
                    count+=1
                    continue

                touch_feature = features['touch']
                hit_feature = features['hit']
                touch_X = touch_feature.fft_signal.reshape((1, 66))
                print(colored(f'{count} ---> ', 'red')+f'{net.test(touch_X)}')

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
                
                if features is None:
                    print(f'{count} ---> EMPTY SEQUENCE')
                    count+=1
                    continue

                touch_feature = features['touch']
                hit_feature = features['hit']            
                touch_X = touch_feature.fft_signal
                hit_X = hit_feature.fft_signal
                touch_hit_X = np.concatenate((touch_X, hit_X)).reshape((1, 132))
                #cprint(touch_hit_X.shape, 'red')
                print(colored(f'{count} ---> ', 'red')+f'{net.test(touch_hit_X)}')

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
        
                if features is None:
                    print(f'{count} ---> EMPTY SEQUENCE')
                    count+=1
                    continue
                
                img_feature = np.concatenate((self.signal[features[0]], self.signal[features[1]]))
                spectrum, freqs, t, img_array = ax.specgram(img_feature, NFFT=len(features[0]), Fs=analysis.fs)
                
                if not os.path.exists(folder+f'spectrum_img'):
                    os.mkdir(folder+f'spectrum_img')

                fig.savefig(folder+f'spectrum_img/{count}.jpg', dpi=300)
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

                features = utility.extract(model, folder+f'spectrum_img/{count}.jpg')
                fig.savefig(f'Char {count}.jpg')
                print(colored(f'{count} ---> ', 'red')+f'{net.test(features)}')

            count+=1

    def run(self, folder, option):
        '''
        Start keylogger and audio recorder
        '''
        self.TIMES = []
        self.password = []

        try:
            while not self.KILLED:
                self.noise_evaluation()
                audio_logger = threading.Thread(target=self.audio, args=(folder,option))
                password_logger = threading.Thread(target=self.password_from_user)
                audio_logger.start()
                password_logger.start()
                password_logger.join()

                #with keyboard.Listener(on_press=self.password_key) as psswd_listener:
                    #Manage keyboard input
                #    psswd_listener.join()

                audio_logger.join()                

        except KeyboardInterrupt:
            #Terminate the keylogger (detected CTRL+C)
            self.KILLED = True
            sleep(2)
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')
            exit(0)

    def analyse(self, signal):
        return np.argwhere(signal > self.noise)


def args_parser():
    '''
    Parser of command line arguments
    '''
    #Parser of command line arguments
    parser = argparse.ArgumentParser()
    
    #Initialization of needed arguments
    parser.add_argument("-dir", "-d", 
                        dest="dir", 
                        help="""If specified, it is the path of the folder with the
                                3 subfolders: 'touch/', 'touch_hit/' and 'spectrum/'.
                                Each of them contains a folder called 'model/' that
                                contains information of the pre-trained network that
                                classifies thekeys of the keyboard""")

    parser.add_argument("-time", "-t", 
                        dest="time", 
                        help="""If specified, it performs human verification through
                                analysis of elapsed time between insertion of 2 keys
                                and time between 2 peaks""",
                        action='store_true')

    parser.add_argument("-deep", "-dl", 
                        dest="deep",
                        help="""If specified, it performs human verification through
                                deep learning method (predicting pressed keys)""",
                        action='store_true')

    #Parse command line arguments
    args = parser.parse_args()

    #ERROR if specified file or output but not plot option
    if not (args.deep and args.dir):
        cprint('[OPTION ERROR]', 'red', end=' ')
        print('The directory specified by', end=' ')
        cprint('-d', 'blue', end=' ')
        print("must be specified only with option", end=' ')
        cprint('-deep', 'green', end='\n\n')
        exit(0)

    return args.dir, args.time, args.deep

def main():
    folder, time_option, deep_option = args_parser()
    
    if deep_option:
        option = utility.select_option_feature()
    
    colorama.init()
    captcha = AcCAPPCHA()
    captcha.run(folder, option)

if __name__=='__main__':
    main()