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
from utility import num_samples
import tempfile
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, use
import colorama
import time
from pynput import keyboard
import sys
import argparse
import datetime
from timeit import default_timer as timer
import timeit
import array
#Deep learning only
from collections import Counter
import ExtractFeatures as ef
import NeuralNetwork as nn
from tensorflow.keras.applications.vgg16 import VGG16

class AcCAPPCHA:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    SECONDS_NOISE = 2.0

    PLOT_FOLDER = '../dat/img/'

    SPECTRUM_FOLDER = 'spectrum/'
    CHAR_SPECTRUM_IMG = 'spectrum_{:02d}.png'

    WAVE_FOLDER = 'wave/'
    FULL_WAVE_IMG = 'audio_from_user.png'
    CHAR_WAVE_PEAKS_IMG = 'peak_{:02d}.png'
    CHAR_WAVE_SPECTRUM_IMG = 'spectrum_{:02d}.png'

    COLORS = ['g', 'r', 'c', 'm', 'y']
    #Tolerance [-5 ms, 5 ms] with respect to peaks
    TIME_THRESHOLD = num_samples(RATE ,0.005)

    def __init__(self, time_option, dl_option, plot_option):
        self.mutex = threading.Lock()
        self.COMPLETED_INSERT = False
        self.KILLED = False
        self.CHAR_RANGE = num_samples(self.RATE, 0.10)
        self.ENTER_RANGE = num_samples(self.RATE, 0.2)
        self.TIME_OPTION = time_option
        self.DL_OPTION = dl_option
        self.PLOT_OPTION = plot_option

        if not os.path.exists(self.PLOT_FOLDER):
            os.mkdir(self.PLOT_FOLDER)

        if self.PLOT_OPTION:
            if not os.path.exists(self.PLOT_FOLDER+self.WAVE_FOLDER):
                os.mkdir(self.PLOT_FOLDER+self.WAVE_FOLDER)
            else:
                files = os.listdir(self.PLOT_FOLDER+self.WAVE_FOLDER)

                for f in files:
                    os.remove(self.PLOT_FOLDER+self.WAVE_FOLDER+f)

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

        '''
        #wf = wave.open(tempfile.gettempdir()+'/tmp.wav', 'wb')
        wf = wave.open('../dat/noise.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        #Reading audio file
        fs, signal = wavfile.read('../dat/noise.wav')
        '''

        audio_string = b''.join(frames)
        signal = np.frombuffer(audio_string, dtype=np.int16)
        
        #Analysis of audio signal
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

        '''
        wf = wave.open('../dat/tmp.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


        #Reading audio file
        fs, signal = wavfile.read('../dat/tmp.wav')
        '''

        audio_string = b''.join(frames)
        signal = np.frombuffer(audio_string, dtype=np.int16)
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
        char_user = utility.getchar()
        
        if char_user != '\r':
            self.password += char_user
            psswd = self.obfuscate()
            print(f'\r{psswd}', end='')
            self.TIMES.append(time.time())
            return True

        else:
            psswd = self.obfuscate()
            print(f'\r{psswd}')

            self.mutex.acquire()
            try:
                self.COMPLETED_INSERT = True
            finally:
                self.mutex.release()

            return False

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

        if self.PLOT_OPTION:
            #Initialize the figure
            fig = plt.figure('AUDIO')
            #Plot of press peak and hit peak with the signal
            plt.plot(self.time_ms*self.ts, self.signal[self.time_ms], color='blue')
        
        char_times = []
        peak_indices = self.analyse(self.signal)
            
        if peak_indices.size > 0:
            char_times = [[peak_indices[0][0],], ]
            
            start = 0
            for i in peak_indices[1:]:
                if (i[0]-char_times[start][0])>self.CHAR_RANGE:
                    char_times.append([i[0],])
                    start += 1
                else:
                    char_times[start].append(i[0])

                    if self.PLOT_OPTION:
                        plt.plot(i[0]*self.ts, self.signal[i[0]], color=self.COLORS[start%len(self.COLORS)], marker='x')
        
            if self.DL_OPTION and self.TIME_OPTION:
                verified_time, checked_char_times = self.correspond_time(char_times)

                self.VERIFIED = verified_time

                if verified_time:
                    self.VERIFIED = self.correspond_keys(checked_char_times, folder, option)
                    
                print(colored(utility.LINE, 'cyan')+f'\n{self.VERIFIED}', end='\n\n')

            elif self.DL_OPTION:
                self.VERIFIED = self.correspond_keys(char_times, folder, option)
                print(colored(utility.LINE, 'cyan')+f'\n{self.VERIFIED}', end='\n\n')

            elif self.TIME_OPTION:
                self.VERIFIED = self.correspond_time(char_times)[0]
                print(colored(utility.LINE, 'cyan')+f'\n{self.VERIFIED}', end='\n\n')

        if self.PLOT_OPTION:
            plt.tick_params(axis='both', which='major', labelsize=6)
            fig.savefig(self.PLOT_FOLDER+self.WAVE_FOLDER+self.FULL_WAVE_IMG)

    def correspond_time(self, char_times):
        length_psswd = len(self.password)
        if len(char_times) < length_psswd:
            return False, None

        peak_times = []
        for list_time in char_times:
            peak_times.append(np.argmax(self.signal[list_time]))

        for i in range(0, len(char_times)):
            if len(char_times) - i < length_psswd:
                return False, None

            checked_char_times = [char_times[i],]
            start = num_samples(self.RATE, i)
            j=i+1
            count_verified = 1 #already verified element in i
            while count_verified < length_psswd and j < len(char_times):
                if (len(char_times) -i) < (length_psswd-count_verified):
                    return False, None

                while j < len(char_times) and \
                      (num_samples(self.RATE, peak_times[j])-self.TIME_THRESHOLD-start) < num_samples(self.RATE, self.TIMES[count_verified]):
                      j += 1

                if j < len(char_times) and \
                      (num_samples(self.RATE, peak_times[j])+self.TIME_THRESHOLD-start) > num_samples(self.RATE, self.TIMES[count_verified]):
                      count_verified += 1
                      checked_char_times.append(char_times[j])

            if count_verified == length_psswd:
                return True, char_times

        return False, None

    def correspond_keys(self, char_times, folder, option):
        keys = self.predict_keys(char_times, folder, option)

        self.password = self.remove_backspace()
        
        if len(keys)<len(self.password):
            return False

        i=0
        for key_list in keys:
            if self.password[i] in key_list:
                i += 1
            
            if i==len(self.password):
                break
        
        return i==len(self.password)

    def predict_keys(self, char_times, folder, option):
        net = nn.NeuralNetwork(option, folder)
        count = 0
        model = VGG16(weights='imagenet', include_top=False)
        keys = []

        for list_time in char_times:
            #Evaluation of press peak and hit peaks
            analysis = ef.ExtractFeatures(self.RATE, self.signal[list_time[0]:list_time[-1]])
            
            if utility.OPTIONS[option]=='touch':
                features = analysis.extract(original_signal=self.signal, index=list_time[0])
                
                if features is None:
                    print(f'{count} ---> EMPTY SEQUENCE')
                    count+=1
                    continue

                touch_feature = features['touch']
                hit_feature = features['hit']
                touch_X = touch_feature.fft_signal.reshape((1, 66))
                keys.append(net.test(touch_X))
                print(colored(f'{count} ---> ', 'red')+utility.results_to_string(keys[-1]))
                
                if self.PLOT_OPTION:
                    self.plot_features(count, touch_feature, hit_feature)

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
                touch_hit_X = np.concatenate((touch_X, hit_X)).reshape((1, 132))                #cprint(touch_hit_X.shape, 'red')
                keys.append(net.test(touch_hit_X))
                print(colored(f'{count} ---> ', 'red')+utility.results_to_string(keys[-1]))
                
                if self.PLOT_OPTION:
                    self.plot_features(count, touch_feature, hit_feature)

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
                
                fig.savefig((self.PLOT_FOLDER+self.SPECTRUM_FOLDER+self.CHAR_SPECTRUM_IMG).format(count), dpi=300)
                plt.close(fig)

                final_features = utility.extract(model, (self.PLOT_FOLDER+self.SPECTRUM_FOLDER+self.CHAR_SPECTRUM_IMG).format(count))
                keys.append(net.test(final_features))
                print(colored(f'{count} ---> ', 'red')+utility.results_to_string(keys[-1]))

                if self.PLOT_OPTION:
                    self.plot_spectrum(count, features, img_feature)

            count+=1

        return keys

    def plot_features(self, count, touch_feature, hit_feature):
        fig = plt.figure('CHARACTER'+str(count))
        fig.tight_layout(pad=3.0)
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
        fig.savefig((self.PLOT_FOLDER+self.WAVE_FOLDER+self.CHAR_WAVE_PEAKS_IMG).format(count))
        #plt.show()

    def plot_spectrum(self, count, features, img_feature):
        fig = plt.figure('CHARACTER'+str(count))
        fig.tight_layout(pad=3.0)
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

        s_bottom.specgram(img_feature, NFFT=len(features[0]), Fs=self.RATE)
        fig.savefig((self.PLOT_FOLDER+self.WAVE_FOLDER+self.CHAR_WAVE_SPECTRUM_IMG).format(count))

    def run(self, folder, option):
        '''
        Start keylogger and audio recorder
        '''
        if self.DL_OPTION and utility.OPTIONS[option]=='spectrum':
            if not os.path.exists(self.PLOT_FOLDER+self.SPECTRUM_FOLDER):
                os.mkdir(self.PLOT_FOLDER+self.SPECTRUM_FOLDER)
            else:
                files = os.listdir(self.PLOT_FOLDER+self.SPECTRUM_FOLDER)

                for f in files:
                    os.remove(self.PLOT_FOLDER+self.SPECTRUM_FOLDER+f)

        try:
            while not self.KILLED:
                self.noise_evaluation()

                self.TIMES = []
                self.password = ''
                cprint('Insert the password', 'red')
                audio_logger = threading.Thread(target=self.audio, args=(folder,option))
                audio_logger.start()

                no_end = True
                sleep(1)
                while no_end:
                    no_end = self.password_from_user()
                
                cprint(len(self.TIMES), 'green')
                first = self.TIMES[0]
                self.TIMES = [t-first for t in self.TIMES]

                audio_logger.join()
                return self.VERIFIED              

        except KeyboardInterrupt:
            self.KILLED = True
            sleep(1)
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')
            exit(0)

    def analyse(self, signal):
        return np.argwhere(signal > self.noise)

    def obfuscate(self):
        psswd = ''

        for x in self.password:
            if x == '\b':
                psswd += '\b \b'
            else:
                psswd += '*'

        return psswd

    def remove_backspace(self):
        psswd = ''
        for x in self.password:
            if x == '\b':
                psswd = psswd[:-1]
            else:
                psswd += x

        return psswd
        

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

    parser.add_argument("-plot", "-p", 
                        dest="plot",
                        help="""If specified, it performs plot of partial results""",
                        action='store_true')

    #Parse command line arguments
    args = parser.parse_args()

    #ERROR 
    #if specified dir but not Deep-learning option or
    #if specified Deep-learning option but not dir
    if (args.deep and not args.dir) or (args.dir and not args.deep):
        cprint('[OPTION ERROR]', 'red', end=' ')
        print('The directory specified by', end=' ')
        cprint('-d', 'blue', end=' ')
        print("must be specified only with option", end=' ')
        cprint('-deep', 'green', end='\n\n')
        exit(0)

    #ERROR 
    #if no methods specified
    if not (args.deep or args.time):
        cprint('[OPTION ERROR]', 'red', end=' ')
        print('You need to specify at least one option between', end=' ')
        cprint('-time', 'blue', end=' ')
        print("and", end=' ')
        cprint('-deep', 'green', end='\n\n')
        exit(0)

    return args.dir, args.time, args.deep, args.plot

def main():
    colorama.init()
    folder, time_option, dl_option, plot_option = args_parser()

    option = -1
    
    if dl_option:
        option = utility.select_option_feature()

    done = False
    count = 0

    username = input(colored('Insert your username:', 'red'))

    while not done or count < 3:
        captcha = AcCAPPCHA(time_option, dl_option, plot_option)
        done = captcha.run(folder, option)
        count += 1

if __name__=='__main__':
    main()