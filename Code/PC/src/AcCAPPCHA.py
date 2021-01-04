import numpy as np
import pyaudio
import wave
import sys
from scipy.io import wavfile
import time
from time import sleep
import threading
from termcolor import cprint, colored
import os
import utility
from utility import num_samples
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, use
import colorama
from pynput import keyboard
import sys
import argparse
#Deep learning only
from collections import Counter
import ExtractFeatures as ef
import NeuralNetwork as nn
from tensorflow.keras.applications.vgg16 import VGG16
import logging
import tensorflow as tf 
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from SecureElement import SecureElement

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
    PASSWORD_STRING = colored('password: ', 'red')

    #Slant Rilief title
    NAME_CAPPCHA= '_____/********+_________________________/********+_____/********+_____/************+____/************+__________/********+__/**+________/**+_____/********+____\n'+ \
                  ' ___/************+____________________/**+////////____/************+__+/**+/////////**+_+/**+/////////**+_____/**+////////__+/**+_______+/**+___/************+__\n' + \
                  '  __/**+/////////**+_________________/**+/____________/**+/////////**+_+/**+_______+/**+_+/**+_______+/**+___/**+/___________+/**+_______+/**+__/**+/////////**+_\n' + \
                  '   _+/**+_______+/**+_____/********__/**+_____________+/**+_______+/**+_+/*************/__+/*************/___/**+_____________+/**************+_+/**+_______+/**+_\n' + \
                  '    _+/**************+___/**+//////__+/**+_____________+/**************+_+/**+/////////____+/**+/////////____+/**+_____________+/**+/////////**+_+/**************+_\n' + \
                  '     _+/**+/////////**+__/**+_________+//**+____________+/**+/////////**+_+/**+_____________+/**+_____________+//**+____________+/**+_______+/**+_+/**+/////////**+_\n' + \
                  '      _+/**+_______+/**+_+//**+_________+///**+__________+/**+_______+/**+_+/**+_____________+/**+______________+///**+__________+/**+_______+/**+_+/**+_______+/**+_\n' + \
                  '       _+/**+_______+/**+__+///********____+////********+_+/**+_______+/**+_+/**+_____________+/**+________________+////********+_+/**+_______+/**+_+/**+_______+/**+_\n' + \
                  '        _+///________+///_____+////////________+/////////__+///________+///__+///______________+///____________________+/////////__+///________+///__+///________+///__\n\n'

    NAME_CAPPCHA2= '                  _____          _____  _____   _____ _    _\n'+ \
                  '     /\\          / ____|   /\\   |  __ \\|  __ \\ / ____| |  | |   /\\ \n' + \
                  '    /  \\   ___  | |       /  \\  | |__) | |__) | |    | |__| |  /  \\ \n' + \
                  '   / /\\ \\ / __| | |      / /\\ \\ |  ___/|  ___/| |    |  __  | / /\\ \\ \n' + \
                  '  / ____ \\ (__  | |____ / ____ \\| |    | |    | |____| |  | |/ ____ \\ \n' + \
                  ' /_/    \\_\\___|  \_____/_/    \\_\\_|    |_|     \_____|_|  |_/_/    \\_\\ \n\n'

    COLORS = ['g', 'r', 'c', 'm', 'y']
    #Tolerance [-100 ms, 100 ms] with respect to stored time instants
    TIME_THRESHOLD = 0.100
    MAIN_COLOR = 'red'
    SHADOW_COLOR = 'yellow'
    BACKGROUND_COLOR = 'blue'

    """
    AcCAPPCHA implementation.

    Args:
        time_option (bool): True if you want to use time correspondence 
                            method, False otherwise

        dl_option (bool): True if you want to use deep learning method, 
                          False otherwise

        plot_option (bool): True if you want to plot the graphics of
                            recorded audio files, False otherwise

        debug_option (bool): True if you want to show more debugging info
                             during the executtion, False otherwise

    Attributes:

                        [****ACQUISITION PARAMETERS****]
        ___________________________________________________________________________
        CHUNK (int): (by default 1024)

        FORMAT (pyaudio format): (by default pyaudio.paInt16)

        CHANNELS (int): (by default 2)

        RATE (int): Sampling rate (by default 44100)

        SECONDS_NOISE (float): Seconds of audio file for evaluation of noise
                               (by default 2 seconds)
        ___________________________________________________________________________


            [****COMMUNICATION BETWEEN RECORDER AND MANAGER OF PASSWORD****]        
        ___________________________________________________________________________
        mutex (threading.Lock): Mutual exclusion LOCK

        COMPLETED_INSERT (bool): True if '\r' is inserted by user during the 
                                 insertion of the password (insertion completed), 
                                 False otherwise

        KILLED (bool): True when the program detects CTRL+C during the execution of 
                the main thread and so the audio recorder ends the acqisition
        ___________________________________________________________________________
        
        
                    [****USEFUL ATTRIBUTES FOR VERIFICATION****]
        ___________________________________________________________________________
        CHAR_RANGE (int): Number of samples for each char peak
        
        ENTER_RANGE (int): Number of samples for last character (ENTER) peak
        
        TIME_OPTION (bool): True if you want to use time correspondence 
                            method, False otherwise
        
        DL_OPTION (bool): True if you want to use deep learning method, 
                          False otherwise
        
        PLOT_OPTION (bool): True if you want to plot the graphics of
                            recorded audio files, False otherwise
        
        DEBUG (bool): True if you want to show more debugging info
                             during the executtion, False otherwise

        TIME_THRESHOLD (float): Threshold in seconds for time correspondence

        VERIFIED (bool): True if a human, False otherwise
        
        NOISE (np.int16): Value of noise, extracted as maximum value of the 
                          recorded noise signal of SECONDS_NOISE seconds
        
        SIGNAL (np.array): Signal from recording session of audio during the
                           password insertion of the user

        PASSWORD (str): Password inserted by the user

        TIMES (list): List of time instants related to pressed keys, one for
                      each character inserted by the user
        ___________________________________________________________________________


                            [****OTHER ATTRIBUTES****]
        ___________________________________________________________________________
        NAME_CAPPCHA (str): Big title name of the application

        NAME_CAPPCHA2 (str): Small title name of the application

        PLOT_FOLDER (str): Path of the folder that will contain plot images

        SPECTRUM_FOLDER (str): Subfolder of PLOT_FOLDER that will contain only the
                               spectrum images (used by NN for prediction)

        CHAR_SPECTRUM_IMG (str): Format string for name of spectrum images

        WAVE_FOLDER (str): Subfolder of PLOT_FOLDER that will contain the image of
                           audio signal with highlighted char ranges, touch and hit
                           peaks, extracted features and spectrograms 
                           (used by NN for prediction)

        FULL_WAVE_IMG (str): Name of image with the whole audio signal recorded 
                             during the insertion of the password and with 
                             highlighted char intervals

        CHAR_WAVE_PEAKS_IMG (str): Format string for name of images with only a 
                                   couple of touch and hit peak highlighted and 
                                   their FFT coefficients

        CHAR_WAVE_SPECTRUM_IMG (str): Format string for name of images with only a 
                                      couple of touch and hit peak highlighted and 
                                      their spectrogram image

        TIME_MS (np.array): Sequence of time instants in ms related to each sample
                            of SIGNAL

        TS (np.array): Sampling period of recording session (1.0/RATE)

        PASSWORD_STRING (str): String shown to user during the insertion of the 
                               password

        COLORS (list): List of pyplot colors, used to color char ranges

        MAIN_COLOR (str): Main color in NAME_CAPPCHA title

        SHADOW_COLOR (str): Color of shadows in NAME_CAPPCHA title

        BACKGROUND_COLOR (str): Color of background in NAME_CAPPCHA title
        ___________________________________________________________________________
    """

    def __init__(self, time_option, dl_option, plot_option, debug_option):
        self.mutex = threading.Lock()
        self.COMPLETED_INSERT = False
        self.KILLED = False
        self.CHAR_RANGE = num_samples(self.RATE, 0.10)
        self.ENTER_RANGE = num_samples(self.RATE, 0.2)
        self.TIME_OPTION = time_option
        self.DL_OPTION = dl_option
        self.PLOT_OPTION = plot_option
        self.DEBUG = debug_option
        self.VERIFIED = False
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('+', colored('\\', self.SHADOW_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('|', colored('|', self.SHADOW_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('/', colored('/', self.SHADOW_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('_', colored('_', self.BACKGROUND_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('*', colored('\\', self.MAIN_COLOR))
        cprint('\n'+self.NAME_CAPPCHA, 'blue')

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
        '''
        Evaluate noise recording an audio long SECONDS_NOISE seconds
        and computing maximum value of the signal
        '''
        
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

        #Reading audio file
        audio_string = b''.join(frames)
        temp_signal = []
        
        for i in range(0, len(audio_string), 4):
            #print(f'{i}:{audio_string[i:i+4]}')
            temp_signal.append(np.frombuffer(audio_string[i:i+4], dtype=np.int16, count=2))
        
        signal = np.array(temp_signal)
        
        #Analysis of audio signal
        self.NOISE = np.max(np.abs(signal))

    def audio(self, folder, option):
        '''
        Record audio while password insertion and verify if the user
        was a human or a bot

        Args:
            folder (str): Folder containing the subfolders related
                          to the extracted features

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction

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
        
        audio_string = b''.join(frames)
        temp_signal = []
        for i in range(0, len(audio_string), 4):
            #print(f'{i}:{audio_string[i:i+4]}')
            temp_signal.append(np.frombuffer(audio_string[i:i+4], dtype=np.int16, count=2))
        
        signal = np.array(temp_signal)

        #Analysis of audio signal
        if self.DEBUG:
            colored(utility.LINE, 'blue')
            cprint('Audio length:', 'green', end=' ')
            print(f'{len(signal)} samples ({len(signal)/self.RATE} s)')
            cprint('Password length:', 'green', end=' ')
            print(f'{len(self.PASSWORD)} characters')

        self.mutex.acquire()
        try:
            self.COMPLETED_INSERT = False    
            self.SIGNAL = signal[:-self.ENTER_RANGE]
            self.verify(folder, option)
        finally:
            self.mutex.release()

    def password_from_user(self):
        '''
        Take the password, one character a time, from the user
        '''
        
        char_user = utility.getchar()
        self.TIMES.append(time.perf_counter())
            
        if char_user != '\r':
            self.PASSWORD += char_user
            psswd = self.obfuscate()
            print(f'\r{self.PASSWORD_STRING}{psswd}', end='')
            return True

        else:
            psswd = self.obfuscate()
            print(f'\r{self.PASSWORD_STRING}{psswd}')

            self.mutex.acquire()
            try:
                self.COMPLETED_INSERT = True
            finally:
                self.mutex.release()

            return False

    def verify(self, folder, option):
        '''
        Verify if the password was inserted by a human or a bot
        
        Args:
            folder (str): Folder containing the subfolders related
                          to the extracted features

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction
        '''

        #Time sample step
        self.TS, self.TIME_MS, self.SIGNAL = utility.signal_adjustment(self.RATE, self.SIGNAL)

        if self.PLOT_OPTION:
            #Initialize the figure
            fig = plt.figure('AUDIO')
            #Plot of press peak and hit peak with the signal
            plt.plot(self.TIME_MS*self.TS, self.SIGNAL[self.TIME_MS], color='blue')
        
        char_times = []
        peak_indices = self.analyse(self.SIGNAL)
            
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
                        plt.plot(i[0]*self.TS, self.SIGNAL[i[0]], color=self.COLORS[start%len(self.COLORS)], marker='x')
        
            if self.DL_OPTION and self.TIME_OPTION:
                cprint(utility.LINE, 'blue')
                verified_time, checked_char_times = self.correspond_time(char_times)

                self.VERIFIED = verified_time

                if verified_time:
                    self.VERIFIED = self.correspond_keys(checked_char_times, folder, option)
                    
            elif self.DL_OPTION:
                cprint(utility.LINE, 'blue')
                self.VERIFIED = self.correspond_keys(char_times, folder, option)

            elif self.TIME_OPTION:
                self.VERIFIED = self.correspond_time(char_times)[0]

        cprint(utility.LINE, 'blue')

        if self.DEBUG:
            print(colored('Human: ', 'green')+f'\n{self.VERIFIED}', end='\n\n')

        if self.PLOT_OPTION:
            plt.tick_params(axis='both', which='major', labelsize=6)
            fig.savefig(self.PLOT_FOLDER+self.WAVE_FOLDER+self.FULL_WAVE_IMG)

    def correspond_time(self, char_times):
        '''
        Verify if there is a time correspondence between time
        instants of insertion of each character of the password
        and instants from the audio file 
        (recorded during insertion of the password)

        Args:
            char_times (list): List of lists 
                               (each one containing indices of signal
                               samples in signal and representing 
                               indices of each time subwindow where
                               peaks must be found)

        Returns:
            response (bool): True if human, False if bot
            
            checked_char_times (list): List of only lists in char_times
                                       related to time instants of each
                                       char insertion of the password

        '''
        
        length_psswd = len(self.PASSWORD)
        if len(char_times) < length_psswd:
            return False, None

        peak_times = []
        for list_time in char_times:
            peak_times.append(list_time[np.argmax(self.SIGNAL[list_time])])

        cprint(self.TIMES,'green')
        for x in peak_times:
            cprint(x/self.RATE, 'cyan', end=' ')
        print('')
        
        for i in range(0, len(char_times)):
            if len(char_times) - i < length_psswd:
                return False, None

            checked_char_times = [char_times[i],]
            start = peak_times[i]/self.RATE
            j=i+1
            count_verified = 1 #already verified element in i
            
            while count_verified < length_psswd and j < len(char_times):
                if (len(char_times) -j) < (length_psswd-count_verified):
                    break

                if ((peak_times[j]/self.RATE)-start) < (self.TIMES[count_verified]-self.TIME_THRESHOLD):
                    j += 1
                elif ((peak_times[j]/self.RATE)-start) < (self.TIMES[count_verified]+self.TIME_THRESHOLD):
                    count_verified += 1
                    checked_char_times.append(char_times[j])
                    j += 1
                elif ((peak_times[j]/self.RATE)-start) > (self.TIMES[count_verified]+self.TIME_THRESHOLD):
                    break

                if count_verified == length_psswd:
                    return True, checked_char_times

        return False, None

    def correspond_keys(self, char_times, folder, option):
        '''
        Verify if there is a character correspondence between 
        character of the password and the predicted ones from
        the audio file 
        
        Args:
            char_times (list): List of lists 
                               (each one containing indices of signal
                               samples in signal and representing 
                               indices of each time subwindow where
                               peaks must be found)

            folder (str): Folder containing the subfolders related
                          to the extracted features

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction

        Returns:
            response (bool): True if human, False if bot
        '''
        
        keys = self.predict_keys(char_times, folder, option)

        self.PASSWORD = self.remove_backspace()

        if len(keys)<len(self.PASSWORD):
            return False

        i=0
        for key_list in keys:
            if self.PASSWORD[i] in key_list:
                i += 1
            
            if i==len(self.PASSWORD):
                break
        
        return i==len(self.PASSWORD)

    def predict_keys(self, char_times, folder, option):
        '''
        Extraction of features from audio peaks of char_times windows
        and prediction through deep learning of 10 most probable labels

        Args:
            char_times (list): List of lists 
                               (each one containing indices of signal
                               samples in signal and representing 
                               indices of each time subwindow where
                               peaks must be found)

            folder (str): Folder containing the subfolders related
                          to the extracted features

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction

        Returns:
            keys (list): List of lists 
                         (each one contains the 10 most probable 
                          labels for a window of indices in char_times)
        '''

        net = nn.NeuralNetwork(option, folder)
        count = 0
        model = VGG16(weights='imagenet', include_top=False)
        keys = []

        for list_time in char_times:
            #Evaluation of press peak and hit peaks
            analysis = ef.ExtractFeatures(self.RATE, self.SIGNAL[list_time[0]:list_time[-1]])
            
            if utility.OPTIONS[option]=='touch':
                features = analysis.extract(original_signal=self.SIGNAL, index=list_time[0])
                
                if features is None:
                    print(colored(f'{count} ---> ', 'red')+'EMPTY SEQUENCE')
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
                features = analysis.extract(original_signal=self.SIGNAL, index=list_time[0])
                
                if features is None:
                    print(colored(f'{count} ---> ', 'red')+'EMPTY SEQUENCE')
                    count+=1
                    continue

                touch_feature = features['touch']
                hit_feature = features['hit']            
                touch_X = touch_feature.fft_signal
                hit_X = hit_feature.fft_signal
                touch_hit_X = np.concatenate((touch_X, hit_X)).reshape((1, 132))                
                #cprint(touch_hit_X.shape, 'red')
                keys.append(net.test(touch_hit_X))
                print(colored(f'{count} ---> ', 'red')+utility.results_to_string(keys[-1]))
                
                if self.PLOT_OPTION:
                    self.plot_features(count, touch_feature, hit_feature)

            else:
                #Extraction of features
                fig, ax = plt.subplots(1)
                fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
                features = analysis.extract(original_signal=self.SIGNAL, index=list_time[0], spectrum=True)
        
                if features is None:
                    print(colored(f'{count} ---> ', 'red')+'EMPTY SEQUENCE')
                    count+=1
                    continue
                
                img_feature = np.concatenate((self.SIGNAL[features[0]], self.SIGNAL[features[1]]))
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
        '''
        Plot touch feature and hit feature for the count-th
        peak of SIGNAL

        count (int): Order of the peak in char_times sequence

        touch_feature (ExtractFeatures.Feature): Feature of touch peak
                                                 related to the count-th
                                                 peak 

        hit_feature (ExtractFeatures.Feature): Feature of hit peak
                                               related to the count-th
                                               peak
        '''
        fig = plt.figure('CHARACTER'+str(count))
        fig.tight_layout(pad=1.0)
        gs = fig.add_gridspec(2, 2)
        gs.update(hspace=0.5)
        s_top = fig.add_subplot(gs[0, :])
        s1 = fig.add_subplot(gs[1,0])
        s2 = fig.add_subplot(gs[1,1])
    
        #Plot of press peak and hit peak with the signal
        s_top.plot(self.TIME_MS*self.TS, self.SIGNAL[self.TIME_MS], color='blue')
        s_top.plot(touch_feature.peak*self.TS, self.SIGNAL[touch_feature.peak], color='red')
        s_top.plot(hit_feature.peak*self.TS, self.SIGNAL[hit_feature.peak], color='green')
        s_top.set_title('Amplitude', fontsize = 10)
        s_top.set_xlabel('Time(ms)', fontsize = 10)
        s_top.tick_params(axis='both', which='major', labelsize=6)

        #Plot FFT double-sided transform of PRESS peak
        s1.plot(touch_feature.freqs, touch_feature.fft_signal, color='red')
        s1.set_xlabel('Frequency (Hz)', fontsize = 10)
        s1.set_ylabel('FFT of TOUCH PEAK', fontsize = 10)
        s1.tick_params(axis='both', which='major', labelsize=6)
        s1.set_xscale('log')
        s1.set_yscale('log')

        #Plot FFT single-sided transform of HIT peak
        s2.plot(hit_feature.freqs, hit_feature.fft_signal, color='green')
        s2.set_xlabel('Frequency(Hz)', fontsize = 10)
        s2.set_ylabel('FFT of HIT PEAK', fontsize = 10)
        s2.tick_params(axis='both', which='major', labelsize=6)
        s2.set_xscale('log')
        s2.set_yscale('log')
        s2.yaxis.set_label_position("right")
        s2.yaxis.tick_right()
        fig.savefig((self.PLOT_FOLDER+self.WAVE_FOLDER+self.CHAR_WAVE_PEAKS_IMG).format(count))
        plt.close()
        #plt.show()

    def plot_spectrum(self, count, features, img_feature):
        '''
        Plot touch feature and hit feature for the count-th
        peak of SIGNAL

        count (int): Order of the peak in char_times sequence

        features (numpy.array): Indices of touch and hit peaks samples
                                related to the count-th peak

        img_feature (numpy.array): Feature (FFT transform) of hit peak 
                                   concatenated to Feature of touch peak
                                   related to the count-th peak
        '''

        fig = plt.figure('CHARACTER'+str(count))
        fig.tight_layout(pad=3.0)
        gs = fig.add_gridspec(2, 1)
        s_top = fig.add_subplot(gs[0, 0])
        s_bottom = fig.add_subplot(gs[1,0])

        #Plot of press peak and hit peak with the signal
        s_top.plot(self.TIME_MS*self.TS, self.SIGNAL[self.TIME_MS], color='blue')
        s_top.plot(features[0]*self.TS, self.SIGNAL[features[0]], color='red')
        s_top.plot(features[1]*self.TS, self.SIGNAL[features[1]], color='green')
        s_top.set_title('Amplitude')
        s_top.set_xlabel('Time(ms)')
        s_top.tick_params(axis='both', which='major', labelsize=6)

        s_bottom.specgram(img_feature, NFFT=len(features[0]), Fs=self.RATE)
        fig.savefig((self.PLOT_FOLDER+self.WAVE_FOLDER+self.CHAR_WAVE_SPECTRUM_IMG).format(count))

    def run(self, folder, option, username):
        '''
        Start password insertion for user and audio recorder 
        and verify if the user is a human or a bot
        
        Args:
            folder (str): Folder containing the subfolders related
                          to the extracted features

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction
    
            username (str): Username of the user for which we check
                            that the password was inserted by a human    

        Returns:
            VERIFIED (bool): True if human, False if bot
        '''
        if self.DL_OPTION and utility.OPTIONS[option]=='spectrum':
            if not os.path.exists(self.PLOT_FOLDER+self.SPECTRUM_FOLDER):
                os.mkdir(self.PLOT_FOLDER+self.SPECTRUM_FOLDER)
            else:
                files = os.listdir(self.PLOT_FOLDER+self.SPECTRUM_FOLDER)

                for f in files:
                    os.remove(self.PLOT_FOLDER+self.SPECTRUM_FOLDER+f)

        try:
            count_trials = 0
            while not self.VERIFIED and count_trials < 3:
                self.COMPLETED_INSERT = False
                self.noise_evaluation()

                self.TIMES = []
                self.PASSWORD = ''
                
                if count_trials > 0:
                    cprint('Try to stay in a quiet environment', 'yellow')
                
                cprint('password:', 'red', end=' ')
                audio_logger = threading.Thread(target=self.audio, args=(folder,option))
                audio_logger.start()

                no_end = True
                sleep(1)
                while no_end:
                    no_end = self.password_from_user()
                
                first = self.TIMES[0]
                self.TIMES = [t-first for t in self.TIMES[:-1]]

                audio_logger.join()

                with SecureElement('127.0.0.1', 8080) as s:
                    check = s.sign(str(self.VERIFIED))        
                    
                    if check:
                        msg = s.credentials(username, self.PASSWORD)
            #            msg = s.credentials('raffaeledndm', 'ciao')
                
                        with open('../dat/html/response.html', 'w') as f:
                            f.write(msg)

                            import webbrowser, os
                            webbrowser.open('file://' + os.path.abspath('../dat/html/response.html'))
                            
                        #print(msg)
                        #msg = s.credentials('RaffaDNDM', 'hello35')
                        #msg = s.credentials('JohnSM', 'password4')
                        #msg = s.credentials('CristiFB', 'byebye12')
                        #msg = s.credentials('IreneRMN', 'flower10')
                        break
                    else:
                        count_trials += 1

                #With open of secure element
            return self.VERIFIED              

        except KeyboardInterrupt:
            self.KILLED = True
            sleep(1)
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')
            exit(0)

    def analyse(self, signal):
        '''
        Find the indices of all the signal samples with
        values higher than the noise maximum value

        Args:
            signal (numpy.array): Signal of which I want to find
                                  the feasible indices 
        '''

        return np.argwhere(signal > self.NOISE)

    def obfuscate(self):
        '''
        Obfuscate PASSWORD by replacing each character
        with a *

        Returns:
            psswd (str): Obfuscated PASSWORD
        '''
        
        psswd = ''

        for x in self.PASSWORD:
            if x == '\b':
                psswd += '\b \b'
            else:
                psswd += '*'

        return psswd

    def remove_backspace(self):
        '''
        Remove backspaces inserted in the password by
        replacing '\b' with '\b \b' 
        (effect of \b on strings in every OS)

        Returns:
            psswd (str): Updated PASSWORD
        '''
        
        psswd = ''
        for x in self.PASSWORD:
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

    parser.add_argument("-debug", "-dbg", 
                        dest="debug",
                        help="""If specified, it shows debug info""",
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

    return args.dir, args.time, args.deep, args.plot, args.debug

def main():
    colorama.init()
    folder, time_option, dl_option, plot_option, debug_option = args_parser()

    option = -1
    
    if dl_option:
        option = utility.select_option_feature()

    captcha = AcCAPPCHA(time_option, dl_option, plot_option, debug_option)
    cprint(f'   Authentication\n{utility.LINE}', 'blue')
    username = input(colored('username: ', 'red'))
    check = captcha.run(folder, option, username)

    if check:
        cprint("#################################", 'yellow')
        cprint('#############', 'yellow', end=' ')
        cprint("HUMAN", 'magenta', end=' ')
        cprint('#############', 'yellow')
        cprint("#################################", 'yellow', end='\n\n')
    else:
        cprint("#################################", 'yellow')
        cprint('##############', 'yellow', end=' ')
        cprint("BOT", 'magenta', end=' ')
        cprint('##############', 'yellow')
        cprint("#################################", 'yellow', end='\n\n')


if __name__=='__main__':
    main()