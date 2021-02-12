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

class BlockAccessError(Exception):
    '''
        Exception raised when a user try to run AcCAPPCHA 
        during the block period (after 3 invalid authentication)
    '''
    pass

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
    TIME_THRESHOLD = 0.05
    MAIN_COLOR = 'red'
    SHADOW_COLOR = 'yellow'
    BACKGROUND_COLOR = 'blue'
    BLOCK_FILE = '../dat/html/block.txt'
    MAX_BOT_TRIALS = 3
    MAX_PWD_TRIALS = 3
    BLOCK_DEADLINE_sec = 100

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
                             during the execution, False otherwise

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
                      during the execution, False otherwise

        TIME_THRESHOLD (float): Threshold in seconds for time correspondence

        VERIFIED (bool): True if a human, False otherwise
        
        NOISE (np.int16): Value of noise, extracted as maximum value of the 
                          recorded noise signal of SECONDS_NOISE seconds
        
        SIGNAL (np.array): Signal from recording session of audio during the
                           password insertion of the user

        PASSWORD (str): Password inserted by the user

        TIMES (list): List of time instants related to pressed keys, one for
                      each character inserted by the user

        BLOCK_FILE (str): Complete path of 'block.txt' file to be used to block
                          access to the secure element

        BLOCK_DEADLINE_sec (int): Number of seconds for which a user must be 
                                  blocked after 3 invalid authentications

        MAX_BOT_TRIALS (int): maximum number of times in which the user could be
                              classified as a bot during the execution of AcCAPPCHA
        
        MAX_PWD_TRIALS (int): maximum number of times in which the user could type
                              wrong credentials during the execution of AcCAPPCHA
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
        #If block.txt exists, read it
        if os.path.exists(self.BLOCK_FILE):
            start = 0
            with open(self.BLOCK_FILE, 'r') as f:
                moment = f.read()
                start = time.mktime(time.strptime(moment, "%a, %d %b %Y %H:%M:%S +0000"))
            
            if (start + self.BLOCK_DEADLINE_sec) > time.time():
                #Block period not elapsed
                raise BlockAccessError()
            else:
                #Block period elapsed (remove the file)
                os.remove(self.BLOCK_FILE)

        #Management of communication between the keylogger and the audio recorder
        self.mutex = threading.Lock()
        self.COMPLETED_INSERT = False
        self.KILLED = False
        #Verification parameters
        self.CHAR_RANGE = num_samples(self.RATE, 0.05)
        #self.ENTER_RANGE = num_samples(self.RATE, 0.2)
        self.TIME_OPTION = time_option
        self.VERIFIED = False
        #Options
        self.DL_OPTION = dl_option
        self.PLOT_OPTION = plot_option
        self.DEBUG = debug_option
        #Colored title
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('+', colored('\\', self.SHADOW_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('|', colored('|', self.SHADOW_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('/', colored('/', self.SHADOW_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('_', colored('_', self.BACKGROUND_COLOR))
        self.NAME_CAPPCHA = self.NAME_CAPPCHA.replace('*', colored('\\', self.MAIN_COLOR))
        cprint('\n'+self.NAME_CAPPCHA, 'blue')

        #If plot folder doesn't exist, create it
        if not os.path.exists(self.PLOT_FOLDER):
            os.mkdir(self.PLOT_FOLDER)

        #If plot option specified
        if self.PLOT_OPTION:
            if not os.path.exists(self.PLOT_FOLDER+self.WAVE_FOLDER):
                #Create subfolder for graphics of audio files during the password insertion
                os.mkdir(self.PLOT_FOLDER+self.WAVE_FOLDER)
            else:
                #Delete graphics of audio files already in the subfolder
                files = os.listdir(self.PLOT_FOLDER+self.WAVE_FOLDER)

                for f in files:
                    os.remove(self.PLOT_FOLDER+self.WAVE_FOLDER+f)

    def noise_evaluation(self):
        '''
        Evaluate noise recording an audio long SECONDS_NOISE seconds
        and computing maximum value of the signal
        '''
        
        #Open the stream with microphone
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK)
        
        #Recording
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
            temp_signal.append(np.frombuffer(audio_string[i:i+4], dtype=np.int16, count=2))
        
        signal = np.array(temp_signal)
        
        #Analysis of audio signal
        self.NOISE = np.max(np.abs(signal))

    def audio(self, folder, option, threshold):
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
        
        #Open the stream with microphone
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK)
        
        #Recording
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
            temp_signal.append(np.frombuffer(audio_string[i:i+4], dtype=np.int16, count=2))
        
        signal = np.array(temp_signal)

        #Analysis of audio signal
        if self.DEBUG:
            cprint(utility.LINE, 'blue')
            cprint('Audio length:', 'green', end=' ')
            print(f'{len(signal)} samples ({len(signal)/self.RATE} s)')
            cprint('Password length:', 'green', end=' ')
            print(f'{len(self.PASSWORD)} characters')

        self.mutex.acquire()
        try:
            self.COMPLETED_INSERT = False    
            #self.SIGNAL = signal[:-self.ENTER_RANGE]
            self.SIGNAL = signal
            self.verify(folder, option, threshold)
        finally:
            self.mutex.release()

    def password_from_user(self):
        '''
        Take the password, one character a time, from the user
        '''
        
        #Get a character and store its time instant
        char_user = utility.getchar()
        self.TIMES.append(time.perf_counter())
            
        if char_user != '\r':
            self.PASSWORD += char_user
            psswd = self.obfuscate()
            print(f'\r{self.PASSWORD_STRING}{psswd}', end='')
            return True

        else:
            #End of the password
            self.TIMES = self.TIMES[:-1]
            psswd = self.obfuscate()
            print(f'\r{self.PASSWORD_STRING}{psswd}')

            #Communication with the audio recorder
            self.mutex.acquire()
            try:
                self.COMPLETED_INSERT = True
            finally:
                self.mutex.release()

            return False

    def verify(self, folder, option, threshold):
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
        
        #List of lists (list containing the samples of all the windows)
        char_times = []
        peak_indices = self.analyse(self.SIGNAL)
            
        #If there are samples with values greater than the noise
        if peak_indices.size > 0:
            #Definition of windows
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
        
            #Verification of the user's activity
            if self.DL_OPTION and self.TIME_OPTION:
                #Remove backspaces
                self.PASSWORD = self.remove_backspace()
                #Time correspondence
                cprint(utility.LINE, 'blue')
                verified_time, checked_char_times = self.correspond_time(char_times, threshold)

                #If time correspondence detects bot (verified_time=False)
                self.VERIFIED = verified_time

                #Character correspondence
                if verified_time:
                    self.VERIFIED = self.correspond_keys(checked_char_times, folder, option)
                    
            elif self.DL_OPTION:
                #Remove backspaces
                self.PASSWORD = self.remove_backspace()

                #Character correspondence
                cprint(utility.LINE, 'blue')
                self.VERIFIED = self.correspond_keys(char_times, folder, option)

            elif self.TIME_OPTION:
                #Time correspondence
                self.VERIFIED = self.correspond_time(char_times, threshold)[0]

        cprint(utility.LINE, 'blue')

        if self.DEBUG:
            print(colored('\rHuman: ', 'green')+f'{self.VERIFIED}', end='\n\n')

        if self.PLOT_OPTION:
            #Save the plot of the audio file recorded during the insertion of the password
            plt.tick_params(axis='both', which='major', labelsize=6)
            fig.savefig(self.PLOT_FOLDER+self.WAVE_FOLDER+self.FULL_WAVE_IMG)

    def correspond_time(self, char_times, threshold):
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
        
        length_thresh = int(len(self.PASSWORD)*threshold)
        length_passwd = len(self.PASSWORD)

        #If number of windows lower than number of characters of the password
        #or password too short (trivial password)
        if len(char_times) < length_thresh or length_passwd < 2:
            return False, None

        #Find the peak for every window
        peak_times = []
        for list_time in char_times:
            peak_times.append(list_time[np.argmax(self.SIGNAL[list_time])])

        if self.DEBUG:
            cprint(f'Stored time instants: [', 'green', end='')
            
            for x in self.TIMES:
                print(x, end=colored(', ', 'green'))

            cprint('\b\b]', 'green')
            
            cprint(f'Detected time instants: [', 'green', end='')
            
            for x in peak_times:
                print(x/self.RATE, end=colored(', ', 'green'))
            
            cprint('\b\b]', 'green')

        max_first = length_passwd-length_thresh

        for first in range(0, max_first+1):

            for i in range(0, len(char_times)):
                if len(char_times) - i < length_thresh:
                    return False, None

                checked_char_times = [char_times[i],]
                checked_password_chars = [self.PASSWORD[first],]
                start = peak_times[i]/self.RATE
                #Already verified element in i
                j=i+1
                count_verified = 1
                password_index = first + 1

                #Analysis of next peaks after peak[i]
                while password_index < length_passwd and j < len(char_times):
                    #Not enough remaining audio peaks to find time correspondence
                    if (len(char_times) -j) < (length_thresh - count_verified):
                        break

                    if ((peak_times[j]/self.RATE)-start) < (self.TIMES[password_index]-self.TIMES[first]-self.TIME_THRESHOLD):
                        #Too low time between peak[i] and peak[j] (analyse next peak)
                        j += 1
                    elif ((peak_times[j]/self.RATE)-start) < (self.TIMES[password_index]-self.TIMES[first]+self.TIME_THRESHOLD):
                        #peak[j] corresponds to time instant of the count_verified-th character of the password
                        checked_char_times.append(char_times[j])
                        checked_password_chars.append(self.PASSWORD[password_index])
                        count_verified += 1
                        password_index += 1

                        if count_verified == length_thresh:
                            #Found time correspondence
                            print(checked_password_chars)
                            self.PASSWORD = ''.join(checked_password_chars)
                            return True, checked_char_times

                        j += 1

                    elif ((peak_times[j]/self.RATE)-start) > (self.TIMES[password_index]-self.TIMES[first]+self.TIME_THRESHOLD):
                        #Too much time between peak[i] and peak[j] (no time correspondence)
                        #Change start peak (start = peak[i+1])
                        password_index += 1


        '''
        for i in range(0, len(char_times)):
            if len(char_times) - i < length_thresh:
                return False, None

            checked_char_times = [char_times[i],]
            checked_password_chars = [self.PASSWORD[0],]
            start = peak_times[i]/self.RATE
            #Already verified element in i
            j=i+1
            count_verified = password_index = 1

            #Analysis of next peaks after peak[i]
            while password_index < length_passwd and j < len(char_times):
                #Not enough remaining audio peaks to find time correspondence
                if (len(char_times) -j) < (length_thresh - count_verified):
                    break

                if ((peak_times[j]/self.RATE)-start) < (self.TIMES[password_index]-self.TIMES[i]-self.TIME_THRESHOLD):
                    #Too low time between peak[i] and peak[j] (analyse next peak)
                    j += 1
                elif ((peak_times[j]/self.RATE)-start) < (self.TIMES[password_index]-self.TIMES[i]+self.TIME_THRESHOLD):
                    #peak[j] corresponds to time instant of the count_verified-th character of the password
                    checked_char_times.append(char_times[j])
                    checked_password_chars.append(self.PASSWORD[password_index])
                    count_verified += 1
                    password_index += 1

                    if count_verified == length_thresh:
                        #Found time correspondence
                        print(checked_password_chars)
                        return True, checked_char_times

                    j += 1

                elif ((peak_times[j]/self.RATE)-start) > (self.TIMES[password_index]+self.TIME_THRESHOLD):
                    #Too much time between peak[i] and peak[j] (no time correspondence)
                    #Change start peak (start = peak[i+1])
                    password_index += 1

        #Look for time correspondence            
        for i in range(0, len(char_times)):
            #Not enough remaining audio peaks to find time correspondence
            if len(char_times) - i < length_psswd:
                return False, None

            checked_char_times = [char_times[i],]
            start = peak_times[i]/self.RATE
            #Already verified element in i
            j=i+1
            count_verified = 1
            
            #Analysis of next peaks after peak[i]
            while count_verified < length_psswd and j < len(char_times):
                #Not enough remaining audio peaks to find time correspondence
                if (len(char_times) -j) < (length_psswd-count_verified):
                    break

                if ((peak_times[j]/self.RATE)-start) < (self.TIMES[count_verified]-self.TIME_THRESHOLD):
                    #Too low time between peak[i] and peak[j] (analyse next peak)
                    j += 1
                elif ((peak_times[j]/self.RATE)-start) < (self.TIMES[count_verified]+self.TIME_THRESHOLD):
                    #peak[j] corresponds to time instant of the count_verified-th character of the password
                    count_verified += 1
                    checked_char_times.append(char_times[j])
                    j += 1
                elif ((peak_times[j]/self.RATE)-start) > (self.TIMES[count_verified]+self.TIME_THRESHOLD):
                    #Too much time between peak[i] and peak[j] (no time correspondence)
                    #Change start peak (start = peak[i+1])
                    break

                if count_verified == length_psswd:
                    #Found time correspondence
                    return True, checked_char_times
        '''

        #No time correspondence found
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
        
        #Find labels related to peaks of all the windows
        keys = self.predict_keys(char_times, folder, option)

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
        
        #Model used to extract features from spectrograms
        model = VGG16(weights='imagenet', include_top=False)
        #Model used for prediction of labels        
        net = nn.NeuralNetwork(option, folder)
        count = 0
        keys = []

        #Analysis of all the windows found
        for list_time in char_times:
            #Evaluation of press peak and hit peaks
            analysis = ef.ExtractFeatures(self.RATE, self.SIGNAL[list_time[0]:list_time[-1]])
            
            if utility.OPTIONS[option]=='touch':
                #Touch feature
                features = analysis.extract(original_signal=self.SIGNAL, index=list_time[0])
                
                if features is None:
                    if self.DEBUG:
                        print(colored(f'{count} ---> ', 'red')+'EMPTY SEQUENCE')
                    
                    count+=1
                    continue

                touch_feature = features['touch']
                hit_feature = features['hit']
                touch_X = touch_feature.fft_signal.reshape((1, 66))
                #Predicted labels are appended in keys list
                keys.append(net.test(touch_X))
                
                if self.DEBUG:
                    print(colored(f'{count} ---> ', 'red')+utility.results_to_string(keys[-1]))
                
                if self.PLOT_OPTION:
                    self.plot_features(count, touch_feature, hit_feature)

            elif utility.OPTIONS[option]=='touch_hit':
                #Touch peak and hit peak
                features = analysis.extract(original_signal=self.SIGNAL, index=list_time[0])
                
                if features is None:
                    if self.DEBUG:
                        print(colored(f'{count} ---> ', 'red')+'EMPTY SEQUENCE')
                    
                    count+=1
                    continue

                touch_feature = features['touch']
                hit_feature = features['hit']            
                touch_X = touch_feature.fft_signal
                hit_X = hit_feature.fft_signal
                touch_hit_X = np.concatenate((touch_X, hit_X)).reshape((1, 132))                
                #Predicted labels are appended in keys list
                keys.append(net.test(touch_hit_X))
                
                if self.DEBUG:
                    print(colored(f'{count} ---> ', 'red')+utility.results_to_string(keys[-1]))
                
                if self.PLOT_OPTION:
                    self.plot_features(count, touch_feature, hit_feature)

            else:
                #Create spectrogram of touch peak and hit peak 
                fig, ax = plt.subplots(1)
                fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
                features = analysis.extract(original_signal=self.SIGNAL, index=list_time[0], spectrum=True)
        
                if features is None:
                    if self.DEBUG:
                        print(colored(f'{count} ---> ', 'red')+'EMPTY SEQUENCE')
                    
                    count+=1
                    continue
                
                img_feature = np.concatenate((self.SIGNAL[features[0]], self.SIGNAL[features[1]]))
                spectrum, freqs, t, img_array = ax.specgram(img_feature, NFFT=len(features[0]), Fs=analysis.fs)
                
                #Save the spectrogram image
                fig.savefig((self.PLOT_FOLDER+self.SPECTRUM_FOLDER+self.CHAR_SPECTRUM_IMG).format(count), dpi=300)
                plt.close(fig)

                #Extract feature using VGG16
                final_features = utility.extract(model, (self.PLOT_FOLDER+self.SPECTRUM_FOLDER+self.CHAR_SPECTRUM_IMG).format(count))
                #Predicted labels are appended in keys list
                keys.append(net.test(final_features))
                
                if self.DEBUG:
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

        #Define the frame characteristics
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
        
        #Save the plot
        fig.savefig((self.PLOT_FOLDER+self.WAVE_FOLDER+self.CHAR_WAVE_PEAKS_IMG).format(count))
        plt.close()

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

        #fig  = plt.figure()
        #fig.tight_layout(pad=3.0)
        #plt.ylabel('Frequency(Hz)')
        #plt.xlabel('Time(s)')

        #Define the frame characteristics
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

        #Save the plot
        s_bottom.specgram(img_feature, NFFT=len(features[0]), Fs=self.RATE)
        fig.savefig((self.PLOT_FOLDER+self.WAVE_FOLDER+self.CHAR_WAVE_SPECTRUM_IMG).format(count))

    def run(self, folder, option, username, threshold):
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
        
        #If deep learning option with spectrograms specified
        if self.DL_OPTION and utility.OPTIONS[option]=='spectrum':
            if not os.path.exists(self.PLOT_FOLDER+self.SPECTRUM_FOLDER):
                #Create subfolder for spectrograms of audio files during the password insertion
                os.mkdir(self.PLOT_FOLDER+self.SPECTRUM_FOLDER)
            else:
                #Delete graphics of audio files already in the subfolder
                files = os.listdir(self.PLOT_FOLDER+self.SPECTRUM_FOLDER)

                for f in files:
                    os.remove(self.PLOT_FOLDER+self.SPECTRUM_FOLDER+f)

        try:
            #Execute AcCAPPCHA
            count_bot_trials = 0
            count_pwd_trials = 0
            
            #Go on until (authentication ok) or (maximum attepmts for wrong 
            #credentials or bot deceted are performed)
            #
            while count_pwd_trials < self.MAX_PWD_TRIALS and \
                  count_bot_trials < self.MAX_BOT_TRIALS:
              
                self.COMPLETED_INSERT = False
                self.noise_evaluation()

                self.TIMES = []
                self.PASSWORD = ''

                #Audio recording thread during the insertion of the password              
                cprint('password:', 'red', end=' ')
                audio_logger = threading.Thread(target=self.audio, args=(folder,option, threshold))
                audio_logger.start()

                #Insertion of the password
                no_end = True
                sleep(1)
                while no_end:
                    no_end = self.password_from_user()
                
                #End of audio recording, after the insertion of the password
                audio_logger.join()

                #Connection to the server
                with SecureElement('127.0.0.1', 8080) as s:
                    #Signature of the message
                    check = s.sign(str(self.VERIFIED))        
                    #Print if human or bot
                    self.bot_human(check)

                    if check:
                        #If humman, send credentials
                        msg = s.credentials(username, self.PASSWORD)
                        
                        #Open the web browser showing the response of the server
                        with open('../dat/html/response.html', 'w') as f:
                            #Store the HTML code
                            f.write(msg)

                            #Store HTML code in web browser
                            import webbrowser
                            webbrowser.open('file://' + os.path.abspath('../dat/html/response.html'))

                        #Analysis of HTML response of the server
                        if not 'Logged in' in msg:
                            #User credentials wrong
                            count_pwd_trials+=1

                            if 'First sign up.' in msg:
                                #Username not in the database
                                cprint(f"The username {username} doesn't exist", 'yellow')
                                
                                if count_pwd_trials != self.MAX_PWD_TRIALS:
                                    cprint(f'\n   Authentication\n{utility.LINE}', 'blue')
                                    username = input(colored('username: ', 'red'))
                                else:
                                    print('')
                
                            elif 'Wrong password.' in msg:
                                #Wrong password for the username (it is in database)
                                cprint(f"The password was wrong", 'yellow')

                        else:
                            #User correctly logged in
                            break

                    else:
                        #Bot detected
                        cprint('Try to stay in a quiet environment', 'yellow')
                        count_bot_trials += 1
    
            if count_pwd_trials == self.MAX_PWD_TRIALS or \
               count_bot_trials == self.MAX_BOT_TRIALS:
                #Block the user 
                #Maximum number of attemps reached for wrong credentials
                #or maximum number of trials for bot user reached               
                moment = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime())

                #Store in block.txt file the current time
                with open(self.BLOCK_FILE, 'w') as f:
                    f.write(moment)
            
            #Return file verification result
            return self.VERIFIED              

        except KeyboardInterrupt:
            #CTRL+C during the execution (Exit from the program)
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
        Obfuscate PASSWORD by replacing each character with a *

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
        i=0
        for x in self.PASSWORD:
            if x == '\b':
                #Remove time instants related to '\b' and the previous character
                self.TIMES=self.TIMES[:i-1]+self.TIMES[i+1:]
                psswd = psswd[:-1]
            else:
                psswd += x
                i+=1

        return psswd

    def bot_human(self, check):
        if check:
            cprint("#################################", 'blue')
            cprint('#############', 'blue', end=' ')
            cprint("HUMAN", 'red', end=' ')
            cprint('#############', 'blue')
            cprint("#################################", 'blue', end='\n\n')
        else:
            cprint("#################################", 'blue')
            cprint('##############', 'blue', end=' ')
            cprint("BOT", 'red', end=' ')
            cprint('##############', 'blue')
            cprint("#################################", 'blue', end='\n\n')

def select_type_feature():
    """
    Select which type of features you want to use

    Returns:
        option (int): Option selected
    """
    
    check = True
    while check:
        try:
            cprint(f'Select which type of features you want to use:\n{utility.LINE}', 'blue')
                
            for i in range(0, len(utility.OPTIONS)):
                cprint(f'{i})', 'yellow', end=' ')
                print(f'{utility.OPTIONS[i]}')

            cprint(f'{utility.LINE}', 'blue')

            option = int(input())
            if option >= 0 and option < len(utility.OPTIONS):
                check = False

        except ValueError:
            cprint('[VALUE ERROR]', 'color', end=' ')
            print('Insert a value of them specified in menu')

    return option

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

    parser.add_argument("-bound", "-b", 
                    dest="bound",
                    help="""It specifies the percentage of relaxation
                            of the time correspondence 
                            (floating point in [0, 1])""")

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

    threshold = 1.0
    if args.bound:
        threshold = float(args.bound)
        
        if threshold < 0.0 or threshold > 1.0:
            threshold = 1.0

    return args.dir, args.time, args.deep, args.plot, args.debug, threshold

def main():
    #Colored print
    colorama.init()
    #Read command line arguments
    folder, time_option, dl_option, plot_option, debug_option, threshold = args_parser()

    #Select tye of the feature for deep learning technique
    option = -1
    
    if dl_option:
        option = select_type_feature()

    try:
        #Initialize AcCAPPCHA
        captcha = AcCAPPCHA(time_option, dl_option, plot_option, debug_option)

        #User not blocked
        cprint(f'   Authentication\n{utility.LINE}', 'blue')
        username = input(colored('username: ', 'red'))
        captcha.run(folder, option, username, threshold)

    except BlockAccessError:
        #User blocked
        cprint("#################################", 'blue')
        cprint('##', 'blue', end=' ')
        cprint("NO ACCESS TO SECURE ELEMENT", 'red', end=' ')
        cprint('##', 'blue')
        cprint("#################################", 'blue', end='\n\n')

if __name__=='__main__':
    main()