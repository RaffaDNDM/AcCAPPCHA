import numpy as np
import math

class ExtractFeatures:
    #Time step = 0.5 ms
    STEP_PEAKS = 5e-4
    #Width of press peak = 20 STEP_PEAKS (10 ms)
    WIDTH_PRESS_PEAK = 20 #TRY 30
    #Start of push peak = 3 STEP_PEAKS (1.5 ms) before push peak (max of wave)
    START_PUSH_PEAK = 3
    #Width of push peak = 3 STEP_PEAKS (3 ms)
    WIDTH_PUSH_PEAK = 6
    #Width of hit peak = 3 STEP_PEAKS (3 ms)
    WIDTH_HIT_PEAK = 6

    '''
    ExtractFeauteres object contains information about
    the signal and the related feature

    Args:
        fs (float): Sample rate
        signal (np.array): Sequence of audio samples, obtained 
                           reading the audio file

    Attributes:
        ts (float): Sampling time step
        time_ms (np.array): Array with float time instances of each 
                            sample in signal
        signal (np.array): Array with audio values of each audio sample
        STEP_PEAKS (float): number of seconds that will be used as 
                            unit of measure to define range of peaks
        WIDTH_PRESS_PEAK (int): width of press peak 
                                in terms of number of STEP_PEAKS, in time
                                space
                                (distance between the start instance of
                                 push peak and the end instance of hit peak)
        START_PUSH_PEAK (int): start instance of push peak
                               in terms of number of STEP_PEAKS, in time 
                               space, before the push peak (max of audio)
        WIDTH_PUSH_PEAK (int): width of push peak
                               in terms of number of STEP_PEAKS, in time 
                               space, from the start index of the push peak
                               (max of the values of audio signal samples)
        WIDTH_PUSH_PEAK (int): width of push peak
                               in terms of number of STEP_PEAKS, in time 
                               space, from the start index of the hit peak
    '''
    def __init__(self, fs, signal):
        self.ts, self.time_ms, self.signal = self.signal_adjustment(fs, signal)
        self.fs = fs


    def extract(self):
        '''
        Extract the feature from the signal
        
        Args:
        
        Raises:

        Returns:
            features (dict): dictionary of Feature objects 
                             related to push and hit peaks
        '''
        #Find peaks (hit and push) of press peaks
        push_peak, hit_peak = self.press_peaks()
        #FFT evaluation from press peaks
        return self.FFT_evaluation(push_peak, hit_peak)
        

    def num_samples(self, seconds):
        '''
        Extract the feature from the signal
        
        Args:
            seconds (float): seconds to be converted in the
                             number of instances, looking to
                             base sampling step in time (ts)

        Raises:

        Returns:
            features: number of time instances related to ts
        
        '''
        return int(seconds/self.ts)


    def press_peaks(self):
        '''
        Extract push peak and hit peak from the signal
        
        Args:
        
        Raises:

        Returns:
            push_peak(np.array): array of indices of samples of
                                 signal that define the push peak
            hit_peak(np.array): array of indices of samples of
                                signal that define the hit peak
        '''
        #Find push peak (max of the wave values)
        max_point = np.argmax(self.signal)
        #Num of instances inside STEP_PEAKS
        num_samples_STEP = math.floor(self.STEP_PEAKS / self.ts)
        #Start and end indices of samples in signal for push and hit peaks
        push_peak_start = max_point - self.START_PUSH_PEAK*num_samples_STEP
        push_peak_end = push_peak_start + self.WIDTH_PUSH_PEAK*num_samples_STEP
        hit_peak_end = push_peak_start + self.WIDTH_PRESS_PEAK*num_samples_STEP
        hit_peak_start = hit_peak_end - self.WIDTH_HIT_PEAK*num_samples_STEP
        #Indices of samples in signal for hit and push peaks
        push_peak = np.arange(math.floor(push_peak_start), math.ceil(push_peak_end))
        hit_peak = np.arange(math.floor(hit_peak_start), math.ceil(hit_peak_end))

        return push_peak, hit_peak


    def signal_adjustment(self, fs, signal):
        '''
        Analyse number of channels, compute mean of signal for
        2-channels signals and other useful information
        
        Args:
            fs (float): sampling frequency of signal
            signal (np.array): signal to be analysed

        Raises:

        Returns:
            ts (float): sampling step in time
            time_ms (np.array): sequence of instances of
                                each sample of the signal
            signal (np.array): signal analysed looking
                               to the number of channels
        '''
        #Duration of audio by looking to its number of samples
        N = signal.shape[0]
        secs = N / float(fs)
        #Computation of time instances for the audio
        ts = 1.0/fs
        time_ms = np.arange(0, N)
        #If num of channels = 2, do mean of the signal
        l_audio = len(signal.shape)

        if l_audio == 2:
            signal = np.mean(signal, axis=1)

        return ts, time_ms, signal


    def FFT_evaluation(self, press_peak, hit_peak):
        '''
        FFT computaion on press peak and hit peak
        of the audio signal 
        
        Args:
            push_peak(np.array): array of indices of samples of
                                 signal that define the push peak
            hit_peak(np.array): array of indices of samples of
                                signal that define the hit peak

        Raises:

        Returns:
            features (dict): dictionary of Feature objects 
                             related to push and hit peaks
        '''
        peaks = {}

        #PUSH PEAK
        #Number of samples in the peak
        N_press = len(press_peak)
        #Values of frequency axis for FFT transform 
        f_press = self.fs*np.arange(math.floor(N_press/2))/N_press
        #FFT transform
        fft_signal_press = np.fft.fft(self.signal[press_peak])[0:int(N_press/2)]/N_press 
        #Single-side FFT transform
        fft_signal_press[1:] = 2*fft_signal_press[1:]
        #Real values (removing complex part)
        fft_signal_press = np.abs(fft_signal_press)
        #Normalize fft
        fft_signal_press = fft_signal_press / np.linalg.norm(fft_signal_press)
        #Store indices of peak samples, FFT coefficents and related frequencies                
        peaks['press'] = Feature(press_peak,
                                 f_press, 
                                 fft_signal_press)

        #HIT PEAK
        #Number of samples in the peak
        N_hit = len(hit_peak)
        #Values of frequency axis for FFT transform 
        f_hit = self.fs*np.arange(math.floor(N_hit/2))/N_hit
        #FFT transform
        fft_signal_hit = np.fft.fft(self.signal[hit_peak])[0:int(N_hit/2)]/N_hit 
        #Single-side FFT transform
        fft_signal_hit[1:] = 2*fft_signal_hit[1:]
        #Real values (removing complex part)
        fft_signal_hit = np.abs(fft_signal_hit)
        #Normalize fft
        fft_signal_hit = fft_signal_press / np.linalg.norm(fft_signal_press)       
        #Store indices of peak samples, FFT coefficents and related frequencies
        peaks['hit'] = Feature(hit_peak,
                               f_hit, 
                               fft_signal_hit)

        return peaks


class Feature:
    '''
    Feauture object contains information the sequence
    of indices of the samples related to peak, sequence
    of frequencies with their related FFT value

    Args:
        peak (np.array): Array of indices of samples of
                         signal that define the peak
        freqs (np.array): Sequence of frequencies related
                          to FFT coefficients in fft_signal
        fft_signal (np.array): Sequence of FFT coefficients

    Attributes:
        peak (np.array): Array of indices of samples of
                         signal that define the peak
        freqs (np.array): Sequence of frequencies related
                          to FFT coefficients in fft_signal
        fft_signal (np.array): Sequence of FFT coefficients
    '''
    def __init__(self, 
                 peak,
                 freqs,
                 fft_signal):

        self.peak = peak
        self.freqs = freqs
        self.fft_signal = fft_signal