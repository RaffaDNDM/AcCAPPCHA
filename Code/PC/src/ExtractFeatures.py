import numpy as np
import math
import utility

class ExtractFeatures:
    #Time step = 0.5 ms
    STEP_PEAKS = 5e-4
    #Width of press peak = 20 STEP_PEAKS (10 ms)
    WIDTH_PRESS_PEAK = 20 #TRY 30
    #Start of touch peak = 3 STEP_PEAKS (1.5 ms) before touch peak (max of wave)
    START_TOUCH_PEAK = 3
    #Width of touch peak = 3 STEP_PEAKS (3 ms)
    WIDTH_TOUCH_PEAK = 6
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
                                 touch peak and the end instance of hit peak)
        START_TOUCH_PEAK (int): start instance of touch peak
                               in terms of number of STEP_PEAKS, in time 
                               space, before the touch peak (max of audio)
        WIDTH_TOUCH_PEAK (int): width of touch peak
                               in terms of number of STEP_PEAKS, in time 
                               space, from the start index of the touch peak
                               (max of the values of audio signal samples)
        WIDTH_HIT_PEAK (int): width of hit peak
                               in terms of number of STEP_PEAKS, in time 
                               space, from the start index of the hit peak
    '''
    def __init__(self, fs, signal):
        self.ts, self.time_ms, self.signal = utility.signal_adjustment(fs, signal)
        self.fs = fs


    def extract(self, original_signal= None, index= None):
        '''
        Extract the feature from the signal
        
        Args:
        
        Raises:

        Returns:
            features (dict): dictionary of Feature objects 
                             related to touch and hit peaks
        '''
        #Find peaks (hit and touch) of press peaks
        touch_peak, hit_peak = self.press_peaks(index)
        #FFT evaluation from press peaks
        if original_signal is None:
            return self.FFT_evaluation(self.signal, touch_peak, hit_peak)        
        else:
            return self.FFT_evaluation(original_signal, touch_peak, hit_peak)

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
        return utility.num_samples(self.fs, seconds)


    def press_peaks(self, index= None):
        '''
        Extract touch peak and hit peak from the signal
        
        Args:
        
        Raises:

        Returns:
            touch_peak(np.array): array of indices of samples of
                                 signal that define the touch peak
            hit_peak(np.array): array of indices of samples of
                                signal that define the hit peak
        '''
        #Find touch peak (max of the wave values)
        max_point = np.argmax(self.signal)
        
        if index:
            max_point += index

        #Num of instances inside STEP_PEAKS
        num_samples_STEP = math.floor(self.STEP_PEAKS / self.ts)
        #Start and end indices of samples in signal for touch and hit peaks
        touch_peak_start = max_point - self.START_TOUCH_PEAK*num_samples_STEP
        touch_peak_end = touch_peak_start + self.WIDTH_TOUCH_PEAK*num_samples_STEP
        hit_peak_end = touch_peak_start + self.WIDTH_PRESS_PEAK*num_samples_STEP
        hit_peak_start = hit_peak_end - self.WIDTH_HIT_PEAK*num_samples_STEP
        #Indices of samples in signal for hit and touch peaks
        touch_peak = np.arange(math.floor(touch_peak_start), math.ceil(touch_peak_end))
        hit_peak = np.arange(math.floor(hit_peak_start), math.ceil(hit_peak_end))

        return touch_peak, hit_peak


    def FFT_evaluation(self, original_signal, touch_peak, hit_peak):
        '''
        FFT computaion on press peak and hit peak
        of the audio signal 
        
        Args:
            touch_peak(np.array): array of indices of samples of
                                 signal that define the touch peak
            hit_peak(np.array): array of indices of samples of
                                signal that define the hit peak

        Raises:

        Returns:
            features (dict): dictionary of Feature objects 
                             related to touch and hit peaks
        '''
        peaks = {}

        #TOUCH PEAK
        #Number of samples in the peak
        N_touch = len(touch_peak)
        #Values of frequency axis for FFT transform 
        f_touch = self.fs*np.arange(math.floor(N_touch/2))/N_touch
        #FFT transform
        fft_signal_touch = np.fft.fft(original_signal[touch_peak])[0:int(N_touch/2)]/N_touch 
        #Single-side FFT transform
        fft_signal_touch[1:] = 2*fft_signal_touch[1:]
        #Real values (removing complex part)
        fft_signal_touch = np.abs(fft_signal_touch)
        #Normalize fft
        fft_signal_touch = fft_signal_touch / np.linalg.norm(fft_signal_touch)
        #Store indices of peak samples, FFT coefficents and related frequencies                
        peaks['touch'] = Feature(touch_peak,
                                 f_touch, 
                                 fft_signal_touch)

        #HIT PEAK
        #Number of samples in the peak
        N_hit = len(hit_peak)
        #Values of frequency axis for FFT transform 
        f_hit = self.fs*np.arange(math.floor(N_hit/2))/N_hit
        #FFT transform
        fft_signal_hit = np.fft.fft(original_signal[hit_peak])[0:int(N_hit/2)]/N_hit 
        #Single-side FFT transform
        fft_signal_hit[1:] = 2*fft_signal_hit[1:]
        #Real values (removing complex part)
        fft_signal_hit = np.abs(fft_signal_hit)
        #Normalize fft
        fft_signal_hit = fft_signal_hit / np.linalg.norm(fft_signal_hit)
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