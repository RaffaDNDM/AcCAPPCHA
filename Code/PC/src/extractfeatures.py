import numpy as np
from scipy import fftpack as fft
import scipy
import math


class ExtractFeauters:
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


    def __init__(self, fs, signal):
        self.ts, self.time_ms, self.signal = self.signal_adjustment(fs, signal)
        self.fs = fs

    def extract(self):
        push_peak, hit_peak = self.press_peaks()
        return self.FFT_evaluation(push_peak, hit_peak)


    def num_samples(self, seconds):
        return math.floor(seconds/self.ts)


    def press_peaks(self):
        #Find pusk and hit peaks from press peak
        max_point = np.argmax(self.signal)
        #Num samples of STEP_PEAKS
        num_samples_STEP = math.floor(self.STEP_PEAKS / self.ts)
        #press_peak = np.arange(max_point - 1*num_samples_1ms, max_point + 9*num_samples_1ms)
        #plt.plot([max_point*ts,], [signal[max_point],], 'x', color='red')
        #plt.plot(press_peak*ts, signal[press_peak], color='red')
        push_peak_start = max_point - self.START_PUSH_PEAK*num_samples_STEP
        push_peak_end = push_peak_start + self.WIDTH_PUSH_PEAK*num_samples_STEP
        hit_peak_end = push_peak_start + self.WIDTH_PRESS_PEAK*num_samples_STEP
        hit_peak_start = hit_peak_end - self.WIDTH_HIT_PEAK*num_samples_STEP

        push_peak = np.arange(math.floor(push_peak_start), math.ceil(push_peak_end))
        hit_peak = np.arange(math.floor(hit_peak_start), math.ceil(hit_peak_end))

        return push_peak, hit_peak


    def signal_adjustment(self, fs, signal):
        # Extract Raw Audio from Wav File
        l_audio = len(signal.shape)

        if l_audio == 2:
            signal_updated = np.mean(signal, axis=1)

        N = signal.shape[0]

        secs = N / float(fs)
        ts = 1.0/fs #Sampling rate in time (ms)
        time_ms = scipy.arange(0, secs, ts)

        return ts, time_ms, signal_updated


    def FFT_evaluation(self, press_peak, hit_peak):
        peaks = {}

        N_press = len(press_peak)
        f_press = self.fs*np.arange(math.floor(N_press/2))/N_press
        fft_signal_press = np.fft.fft(self.signal[press_peak])[0:int(N_press/2)]/N_press 
        fft_signal_press[1:] = 2*fft_signal_press[1:]
        fft_signal_press = np.abs(fft_signal_press)
        
        
        peaks['press'] = Feature(press_peak,
                                 f_press, 
                                 fft_signal_press)

        N_hit = len(hit_peak)
        f_hit = self.fs*np.arange(math.floor(N_hit/2))/N_hit
        fft_signal_hit = np.fft.fft(self.signal[hit_peak])[0:int(N_hit/2)]/N_hit 
        fft_signal_hit[1:] = 2*fft_signal_hit[1:]
        fft_signal_hit = np.abs(fft_signal_hit)
        
        
        peaks['hit'] = Feature(hit_peak,
                               f_hit, 
                               fft_signal_hit)

        return peaks


class Feature:

    def __init__(self, 
                 peak,
                 freqs,
                 fft_signal):

        self.peak = peak
        self.freqs = freqs
        self.fft_signal = fft_signal