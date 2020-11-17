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

    def get_time(self):
        return self.time_ms


    def get_signal(self):
        return self.signal


    def get_time_rate(self):
        return self.ts


    def press_peaks(self):
        #Find pusk and hit peaks from press peak
        max_point = np.argmax(self.signal)
        #Num samples of STEP_PEAKS
        num_samples_STEP = math.floor(self.STEP_PEAKS / self.ts)
        #press_peak = np.arange(max_point - 1*num_samples_1ms, max_point + 9*num_samples_1ms)
        #plt.plot([max_point*ts,], [signal[max_point],], 'x', color='red')
        #plt.plot(press_peak*ts, signal[press_peak], color='red')
        push_peak_start = max_point - (self.START_PUSH_PEAK*self.STEP_PEAKS)*num_samples_STEP
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


    def FFT_evaluation(self):
        fft_signal = fft.fft(self.signal)
        fft_signal_side = fft_signal[range(len(fft_signal)//2)]
        freqs = fft.fftfreq(len(self.signal), self.time_ms[1]-self.time_ms[0])
        fft_freqs = np.array(freqs)
        freqs_side = freqs[range(len(fft_signal)//2)]
        fft_freqs_side = np.array(freqs_side)

        return freqs, fft_freqs, fft_signal, freqs_side, fft_freqs_side, fft_signal_side