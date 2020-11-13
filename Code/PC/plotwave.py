import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import fftpack as fft
from scipy.io import wavfile as wave
import scipy

class Plot:
	DATA_FOLDER = 'data/'

	def __init__(self, filename):
		if filename:
			if os.path.exists(self.DATA_FOLDER+filename):
				self.wave_files = [filename, ]
			else:
				exit(0)
		else:
			self.wave_files = [x for x in os.listdir(self.DATA_FOLDER) if x.endswith('.wav')]

	
	def plot_waves(self, f, time_ms, signal, freqs, fft_signal, freqs_side, fft_signal_side):
		fig = plt.figure(f[:-len('.wav')])
		gs = fig.add_gridspec(2, 2)
		s_top = fig.add_subplot(gs[0, :])
		s1 = fig.add_subplot(gs[1,0])
		s2 = fig.add_subplot(gs[1,1])
		fig.tight_layout(pad=3.0)
		s_top.plot(time_ms, signal, 'b')
		s_top.set_title('Amplitude')
		s_top.set_xlabel('Time(ms)')
		s_top.tick_params(axis='both', which='major', labelsize=6)
		s1.plot(freqs, fft_signal, 'g')
		s2.plot(freqs_side, fft_signal_side, 'r')
		s1.set_xlabel('Frequency (Hz)')
		s1.set_ylabel('Double-sided FFT')
		s1.tick_params(axis='both', which='major', labelsize=6)
		s2.set_xlabel('Frequency(Hz)')
		s2.set_ylabel('Single-sided FFT')
		s2.tick_params(axis='both', which='major', labelsize=6)
		plt.show()


	def plot(self):
		try:
			for f in self.wave_files:
				fs, signal = wave.read(self.DATA_FOLDER+f)

				# Extract Raw Audio from Wav File
				l_audio = len(signal.shape)
				
				if l_audio == 2:
					signal = np.mean(signal, axis=1)

				N = signal.shape[0]

				secs = N / float(fs)
				ts = 1.0/fs #Sampling rate in time (ms)
				time_ms = scipy.arange(0, secs, ts)

				fft_signal = fft.fft(signal)
				fft_signal_side = fft_signal[range(len(fft_signal)//2)]
				freqs = fft.fftfreq(len(signal), time_ms[1]-time_ms[0])
				fft_freqs = np.array(freqs)
				freqs_side = freqs[range(len(fft_signal)//2)]
				fft_freqs_side = np.array(freqs_side)        

				self.plot_waves(f, time_ms, signal, fft_freqs, fft_signal, fft_freqs_side, fft_signal_side)

		except KeyboardInterrupt:
			plt.close()
			exit(0)
