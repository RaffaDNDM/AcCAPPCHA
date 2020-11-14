import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import fftpack as fft
from scipy.io import wavfile as wave
from scipy.signal import find_peaks
import scipy
import math
from termcolor import cprint


class Plot:
	DATA_FOLDER = ''
	EXTENSION = '.png'
	RECURSIVE = False

	def __init__(self, filename, audio_dir, output_dir):
		self.OUTPUT_FOLDER = output_dir
		
		if self.OUTPUT_FOLDER and not os.path.exists(self.OUTPUT_FOLDER):
			os.mkdir(self.OUTPUT_FOLDER)

		if filename:
			if os.path.exists(self.DATA_FOLDER+filename):
				self.wave_files = [filename, ]
			else:
				exit(0)
		elif audio_dir:
			self.DATA_FOLDER = audio_dir
			files = os.listdir(audio_dir)
			count = 0
			
			for f in files:
				if not os.path.isdir(audio_dir+f):
					break

				count = count + 1

			if count==len(files):
				self.RECURSIVE = True
				self.wave_files = {file:[x for x in os.listdir(self.DATA_FOLDER+file) if x.endswith('.wav')] for file in files}
			else:
				print("There aren't only directories in the folder specified by -d option, so the plot will be performed on audio files in it")
				self.wave_files= [x for x in os.listdir(self.DATA_FOLDER) if x.endswith('.wav')]
					

	def plot_waves(self, f, time_ms, signal, freqs, fft_signal, freqs_side, fft_signal_side):
		key = f[:-len('.wav')]
		fig = plt.figure(key)
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

		if self.OUTPUT_FOLDER:
			print(os.path.dirname(self.OUTPUT_FOLDER))
			fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+key+self.EXTENSION)
		else:
			plt.show()


	def plot(self):
		try:
			for f in self.wave_files:
				fs, signal = wave.read(self.DATA_FOLDER+f)
				time_ms, signal = self.signal_adjustment(fs, signal)
				fft_freqs, fft_signal, fft_freqs_side, fft_signal_side = self.FFT_evaluation(time_ms, signal)
				self.plot_waves(f, time_ms, signal, fft_freqs, fft_signal, fft_freqs_side, fft_signal_side)

		except KeyboardInterrupt:
			plt.close()
			exit(0)


	def signal_adjustment(self, fs, signal):
		# Extract Raw Audio from Wav File
		l_audio = len(signal.shape)
		
		if l_audio == 2:
			signal_updated = np.mean(signal, axis=1)

		N = signal.shape[0]

		secs = N / float(fs)
		ts = 1.0/fs #Sampling rate in time (ms)
		time_ms = scipy.arange(0, secs, ts)

		return time_ms, signal_updated


	def FFT_evaluation(self, time_ms, signal):
		fft_signal = fft.fft(signal)
		fft_signal_side = fft_signal[range(len(fft_signal)//2)]
		freqs = fft.fftfreq(len(signal), time_ms[1]-time_ms[0])
		fft_freqs = np.array(freqs)
		freqs_side = freqs[range(len(fft_signal)//2)]
		fft_freqs_side = np.array(freqs_side)

		return fft_freqs, fft_signal, fft_freqs_side, fft_signal_side

	def plot_more_signals(self, subfolder, signals):
		n = int(math.sqrt(len(signals)))
		m = math.ceil(len(signals)/n)

		fig = plt.figure(subfolder)
		
		if m>n:
			x = m
			m = n
			n = x

		cprint(f'[{m},{n}]', 'red', end='\n\n')
		gs = fig.add_gridspec(m, n)
		fig.tight_layout(pad=5.0)

		x = 0
		y = 0
		for (time_ms, signal) in signals:
			print(f'({y},{x})')
			s = fig.add_subplot(gs[y,x])
			s.plot(time_ms, signal, 'b')
			s.tick_params(axis='both', which='major', labelsize=6)
			#max_point = np.argmax(np.abs(signal))
			#s.plot([max_point,], [signal[max_point],], 'x', color='red')

			if self.OUTPUT_FOLDER:
				print(os.path.dirname(self.OUTPUT_FOLDER))
				fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+subfolder+self.EXTENSION)
			else:
				plt.show()
			
			if x == (n-1):
				y = y + 1
				x=0
			else:
				x = x + 1


	def plot_extract(self, subfolder, time_ms, signal):
		fig = plt.figure(subfolder)
		max_point = np.argmax(np.abs(signal))
		plt.plot([max_point,], [signal[max_point],], 'x', color='red')
		plt.show()


	def extract(self):
		try:
			if self.RECURSIVE:			
				for subfolder in self.wave_files.keys():
					subfolder_files = self.wave_files[subfolder]
					signals = []
					print(subfolder)
					for f in subfolder_files:
						fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+f)
						time_ms, signal = self.signal_adjustment(fs, signal)
						signals.append((time_ms,signal))
						#FFT
						#fft_freqs, fft_signal, fft_freqs_side, fft_signal_side = self.FFT_evaluation(time_ms, signal)

					self.plot_more_signals(subfolder, signals)
					
			else:
				for f in self.wave_files:
					fs, signal = wave.read(self.DATA_FOLDER+f)
					time_ms, signal = self.signal_adjustment(fs, signal)
					self.plot_extract(f, time_ms, signal)
					#FFT
					fft_freqs, fft_signal, fft_freqs_side, fft_signal_side = self.FFT_evaluation(time_ms, signal)

		except KeyboardInterrupt:
			plt.close()
			exit(0)