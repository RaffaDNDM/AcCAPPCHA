import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile as wave
import math
from termcolor import cprint
import utility
import extractfeatures


class Plot:
	DATA_FOLDER = ''
	EXTENSION = '.png'
	RECURSIVE = False


	def __init__(self, is_extract, filename, audio_dir, output_dir):
		if output_dir and (not os.path.exists(output_dir) or not os.path.isdir(output_dir)):
			os.mkdir(output_dir)
		
		if is_extract and output_dir:
			self.OUTPUT_FOLDER = utility.uniform_dir_path(output_dir)

		if filename:
			if os.path.exists(self.DATA_FOLDER+filename):
				self.wave_files = [filename, ]
			else:
				exit(0)
		elif audio_dir:
			if not(os.path.exists(audio_dir) and os.path.isdir(audio_dir)):
				cprint("[ERROR]", 'red', end=' ')
				print("the input directory doesn't exist")
				exit(0)

			self.DATA_FOLDER = utility.uniform_dir_path(audio_dir)
			files = os.listdir(self.DATA_FOLDER)
			files.sort()
			count = 0
			
			#Check if there is at least a file in audio_dir
			#In this case, doesn't apply recursive analysis on all the subfolders
			#But analyses only the wav files in the folder
			for f in files:
				if not os.path.isdir(self.DATA_FOLDER+f):
					break

				count = count + 1

			if count==len(files):
				self.RECURSIVE = True
				self.wave_files = {file:[x for x in os.listdir(self.DATA_FOLDER+file) if x.endswith('.wav')] for file in files}
			else:
				print("There aren't only directories in the folder specified by -d option, so the plot will be performed on audio files in it")
				self.wave_files= [x for x in os.listdir(self.DATA_FOLDER) if x.endswith('.wav')]
					

	def plot_waves(self, filename, analysis):
		ts = analysis.get_time_rate()
		freqs, fft_freqs, fft_signal, freqs_side, fft_freqs_side, fft_signal_side = analysis.FFT_evaluation()

		key = filename[:-len('.wav')]
		fig = plt.figure(key)
		gs = fig.add_gridspec(2, 2)
		s_top = fig.add_subplot(gs[0, :])
		s1 = fig.add_subplot(gs[1,0])
		s2 = fig.add_subplot(gs[1,1])
		fig.tight_layout(pad=3.0)
		s_top.plot(analysis.get_time(), analysis.get_signal(), 'b')
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
			for filename in self.wave_files:
				fs, signal = wave.read(self.DATA_FOLDER+filename)
				analysis = extractfeatures.ExtractFeauters(fs, signal)
				self.plot_waves(filename, analysis)

		except KeyboardInterrupt:
			plt.close()
			exit(0)


	def plot_extract_many(self, subfolder, signals):
		n = int(math.sqrt(len(signals)))
		m = math.ceil(len(signals)/n)

		fig = plt.figure(subfolder)
		
		if m>n:
			x = m
			m = n
			n = x


		cprint(f'[{m},{n}]', 'red')
		gs = fig.add_gridspec(m, n)
		fig.tight_layout(pad=5.0)

		x = 0
		y = 0
		for (filename, analysis) in signals:
			push_peak, hit_peak = analysis.press_peaks()
			ts = analysis.get_time_rate()
			signal = analysis.get_signal()
	
			print('{:6s} ----> '.format(filename), end='')
			cprint(f'({y},{x})', 'cyan')
			s = fig.add_subplot(gs[y,x])
			s.tick_params(axis='both', which='major', labelsize=6)
			s.plot(analysis.get_time(), analysis.get_signal(), color='blue')
			s.plot(push_peak*ts, signal[push_peak], color='red')
			s.plot(hit_peak*ts, signal[hit_peak], color='green')

			if self.OUTPUT_FOLDER:
				fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+subfolder+self.EXTENSION)
			else:
				plt.show()
			
			if x == (n-1):
				y = y + 1
				x=0
			else:
				x = x + 1


	def plot_extract_single(self, filename, analysis):
		#Evaluation of push peak and hit peak
		push_peak, hit_peak = analysis.press_peaks()
		#Time sample step
		ts = analysis.get_time_rate()
		signal = analysis.get_signal()

		#Plot of push peak and hit peak
		fig = plt.figure(filename)
		plt.plot(analysis.get_time(), signal)
		plt.plot(push_peak*ts, signal[push_peak], color='red')
		plt.plot(hit_peak*ts, signal[hit_peak], color='green')
		plt.show()


	def plot_extract(self):
		try:
			if self.RECURSIVE:			
				for subfolder in self.wave_files.keys():
					subfolder_nums_ordered = [int(x[:-len('.wav')]) for x in self.wave_files[subfolder]]
					subfolder_nums_ordered.sort()
					subfolder_files = [str(x)+'.wav' for x in subfolder_nums_ordered]
					signals = []
					cprint('\n'+subfolder, 'red')
					for f in subfolder_files:
						fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+f)
						analysis = extractfeatures.ExtractFeauters(fs, signal)
						signals.append((f, analysis))
						#FFT
						#fft_freqs, fft_signal, fft_freqs_side, fft_signal_side = self.FFT_evaluation(time_ms, signal)

					self.plot_extract_many(subfolder, signals)
					
			else:
				for filename in self.wave_files:
					fs, signal = wave.read(self.DATA_FOLDER+filename)
					analysis = extractfeatures.ExtractFeauters(fs, signal)
					self.plot_extract_single(filename, analysis)
					#FFT
					#fft_freqs, fft_signal, fft_freqs_side, fft_signal_side = analysis.FFT_evaluation()

		except KeyboardInterrupt:
			plt.close()
			exit(0)