import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile as wave
import math
from termcolor import cprint
import utility
import extractfeatures


class PlotExtract:
	DATA_FOLDER = ''
	EXTENSION_PLOT = '.png'
	RECURSIVE = False
	OUTPUT_FOLDER = None

	'''
	PlotExtract object extracts features and plots
	waves and FFT of acquired audio of key press

	Args:
		is_extract (bool): True if extract option, False if plot option
		filename (str): Path of the wav file that you want to plot
						or from which you want to extract features
		audio_dir (str): Path of the directory with files (or subfolders
						 with files) that you want to plot or from which 
						 you want to extract features
		output_dir (str): Path of the directory where you want to save 
						  the plot extracted features and peaks of the
						  audio files

	Attributes:
		filename (str): Path of the wav file that you want to plot
						or from which you want to extract features
		DATA_FOLDER (str): Path of the directory with files (or subfolders
						   with files) that you want to plot or from which 
						   you want to extract features
		OUTPUT_FOLDER (str): Path of the directory where you want to save 
							 the plot extracted features and peaks of the
						  	 audio files
		wav_files (list): [-p option]
						  [-e option with (-f option) or (-d option but folder
						   with not only subfolders but also wav files)]
						  List containing wav files in the path audio_dir or 
						  wav file at path filename
		wav_files (dict): [-e option with output_dir] 
						  Dictionary with elements composed by:
						  key (str): subfolder_name of audio_dir (key name)
						  value (list): wav files names in subfolder with name
						  				equal to key
	'''
	def __init__(self, filename, audio_dir, output_dir):		
		
		if output_dir:
			self.OUTPUT_FOLDER = utility.uniform_dir_path(output_dir)

		if filename:
			if os.path.exists(self.DATA_FOLDER+filename):
				self.wav_files = [filename, ]
			else:
				cprint('\n[Not existing FILE]', 'blue', end=' ')
				print("The file", end=' ')
				cprint(f'{self.DATA_FOLDER+filename}', 'green', end=' ')
				print("doesn't exist", end='\n\n')
				exit(0)

		elif audio_dir:
			if not(os.path.exists(audio_dir) and os.path.isdir(audio_dir)):
				cprint("\n[Not existing FOLDER]", 'blue', end=' ')
				print("The input directory", end=' ')
				cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
				print("doesn't exist", end='\n\n')
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
				self.wav_files = {file:[x for x in os.listdir(self.DATA_FOLDER+file) if x.endswith('.wav')] for file in files}
			
			else:
				cprint('\n[NOT ONLY DIRECTORIES IN INPUT FOLDER]', 'blue', end=' ')
				print('I try to perform plot on audio files in', end=' ')
				cprint(f'{self.DATA_FOLDER}', 'green')

				self.wav_files= [x for x in os.listdir(self.DATA_FOLDER) if x.endswith('.wav')]
				if not self.wav_files:
					cprint('[EMPTY FOLDER]', 'blue', end=' ')
					print('The directory', end=' ')
					cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
					print("doesn't contain .wav files or directories", end='\n\n')
					exit(0)


	def plot(self, zoom):
		'''
        Plot the waves of audio signals w.r.t.
		selected mode
        '''
		try:
			if self.RECURSIVE:			
				#Plot for each subfolder of DATA_FOLDER
				for subfolder in self.wav_files.keys():
					#Plot together the audio signals of all the wav files
					# inside the subfolder  
					#(a window with audio signal with highlighted peaks)

					#Sort wav files in subfolder by lessicographic order
					subfolder_files = [x for x in self.wav_files[subfolder]]
					subfolder_files.sort()

					if subfolder_files:
						#Collect the analysis objects, one for
						# each wav files in the subfolder
						signals = []
						cprint('\n'+subfolder, 'red')
						
						for f in subfolder_files:
							#Reading audio file
							fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+f)
							#Analysis of audio signal
							analysis = extractfeatures.ExtractFeatures(fs, signal)
							#Append the analysis object and filename to the lists
							signals.append((f, analysis))
							
						#Plot of features in OUTPUT_FOLDER
						#(one plot for each element of signals)
						#(one plot for each subfolder)
						self.plot_many_waves(subfolder, signals, zoom)

					else:
						cprint('[EMPTY FOLDER]', 'blue', end=' ')
						print("No '.wav' files inside", end=' ')
						cprint(f'{self.DATA_FOLDER+subfolder}', 'green')
			else:
				#Plot features of each file in wav_files separately
				#(a window with audio signal with highlighted peaks
				#    and FFT transform of press and hit peaks)
				for filename in self.wav_files:
					#Reading audio file
					fs, signal = wave.read(self.DATA_FOLDER+filename)
					#Analysis of audio signal
					analysis = extractfeatures.ExtractFeatures(fs, signal)
					#Plot of features
					self.plot_single_wave(filename, analysis, zoom)
					
		except KeyboardInterrupt:
			#Terminate the program (detected CTRL+C)
			plt.close()
			exit(0)


	def plot_many_waves(self, subfolder, signals, zoom):
		'''
        Plot all the waves of audio signals 
		in subfolder with highlighted peaks
        
        Args:
			subfolder (str): name of subfolder
			signals (list): List of tuples, with each one composed by:
							f: name of file
							analysis: ExtractFeatures related to the
									  wav file with name f
        '''
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
			press_peak, hit_peak = analysis.press_peaks()
			ts = analysis.ts
			signal = analysis.signal
	
			print('{:6s} ----> '.format(filename), end='')
			cprint(f'({y},{x})', 'cyan')
			s = fig.add_subplot(gs[y,x])
			s.tick_params(axis='both', which='major', labelsize=6)

			time_ms = analysis.time_ms

			#Time in ms in the behaviour of press peak (10 ms before and 10 ms after the detected one)
			if zoom:
				first_time = press_peak[0] - analysis.num_samples(1e-2)
				last_time = hit_peak[-1] + analysis.num_samples(1e-2)
				time_ms = np.arange(first_time, last_time)

			s.plot(time_ms*ts, signal[time_ms], color='blue')
			s.plot(press_peak*ts, signal[press_peak], color='red')
			s.plot(hit_peak*ts, signal[hit_peak], color='green')
			
			if x == (n-1):
				y = y + 1
				x=0
			else:
				x = x + 1

		if self.OUTPUT_FOLDER:
			fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+subfolder+self.EXTENSION_PLOT)
		else:
			plt.show()



	def plot_single_wave(self, filename, analysis, zoom):
		'''
        Plot the audio signal with name filename with
		highlighted peaks and FFT of push and hit peaks
        
        Args:
			filename (str): name of wav file
			analysis (ExtractFeatures): ExtractFeatures related to the
  	    								wav file with name filename
        '''
		#Evaluation of press peak and hit peaks
		features = analysis.extract()
		press_feature = features['press']
		hit_feature = features['hit']

		time_ms = analysis.time_ms
		#Time in ms in the behaviour of press peak (10 ms before and 10 ms after the detected one)
		if zoom:
			first_time = press_feature.peak[0] - analysis.num_samples(1e-2)
			last_time = hit_feature.peak[-1] + analysis.num_samples(1e-2)
			time_ms = np.arange(first_time, last_time)

		#Time sample step
		ts = analysis.ts
		#Signal rearranged
		signal = analysis.signal

		#Initialize the figure
		key = filename[:-len('.wav')]
		fig = plt.figure(key)
		gs = fig.add_gridspec(2, 2)
		s_top = fig.add_subplot(gs[0, :])
		s1 = fig.add_subplot(gs[1,0])
		s2 = fig.add_subplot(gs[1,1])
		fig.tight_layout(pad=3.0)

		#Plot of press peak and hit peak with the signal
		s_top.plot(time_ms*ts, signal[time_ms], color='blue')
		s_top.plot(press_feature.peak*ts, signal[press_feature.peak], color='red')
		s_top.plot(hit_feature.peak*ts, signal[hit_feature.peak], color='green')
		s_top.set_title('Amplitude')
		s_top.set_xlabel('Time(ms)')
		s_top.tick_params(axis='both', which='major', labelsize=6)

		#Plot FFT double-sided transform of PRESS peak
		s1.plot(press_feature.freqs, press_feature.fft_signal, color='red')
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

		if self.OUTPUT_FOLDER:
			fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+filename+self.EXTENSION_PLOT)
		else:
			plt.show()
		

	def extract(self):
		try:
			if self.RECURSIVE:
				#Extraction for each subfolder of DATA_FOLDER
				for subfolder in self.wav_files.keys():
					#Extract features from the audio signals of all the wav files
					#inside the subfolder  
					
					#Sort wav files in subfolder by increasing num order
					subfolder_files_nums = [int(x[:-len('.wav')]) for x in self.wav_files[subfolder]]
					subfolder_files_nums.sort()					
					subfolder_files = [str(x)+'.wav' for x in subfolder_files_nums]

					if subfolder_files:
						#Collect the analysis objects, one for
						# each wav files in the subfolder
						signals = []
						cprint('\n'+subfolder, 'red')
						
						for f in subfolder_files:
							#Reading audio file
							fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+f)
							#Analysis of audio signal
							analysis = extractfeatures.ExtractFeatures(fs, signal)
							#Append the analysis object and filename to the lists
							signals.append(analysis)
							
						#Plot of features in OUTPUT_FOLDER
						#(one plot for each element of signals)
						#(one plot for each subfolder)
						self.store_many_features(subfolder, signals)

					else:
						cprint('[Folder EMPTY]', 'blue', end=' ')
						print("No '.wav' files inside it")

			else:
				#Plot features of each file in wav_files separately
				#(a window with audio signal with highlighted peaks
				#    and FFT transform of press and hit peaks)
				for filename in self.wav_files:
					#Reading audio file
					fs, signal = wave.read(self.DATA_FOLDER+filename)
					#Analysis of audio signal
					analysis = extractfeatures.ExtractFeatures(fs, signal)
					#Plot of features
					self.store_single_feature(filename, analysis)
					
		except KeyboardInterrupt:
			#Terminate the program (detected CTRL+C)
			plt.close()
			exit(0)


	def store_many_features(self, subfolder, signals):
		'''
        Plot all the waves of audio signals 
		in subfolder with highlighted peaks
        
        Args:
			subfolder (str): name of subfolder
			signals (list): List of tuples, with each one composed by:
							f: name of file
							analysis: ExtractFeatures related to the
									  wav file with name f
        
		'''
		n = int(math.sqrt(len(signals)))
		m = math.ceil(len(signals)/n)

		fig = plt.figure(subfolder)
		
		if m>n:
			m, n = utility.swap(m, n)

		cprint(f'[{m},{n}]', 'red')
		gs = fig.add_gridspec(m, n)
		fig.tight_layout(pad=5.0)

		x = 0
		y = 0
		for analysis in signals:
			#Evaluation of press peak and hit peaks
			features = analysis.extract()
			press_feature = features['press']
			hit_feature = features['hit']

			#Sequence of temporal instances
			time_ms = analysis.time_ms
			#Time sample step
			ts = analysis.ts
			#Signal rearranged
			signal = analysis.signal
			

	def store_single_feature(self, filename, analysis):
		'''
        Plot the audio signal with name filename with
		highlighted peaks and FFT of push and hit peaks
        
        Args:
			filename (str): name of wav file
			analysis (ExtractFeatures): ExtractFeatures related to the
  	    								wav file with name filename
        '''
		#Evaluation of press peak and hit peaks
		features = analysis.extract()
		press_feature = features['press']
		hit_feature = features['hit']

		#Sequence of temporal instances
		time_ms = analysis.time_ms
		#Time sample step
		ts = analysis.ts
		#Signal rearranged
		signal = analysis.signal

		#return press_feature.fft_signal, hit_feature.fft_signal