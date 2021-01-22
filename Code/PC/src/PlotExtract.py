import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile as wave
from scipy import signal
import math
from termcolor import cprint
import utility
import ExtractFeatures as ef
from csv import writer
from PIL import Image
import librosa
import progressbar

class PlotExtract:
	LINE = '_______________________________'
	DATA_FOLDER = ''
	EXTENSION_PLOT = '.png'
	RECURSIVE = False
	DEFAULT_OUTPUT = '../dat/'
	OUTPUT_FOLDER = DEFAULT_OUTPUT
	OUTPUT_CSV_TRAINING = 'dataset.csv'
	OUTPUT_CSV_DICT_LABEL = 'label_dict.csv'
	WIDGETS = [progressbar.Bar('=', '[', ']'),
				' ',
			   progressbar.Percentage()]

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
		
		DEFAULT_OUTPUT = Default output folder

		FEATURE_SIZE (int): Size of a feature

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
	
		RECURSIVE (bool): If wav_files is a dict

		LINE (str): Line string for print purpose
		
		DATA_FOLDER (str): Folder containing audio files or subfolders
						   with audio files inside them
		
		EXTENSION_PLOT (str): Format for plot images [-p option specified]
		
		OUTPUT_CSV_TRAINING (str): Name of csv file with extracted features + labels
		
		OUTPUT_CSV_DICT_LABEL = Name of csv file with labels and related indices

		WIDGETS (list): List of progress bar widgets, used to show evolution
						of extraction of features
	'''

	def __init__(self, filename, audio_dir, output_dir):		
		#Define output folder if the specified one exists
		#Otherwise use the default one
		if output_dir and os.path.isdir(output_dir):
			self.OUTPUT_FOLDER = utility.uniform_dir_path(output_dir)

		if filename:
			#If specified single input file
			if os.path.exists(self.DATA_FOLDER+filename):
				#List of audio files to be extracted/plot (with only one file)
				self.wav_files = [filename, ]
			else:
				cprint('\n[Not existing FILE]', 'blue', end=' ')
				print("The file", end=' ')
				cprint(f'{self.DATA_FOLDER+filename}', 'green', end=' ')
				print("doesn't exist", end='\n\n')
				exit(0)

		elif audio_dir:
			#If specified input directory
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

				count += 1

			if count==len(files):
				#There are subfolders in input directory
				self.RECURSIVE = True
				#Perform extraction/plot on all the wav files in each subfolder 
				self.wav_files = {file:[x for x in os.listdir(self.DATA_FOLDER+file) if x.endswith('.wav')] for file in files}
			
			else:
				#There aren't subfolders in input directory
				cprint('\n[NOT ONLY DIRECTORIES IN INPUT FOLDER]', 'blue', end=' ')
				print('I try to perform plot on audio files in', end=' ')
				cprint(f'{self.DATA_FOLDER}', 'green')

				#Perform extraction/plot on wav files in input dir
				self.wav_files= [x for x in os.listdir(self.DATA_FOLDER) if x.endswith('.wav')]
				
				if not self.wav_files:
					cprint('[EMPTY FOLDER]', 'blue', end=' ')
					print('The directory', end=' ')
					cprint(f'{self.DATA_FOLDER}', 'green', end=' ')
					print("doesn't contain .wav files or directories", end='\n\n')
					exit(0)

	def plot(self, zoom):
		'''
        Plot the waves of audio signals w.r.t. selected mode

		Args:
			zoom (bool): True if you want to plot zoomed version of
						 press peak, False otherwise
        '''
		bar = progressbar.ProgressBar(maxval=20,
									  widgets= self.WIDGETS)

		try:
			if self.RECURSIVE:
				count=0
				#Plot for each subfolder of DATA_FOLDER
				for subfolder in self.wav_files.keys():
					#Plot together the audio signals of all the wav files
					# inside the subfolder  
					#(a window with audio signal with highlighted peaks)
					cprint('\n'+subfolder, 'red', end='\n\n')
					bar.start()
					#Sort wav files in subfolder by lessicographic order
					subfolder_files = [x for x in self.wav_files[subfolder]]
					subfolder_files.sort()

					if subfolder_files:
						#Collect the analysis objects, one for
						# each wav files in the subfolder
						signals = []

						for f in subfolder_files:
							#Reading audio file
							fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+f)
							#Analysis of audio signal
							analysis = ef.ExtractFeatures(fs, signal)
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

					count += 1
					bar.update((count/len(self.wav_files.keys()))*20)

				bar.finish()

			else:
				#Plot features of each file in wav_files separately
				#(a window with audio signal with highlighted peaks
				#    and FFT transform of press and hit peaks)
				count = 0
				cprint(f'\n{os.path.basename(self.DATA_FOLDER[:-1])}', 'red')
				bar.start()
				
				for filename in self.wav_files:
					#Reading audio file
					fs, signal = wave.read(self.DATA_FOLDER+filename)
					#Analysis of audio signal
					analysis = ef.ExtractFeatures(fs, signal)
					#Plot of features
					self.plot_single_wave(filename, analysis, zoom)
					count += 1
					bar.update((count/len(self.wav_files))*20)

				bar.finish()

		except KeyboardInterrupt:
			#Terminate the program (detected CTRL+C)
			plt.close()
			exit(0)

		bar.finish()

	def plot_many_waves(self, subfolder, signals, zoom):
		'''
        Plot all the waves of audio signals 
		in subfolder with highlighted peaks
        
        Args:
			subfolder (str): Name of subfolder

			signals (list): List of tuples, with each one composed by:
							f: name of file
							analysis: ExtractFeatures related to the
									  wav file with name f

			zoom (bool): True if you want to plot zoomed version of
						 press peak, False otherwise
        '''
		#Size of the grid in the frame
		n = int(math.sqrt(len(signals)))
		m = math.ceil(len(signals)/n)

		#Creation of the frame
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
			#Compute touch and hit peak
			touch_peak, hit_peak = analysis.press_peaks()
			ts = analysis.ts
			signal = analysis.signal
	
			print('{:6s} ----> '.format(filename), end='')
			cprint(f'({y},{x})', 'cyan')
			s = fig.add_subplot(gs[y,x])
			s.tick_params(axis='both', which='major', labelsize=6)

			time_ms = analysis.time_ms

			#Time in ms in the behaviour of press peak (10 ms before and 10 ms after the detected one)
			if zoom:
				first_time = touch_peak[0] - analysis.num_samples(1e-2)
				last_time = hit_peak[-1] + analysis.num_samples(1e-2)
				time_ms = np.arange(first_time, last_time)

			#Plot an audio files with highlighted peaks
			s.plot(time_ms*ts, signal[time_ms], color='blue', label = filename[:-4])
			s.plot(touch_peak*ts, signal[touch_peak], color='red')
			s.plot(hit_peak*ts, signal[hit_peak], color='green')
			s.set_yticklabels([])
			s.set_xticklabels([])
			s.legend(loc='upper right', fontsize=5, framealpha=0.5)

			if x == (n-1):
				y += 1
				x=0
			else:
				x += 1

		if self.OUTPUT_FOLDER != self.DEFAULT_OUTPUT:
			#Store the graphics as image on the File System
			fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+subfolder+self.EXTENSION_PLOT)
			plt.close(fig)
		else:
			#Show the graphics in a window
			plt.show()

	def plot_single_wave(self, filename, analysis, zoom):
		'''
        Plot the audio signal with name filename with
		highlighted peaks and FFT of push and hit peaks
        
        Args:
			filename (str): name of wav file

			analysis (ExtractFeatures): ExtractFeatures related to the
  	    								wav file with name filename

			zoom (bool): True if you want to plot zoomed version of
						 press peak, False otherwise
        '''
		#Evaluation of press peak and hit peaks
		features = analysis.extract()
		touch_feature = features['touch']
		hit_feature = features['hit']

		time_ms = analysis.time_ms
		#Time in ms in the behaviour of press peak (10 ms before and 10 ms after the detected one)
		if zoom:
			first_time = touch_feature.peak[0] - analysis.num_samples(1e-2)
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
		s_top.plot(touch_feature.peak*ts, signal[touch_feature.peak], color='red')
		s_top.plot(hit_feature.peak*ts, signal[hit_feature.peak], color='green')
		s_top.set_title('Amplitude')
		s_top.set_xlabel('Time(s)')
		s_top.tick_params(axis='both', which='major', labelsize=6)

		#Plot FFT double-sided transform of PRESS peak
		s1.plot(touch_feature.freqs, touch_feature.fft_signal, color='red')
		s1.set_xlabel('Frequency (Hz)')
		s1.set_ylabel('FFT of TOUCH PEAK')
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

		if self.OUTPUT_FOLDER != self.DEFAULT_OUTPUT:
			#Store the graphics as image on the File System
			fig.savefig(os.path.dirname(self.OUTPUT_FOLDER)+'/'+filename[:-4]+self.EXTENSION_PLOT)
			plt.close(fig)
		else:
			#Show the graphics in a window
			plt.show()
		
	def extract(self, option):
		'''
		Extract features, store them in OUTPUT_CSV_TRAINING with related
		labels and store labels with their indices in OUTPUT_CSV_DICT_LABEL
        
        Args:
            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction
        '''
		try:
			if self.RECURSIVE:
				#Extraction performed only on recusrive input folder
				cprint('\n   Completed extraction for:', 'red')
				cprint(self.LINE, 'red')
				
				#If specified output dir for csv doesn't exist, it creates it
				path_csv = utility.uniform_dir_path(self.OUTPUT_FOLDER+utility.OPTIONS[option])
				if not utility.OPTIONS[option] in os.listdir(self.OUTPUT_FOLDER):
					if not os.path.exists(path_csv):
						os.mkdir(path_csv)

				#Write features+labels and labels+index in csv files
				label = 0
				with open(path_csv+self.OUTPUT_CSV_TRAINING, 'w', newline='') as train_fp,\
					 open(path_csv+self.OUTPUT_CSV_DICT_LABEL, 'w', newline='') as label_fp:
					csv_train  = writer(train_fp)
					csv_label  = writer(label_fp)
					label=self.compute_entry(csv_train, csv_label, option)

				#Debug info
				cprint('\n'+self.LINE, 'red')
				print('{:>20s}'.format('Features size:'), end=' ')
				cprint('{:<d}'.format(self.FEATURE_SIZE), 'green')
				print('{:>20s}'.format('Number of keys:'), end=' ')
				cprint('{:<d}'.format(label), 'green')
				cprint(self.LINE, 'red')

		except KeyboardInterrupt:
			#Terminate the program (detected CTRL+C)
			plt.close()
			exit(0)

	def compute_entry(self, csv_train, csv_label, option):
		'''
		Compute entries (feature+label) for each wav file
        
        Args:
			csv_train (csv.writer): Writer used to write feature+label for 
									each audio file

			csv_label (csv.writer): Writer used to write labels related with
									their indices

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction

		Returns:
			label (int): Number of keys processed
		'''
		label = 0
		row_length = 0
		#Extraction for each subfolder of DATA_FOLDER
		for subfolder in self.wav_files.keys():
			#Extract features from the audio signals of all the wav files
			#inside the subfolder  
			
			#Sort wav files in subfolder by increasing num order
			subfolder_files = [x for x in self.wav_files[subfolder]]
			subfolder_files.sort()

			if subfolder_files:
				#Collect the analysis objects, one for
				# each wav files in the subfolder

				#Print the key to be avaluated and store it with its label
				#in the file OUTPUT_CSV_DICT_LABEL
				row_length = self.print_key(csv_label, subfolder, label, row_length)

				#FIRST WAV
				#Reading audio file
				fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+subfolder_files[0])
				#Analysis of audio signal
				analysis = ef.ExtractFeatures(fs, signal)

				#Store features of first audio in OUTPUT_CSV_FILE
				if label==0:
					self.FEATURE_SIZE = self.store_features(csv_train, subfolder, subfolder_files[0], analysis, label, option)
				else:
					size = self.store_features(csv_train, subfolder, subfolder_files[0], analysis, label, option)

				#OTHER WAV FILEs
				for f in subfolder_files[1:]:
					#Reading audio file
					fs, signal = wave.read(self.DATA_FOLDER+subfolder+'/'+f)
					#Analysis of audio signal
					analysis = ef.ExtractFeatures(fs, signal)
					#Store features in OUTPUT_CSV_FILE
					self.store_features(csv_train, subfolder, f, analysis, label, option)


				#Update label value
				label += 1

			else:
				cprint('[Folder EMPTY]', 'blue', end=' ')
				print("No '.wav' files inside the subfolder", end=' ')
				cprint(f'{self.DATA_FOLDER+subfolder}', 'green')

		return label

	def print_key(self, csv_label, key, label, row_length):
		"""
		Print a key and insert it in the csv file with its label

		Args:
			csv_label (csv.writer): Writer used to write labels related with
									their indices

			key (str): Key value mapped by keylogger

			label (int): Label (index) related to key that will be used to
						 classify a press peak later.

			row_length (int): Length of last row of printed keys
							  (to obtain symmetry and to be readable on CLI)

		Returns:
			label (int): Number of keys processed
		"""

		#Manage the print if the key is going to overcome line limit
		if (row_length + len(key)) > (len(self.LINE)-1):
			print(f'\n{key}', end='  ')
			row_length = 0
		else:
			print(f'{key}', end='  ')

		row_length += (len(key) + 2)
		csv_label.writerow([key, label])
		
		return row_length

	def store_features(self, csv_writer, subfolder, filename, analysis, label, option):
		"""
		Print a key and insert it in the csv file with its label

		Args:
			csv_train (csv.writer): Writer used to write feature+label for 
									each audio file

			subfolder (str): subfolder of OUTPUT_FOLDER+'/spectrum_less' where all the 
							 spectrogram images related to 'subfolder' label will be stored

			filename (str): Filename format for each spectrogram image

			analysis (ExtractFeatures.ExtractFeatures): Object for analysis of a particular
														audio peak

			label (int): Number of keys processed

            option (int): Number used to select type of features
                          and subfolder to be used for trained model
                          extraction

		Returns:
			length_feature (int): Length of the feature computed
		"""

		length_feature = 0

		if utility.OPTIONS[option]=='touch':
			#Extraction of touch feature
			features = analysis.extract()
			touch_features = features['touch'].fft_signal
			final_features = np.append(touch_features, label)
			csv_writer.writerow(final_features)
			length_feature = len(final_features)

		elif utility.OPTIONS[option]=='touch_hit':
			#Extraction of touch_hit feature
			features = analysis.extract()
			touch_features = features['touch'].fft_signal
			hit_features = features['hit'].fft_signal	
			final_features = np.concatenate((touch_features, hit_features))
			final_features = np.append(final_features, label)
			csv_writer.writerow(final_features)
			length_feature = len(final_features)
		
		elif utility.OPTIONS[option]=='spectrum':
			#Extraction of feature from spectrogram
			features = analysis.extract(spectrum=True)
			fig, ax = plt.subplots(1)
			fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
			img_feature = np.concatenate((analysis.signal[features[0]], analysis.signal[features[1]]))
			ax.axis('equal')
			spectrum, freqs, t, img_array = ax.specgram(img_feature, NFFT=len(features[0]), Fs=analysis.fs)

			#Creation of subfolder in output folder to store the spectrogram images
			if not os.path.exists(self.OUTPUT_FOLDER+'spectrum/'+subfolder):
				os.mkdir(self.OUTPUT_FOLDER+'spectrum/'+subfolder)

			#plt.show()
			fig.savefig(self.OUTPUT_FOLDER+'spectrum_less/'+subfolder+'/'+filename[:-4]+'.jpg', dpi=300)
			plt.close(fig)
			#print('Completed image')

			#img = Image.fromarray(img_array)
			#img.save(self.OUTPUT_FOLDER+'spectrum/'+subfolder+'/'+filename[:-4]+'.jpg')

			#with Image.open(self.OUTPUT_FOLDER+'spectrum/'+subfolder+'/'+filename[:-4]+'.jpg') as img:
			#	final_features = np.asarray(img)
			#	final_features = np.append(final_features, label)
			#	csv_writer.writerow(final_features)
			#	length_feature = len(final_features)

		return length_feature