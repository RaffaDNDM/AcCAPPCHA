import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import sys
import scipy
import scipy.fftpack as fftpk

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
    
amplitude = b''.join(frames)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

time = np.linspace(0, RECORD_SECONDS, num=len(frames))

fig = plt.figure()
s = fig.add_subplot(111)
s1 = fig.add_subplot(321)
s2 = fig.add_subplot(322)
s3 = fig.add_subplot(323)
s4 = fig.add_subplot(324)
s5 = fig.add_subplot(325)
s6 = fig.add_subplot(326)
fig.tight_layout(pad=3.0)
amplitude = np.fromstring(amplitude, np.int16)
s.plot(amplitude)
s1.plot(amplitude)
s1.specgram(amplitude)
fft_amplitude = np.fft.fft(amplitude)
s3.plot(fft_amplitude)
s4.specgram(fft_amplitude)
s5.plot(np.abs(fft_amplitude))
s6.specgram(np.abs(fft_amplitude))
plt.show()

'''
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
'''