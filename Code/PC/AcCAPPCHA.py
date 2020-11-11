import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import sys
import scipy
import scipy.fftpack as fftpk
from time import sleep

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

sleep(1)
print('Type')
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
gs = fig.add_gridspec(2, 2)
s_top = fig.add_subplot(gs[0, :])
s1 = fig.add_subplot(gs[1,0])
s2 = fig.add_subplot(gs[1,1])
fig.tight_layout(pad=3.0)
amplitude = np.fromstring(amplitude, np.int16)
s_top.plot(amplitude)
fft_amplitude = np.fft.fft(amplitude)
s1.plot(np.abs(fft_amplitude))
s2.specgram(np.abs(fft_amplitude)
plt.show()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
