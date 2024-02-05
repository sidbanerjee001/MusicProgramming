import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import matplotlib as mpl

# HEADER: Useful functions defined below *MOST/ALL DO NOT ACCOUNT FOR ERRONEOUS INPUT

# Option to write numpy array data to file for reading
def writeExport(filepath, data):
	file = open(filepath, 'w')
	for d in data:
		file.write(str(d) + "\n")
	file.close()

# Create a plot with x_axis and y_axis data inputs
def createPlot(x_axis, y_axis, x_axis_label="x-axis", y_axis_label="y-axis", title="Title"):
	plt.clf()

	plt.plot(x_axis, y_axis)
	plt.ylabel(y_axis_label)
	plt.xlabel(x_axis_label)
	plt.title(title)

	plt.show()

# FUNCTIONALITY

samplerate, data = wavfile.read('samples/sample1.wav')

sample_length_seconds = data.size/samplerate/2 # Alternatively, data.shape[0] / samplerate
Time = np.linspace(0., sample_length_seconds, data.shape[0])

# Subchunk size: 48721784 = NumSamples * NumChannels * BitsPerSample/8
# BitsPerSample = 48
# NumChannels = 2
# data.size returns 2*actual_size since there are two channels

# CHANNELS => data[:, 0] (LEFT), data[:, 1] (RIGHT)

data = np.flip(data,0) # account for little-endian data (WAV inherency)
data = data / 32768 # normalize between 0 and 1 (16-bit sample)
dataSetSize = data.size

# createPlot(Time, data, "Time", "Amplitude", "Sample 1")

# Sample 2.2 second clip
lower = (data.size//2)//2
upper = lower+int(2.2 * samplerate)
sampled_clip = data[lower:upper]

# Phase 1: Filterbank
X_FFT = fft(sampled_clip)
n = np.arange(0, len(X_FFT))
freq = n/(len(X_FFT)/samplerate)

# # writing to wave file using scipy
# wavfile.write('dataset/exports/sample1_zeroed.wav', samplerate, data)

