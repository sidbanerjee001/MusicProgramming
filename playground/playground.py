import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile # For reading/writing wav files
import sounddevice as sd # For playing/recording audio

[samplerate, audio] = scipy.io.wavfile.read("samples/sample4.wav")

# Get a few seconds of this audio
print(samplerate)
n = samplerate*10

audio_left = []

for i in range(0, len(audio)):
    audio_left += [audio[i][0]]

y = np.fft.fft(audio_left)
keepFraction = 20

# Low pass
# yMod = np.array(y)
# yMod[(n//keepFraction):-(n//keepFraction)] = 0

# High pass
yMod = np.array(y)
yMod[0:(n//keepFraction)] = 0
yMod[-(n//keepFraction):-1] = 0



plt.plot(np.real(y))
plt.show()


# playback
# xMod = np.real(np.fft.ifft(yMod)).astype(np.int16)
# sd.play(xMod, samplerate)
# sd.wait()