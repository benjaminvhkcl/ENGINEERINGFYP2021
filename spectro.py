import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

#import file and initiate file and parameters

scale_file = "audio/Untitled.m4a"


ipd.Audio(scale_file)
scale, sr = librosa.load(scale_file)
FRAME_SIZE = 2048
HOP_SIZE = 512

S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
S_scale.shape
type(S_scale[0][0])

Y_scale = np.abs(S_scale) ** 2

#Visualize Spectogram function
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,sr=sr,hop_length=hop_length, x_axis="time", y_axis=y_axis, cmap='magma')
    plt.colorbar(format="%+2.f")
    plt.show()

# plot_spectrogram(Y_scale, sr, HOP_SIZE)

Y_log_scale = librosa.power_to_db(Y_scale)



mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr, cmap='magma')
plt.colorbar(format="%+2.f")
plt.title("Einaudi - Experience - Salle Pleyel - 2019 - Logmel")
plt.show()

