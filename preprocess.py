import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import fftpack
from scipy.io import wavfile as wav

frame_size = 0.025
stride = 0.001
pre_emphasis = 0.95
NFFT = 512
nfilt = 40


sample_rate, input_signal = wav.read('test.wav')

# Pre Emphasis
pre_emph_signal = np.append(input_signal[0], input_signal[1:] - pre_emphasis * input_signal[:-1])

# Split to frames
frame_length = int(round(sample_rate * frame_size))
frame_stride = int(round(sample_rate * stride))
signal_length = len(input_signal)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_stride))
pad_signal_length = num_frames * frame_stride + frame_length
z = np.zeros((pad_signal_length - signal_length))
padded_signal = np.append(pre_emph_signal, z)
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_length, 1)).T
frames = padded_signal[indices.astype(np.int32, copy=False)]

# Apply hamming window
frames = frames * np.hamming(frame_length)

# Apply FFT and Power 
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

# Apply triangular filter in MEL's scale
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
hz_points = (700 * (10**(mel_points / 2595) - 1))
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])
    f_m = int(bin[m])
    f_m_plus = int(bin[m + 1])

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
filter_banks = 20 * np.log10(filter_banks)