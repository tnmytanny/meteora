import scipy.signal as signal
import scipy.io.wavfile as wavfile
import wave
import struct
import librosa.core as audio
import numpy as np

def open_wavfile(filename='test.wav'):
	input_signal = wave.open(filename)
	return input_signal

def to_float(w):
	astr = w.readframes(w.getnframes())
	a = struct.unpack("%ih" % (w.getnframes()* w.getnchannels()), astr)
	a = [float(val) / pow(2, 15) for val in a]
	print(np.asarray(a))
	return np.asarray(a)

def audio_to_spectrogram(input_signal, n_fft, hop_length, win_length, window='hann', center=True):
	return(audio.stft(input_signal, n_fft, hop_length, win_length, window, center))

def spectrogram_to_audio(input_spectrogram, hop_length, win_length, window='hann', center=True):
	return(audio.istft(input_spectrogram, hop_length, win_length, window, center))

def real_spectrogram(input_spectrogram):
	return np.absolute(input_spectrogram)