from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
from python_speech_features import fbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

(rate,sig) = wav.read("test.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

plt.imshow(fbank_feat, cmap='hot', interpolation='nearest')
plt.show()