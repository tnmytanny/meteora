import scipy.io.wavfile as wav
import os

dir1 = 'Data/1/'
dir2 = 'Data/2/'

files1 = next(os.walk(dir1))[2]
files2 = next(os.walk(dir2))[2]

lenfiles1 = len(files1)
lenfiles2 = len(files2)

if lenfiles1>lenfiles2:
	large = files1
	small = files2
else:
	large = files2
	small = files1

for i in small:
	for j in large:
		