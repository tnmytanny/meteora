import os

# filenames = os.listdir(os.curdir)
# print(filenames)

# j = 1
# for i in filenames:
# 	if i.find('.py') == -1 :
# 		if os.path.isfile(i):
# 			# os.rename(i,str(j)+'.mp4')
# 			os.system('ffmpeg -ss 00:00:00 -i '+ i + ' -t 00:00:03 -c copy ./videos/'+str(j)+'.mp4')
# 			os.system('ffmpeg -i ./videos/' + str(j) + '.mp4 -vf fps=25 ./videos/images/' + str(j) + '_%04d.jpg -hide_banner')
# 			j+=1

def removeExtension(basename):
    lastDotPosition = basename.rindex(".")
    # if (lastDotPosition === -1) return srcKey
    return basename[0:lastDotPosition]

image_files = os.listdir('./videos/images')
# print(image_files)
for x in image_files:
	y = removeExtension(x)
	print(y[:-5])
	if int(y[-4:]) > 75 :
		os.remove('./videos/images/' + x)