import boto3
import os
import tensorflow as tf
import spectrogram as S
import numpy as np
from PIL import Image

s3 = boto3.resource('s3')
my_bucket = s3.Bucket('tannyisback')

# for obj in my_bucket.objects.all():
#     print(str(obj.key))
#     if str(obj).find('images/') != -1 and str(obj.key) != 'images/':
#         print(str(obj.key))
#         my_bucket.download_file(obj.key, os.path.join("Down", obj.key))
#         print("done")
#     elif str(obj).find('text/') != -1:
#         print(str(obj.key))
#         my_bucket.download_file(obj.key, os.path.join("Downn", obj.key))
#         print("done")   

i = []

n_hidden = 128

def removeExtension(basename):
    lastDotPosition = basename.rindex(".")
    # if (lastDotPosition === -1) return srcKey
    return basename[0:lastDotPosition]

def getpicname(textfile):
    pic = 'Down/images/' + removeExtension(textfile) + '.jpg'
    return pic

def helper(textfile):
    # bucket='tannyisback'
    # client=boto3.client('rekognition')

    # response = client.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':photo}},Attributes=['ALL'])

    # print('Detected faces for ' + photo)    
    # for faceDetail in response['FaceDetails']:
    #     print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
    #     print('Here are the other attributes:')
    #     k = str(json.dumps(faceDetail, indent=4, sort_keys=True))
    #     k_start = k.find('BoundingBox')
    #     k_last = k.find('Confidence',k_start)
    #     my_start = k[k_start+14:k_last-2]
    #     print(my_start)
    photo = getpicname(textfile)
    my_text = 'Downn/text/' +textfile
    with open(my_text,'r') as dat:
        data = dat.read()
        first = data.find(' ')
        second = data.find(' ',first+1)
        third = data.find(' ',second+1)
        x_start = int(data[:first])-2
        y_start = int(data[first+1:second])-2
        x_end = int(data[second+1:third])-2
        y_end = int(data[third+1:])-2
    im = Image.open(photo).convert('L')
    im = im.crop((x_start,y_start,x_end,y_end))
    im5 = im.resize((32, 32), Image.ANTIALIAS)
    i.append([list(im5.getdata())])

if(__name__ == '__main__'):
    path = os.curdir+"/Downn/text"
    print(path)
    for filename in os.listdir(path):
        print(filename)
        helper(filename)
    #--------------------------------------------------------------------#    
    input_signal = S.open_wavfile()
    float_signal = S.to_float(input_signal)
    # resampled_signal = S.resample(input_signal)
    spectrogram = S.audio_to_spectrogram(float_signal, 512, 160, 400)
    spectrogram = np.transpose(spectrogram)
    print(spectrogram.shape[0])
    y = spectrogram.shape[0]
    if y>298:
        spectrogram = spectrogram[:298-y,:]
    print(spectrogram.shape)

    real_part = spectrogram.real
    imag_part = spectrogram.imag

    split_channel = np.array([[real_part, imag_part]])
    split_channel = split_channel.reshape([1,298,257,2])
    #--------------------------------------------------------------------# 

    image = tf.placeholder(dtype = tf.float32,shape = [75,1,1024],name = 'image')
    images  = tf.reshape(image, [1, 75, 1, 1024], name='image')

    audio_tensor = tf.placeholder("float", [1,298,257,2])

    audio_kernel1 = tf.Variable(tf.random_normal([1,7,2,96]), dtype=tf.float32, name='audio_kernel1')
    audio_kernel2 = tf.Variable(tf.random_normal([7,1,96,96]), dtype=tf.float32, name='audio_kernel2')
    audio_kernel3 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel3')
    audio_kernel4 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel4')
    audio_kernel5 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel5')
    audio_kernel6 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel6')
    audio_kernel7 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel7')
    audio_kernel8 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel8')
    audio_kernel9 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel9')
    audio_kernel10 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel10')
    audio_kernel11 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel11')
    audio_kernel12 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel12')
    audio_kernel13 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel13')
    audio_kernel14 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel14')
    audio_kernel15 = tf.Variable(tf.random_normal([1,1,96,8]), dtype=tf.float32, name='audio_kernel15')   

    audio_conv1 = tf.nn.convolution(input=audio_tensor, filter=audio_kernel1, padding='SAME', strides=None, dilation_rate=None, name='audio_conv1', data_format=None)
    r1 = tf.nn.relu(audio_conv1, name='r1')
    audio_conv2 = tf.nn.convolution(input=r1, filter=audio_kernel2, padding='SAME', strides=None, dilation_rate=None, name='audio_conv2', data_format=None)
    r2 = tf.nn.relu(audio_conv2, name='r2')
    audio_conv3 = tf.nn.convolution(input=r2, filter=audio_kernel3, padding='SAME', strides=None, dilation_rate=None, name='audio_conv3', data_format=None)
    r3 = tf.nn.relu(audio_conv3, name='r3')
    audio_conv4 = tf.nn.convolution(input=r3, filter=audio_kernel4, padding='SAME', strides=None, dilation_rate=[2,1], name='audio_conv4', data_format=None)
    r4 = tf.nn.relu(audio_conv4, name='r4')
    audio_conv5 = tf.nn.convolution(input=r4, filter=audio_kernel5, padding='SAME', strides=None, dilation_rate=[4,1], name='audio_conv5', data_format=None)
    r5 = tf.nn.relu(audio_conv5, name='r5')
    audio_conv6 = tf.nn.convolution(input=r5, filter=audio_kernel6, padding='SAME', strides=None, dilation_rate=[8,1], name='audio_conv6', data_format=None)
    r6 = tf.nn.relu(audio_conv6, name='r6')
    audio_conv7 = tf.nn.convolution(input=r6, filter=audio_kernel7, padding='SAME', strides=None, dilation_rate=[16,1], name='audio_conv7', data_format=None)
    r7 = tf.nn.relu(audio_conv7, name='r7')
    audio_conv8 = tf.nn.convolution(input=r7, filter=audio_kernel8, padding='SAME', strides=None, dilation_rate=[32,1], name='audio_conv8', data_format=None)
    r8 = tf.nn.relu(audio_conv8, name='r8')
    audio_conv9 = tf.nn.convolution(input=r8, filter=audio_kernel9, padding='SAME', strides=None, dilation_rate=None, name='audio_conv9', data_format=None)
    r9 = tf.nn.relu(audio_conv9, name='r9')
    audio_conv10 = tf.nn.convolution(input=r9, filter=audio_kernel10, padding='SAME', strides=None, dilation_rate=[2,2], name='audio_conv10', data_format=None)
    r10 = tf.nn.relu(audio_conv10, name='r10')
    audio_conv11 = tf.nn.convolution(input=r10, filter=audio_kernel11, padding='SAME', strides=None, dilation_rate=[4,4], name='audio_conv11', data_format=None)
    r11 = tf.nn.relu(audio_conv11, name='r11')
    audio_conv12 = tf.nn.convolution(input=r11, filter=audio_kernel12, padding='SAME', strides=None, dilation_rate=[8,8], name='audio_conv12', data_format=None)
    r12 = tf.nn.relu(audio_conv12, name='r12')
    audio_conv13 = tf.nn.convolution(input=r12, filter=audio_kernel13, padding='SAME', strides=None, dilation_rate=[16,16], name='audio_conv13', data_format=None)
    r13 = tf.nn.relu(audio_conv13, name='r13')
    audio_conv14 = tf.nn.convolution(input=r13, filter=audio_kernel14, padding='SAME', strides=None, dilation_rate=[32,32], name='audio_conv14', data_format=None)
    r14 = tf.nn.relu(audio_conv14, name='r14')
    audio_conv15 = tf.nn.convolution(input=r14, filter=audio_kernel15, padding='SAME', strides=None, dilation_rate=None, name='audio_conv15', data_format=None)
    final_signal = tf.nn.relu(audio_conv15, name='final')


    final_audio = tf.reshape(final_signal,[1,298,1,2056])

    kernel1 = tf.Variable(tf.random_normal([7,1,1024,256]), dtype=tf.float32, name='kernel1')
    res1 = tf.nn.convolution(images, kernel1,padding = "SAME",strides = [1,1],dilation_rate = [1,1])
    conv1 = tf.nn.relu(res1, name='conv1')
    kernel2 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel2')
    res2 = tf.nn.convolution(conv1,kernel2,padding = "SAME",strides = [1,1],dilation_rate = [1,1])
    conv2 = tf.nn.relu(res2, name='conv2')
    kernel3 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel3')
    res3 = tf.nn.convolution(conv2,kernel3,padding = "SAME",strides = [1,1],dilation_rate = [2,1])
    conv3 = tf.nn.relu(res3, name='conv3')
    kernel4 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel4')
    res4 = tf.nn.convolution(conv3,kernel4,padding = "SAME",strides = [1,1],dilation_rate = [4,1])
    conv4 = tf.nn.relu(res4, name='conv4')
    kernel5 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel5')
    res5 = tf.nn.convolution(conv4,kernel5,padding = "SAME",strides = [1,1],dilation_rate = [8,1])
    conv5 = tf.nn.relu(res5, name='conv5')
    reshape_conv5 = tf.reshape(conv5,[1,256,1,75],name='reshape_conv5')
    kernel6 = tf.Variable(tf.random_normal([5,1,75,298]), dtype=tf.float32, name='kernel6')
    res6 = tf.nn.convolution(reshape_conv5,kernel6,padding = "SAME",strides = [1,1],dilation_rate = [16,1])
    conv6 = tf.nn.relu(res6, name='conv6')

    final_visual = tf.reshape(conv6,[1,298,1,256],name='final_visual')

    final_input = tf.concat([final_audio,final_visual],3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(final_input, feed_dict={audio_tensor:split_channel,image:i}).shape)