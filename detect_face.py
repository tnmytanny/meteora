import boto3
import os
from PIL import Image
import tensorflow as tf
# Create an S3 client
#intiate s3 resource
s3 = boto3.resource('s3')

# select bucket
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

    image = tf.placeholder(dtype = tf.float32,shape = [75,1,1024],name = 'image')
    kernel1 = tf.Variable(tf.random_normal([7,1,1024,256]), dtype=tf.float32, name='kernel1')
    # kernel = tf.reshape(k, [7, 1, 1024, 256], name='kernel')
    images  = tf.reshape(image, [1, 75, 1, 1024], name='image')
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

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, conv6, dtype=tf.float32)
    
    # final_name = removeExtension(textfile) + '.png'
    # im5.save(final_name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(conv6, feed_dict={image:i}).shape)
