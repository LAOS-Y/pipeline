import tensorflow as tf
import numpy as np
import skimage.io,skimage.transform
import os
from PIL import Image

f = open('../datasets/train.txt')
imagedir = '../datasets/train/'
trainwriter = tf.python_io.TFRecordWriter("../datasets/train.tfrecord")
randomsize = 1

lines = f.readlines()
flist = []
for line in lines:
    temp = line.strip('\n').split(' ')
    fname = temp[0]
    fclass = int(temp[1])-1
    fd = [fname,fclass]
    flist.append(fd)
f.close()
for _ in range(randomsize):
    indicestrain = np.random.permutation(len(flist))
    for i in range(len(flist)):
        imgfile = imagedir + flist[indicestrain[i]][0]
        imgfileclass = flist[indicestrain[i]][1]
        img = skimage.io.imread(imgfile)
        img_l = skimage.transform.resize(img,(400,400),mode='edge',preserve_range=True)
        img_m = skimage.transform.resize(img,(299,299),mode='edge',preserve_range=True)
        img_s = skimage.transform.resize(img,(224,224),mode='edge',preserve_range=True)
        # img=np.uint8(img)
        sample = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[imgfileclass])),
            'imgl': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Image.fromarray(img_l.astype('uint8'), 'RGB').tobytes()])),
            'imgm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Image.fromarray(img_m.astype('uint8'), 'RGB').tobytes()])),
            'imgs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[Image.fromarray(img_s.astype('uint8'), 'RGB').tobytes()])),
        }))
        trainwriter.write(sample.SerializeToString())
        print(i)
trainwriter.close()
