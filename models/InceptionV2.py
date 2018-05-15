import tensorflow as tf
import numpy as np
import skimage.io,skimage.transform
import tensorlayer as tl
from tensorflow.contrib.slim.python.slim.nets.inception_resnet_v2 import inception_resnet_v2,inception_resnet_v2_arg_scope
from tensorflow.python.ops import variables

train_tfrecord = "../datasets/train.tfrecord"
test_txt="../datasets/test.txt"
result_txt="../datasets/"
imagedir = '../datasets/test/'
image_size = 299
BATCH_SIZE = 4
BATCH_CAPACITY = 512
MIN_AFTER_DEQU = 192
train_step = 500000
test_step = 1400
image_channel = 3
slim = tf.contrib.slim
classnum = 100

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float32),
                                           'imgl': tf.FixedLenFeature([], tf.string),
                                           'imgm': tf.FixedLenFeature([], tf.string),
                                           'imgs': tf.FixedLenFeature([], tf.string)
                                       })
    imgl = tf.decode_raw(features['imgl'], tf.uint8)
    imgl = tf.reshape(imgl, [400, 400, 3])
    imgl = tf.cast(imgl, tf.float32)
    imgl=tf.image.random_brightness(imgl,50)
    imgl=tf.image.random_contrast(imgl,0.6,1.4)
    imgl=tf.image.random_hue(imgl,0.05)
    imgl = tf.image.per_image_standardization(imgl)

    imgm = tf.decode_raw(features['imgm'], tf.uint8)
    imgm = tf.reshape(imgm, [299, 299, 3])
    imgm = tf.cast(imgm, tf.float32)
    imgm=tf.image.random_brightness(imgm,50)
    imgm=tf.image.random_contrast(imgm,0.6,1.4)
    imgm=tf.image.random_hue(imgm,0.05)
    imgm = tf.image.per_image_standardization(imgm)

    imgs = tf.decode_raw(features['imgs'], tf.uint8)
    imgs = tf.reshape(imgs, [224, 224, 3])
    imgs = tf.cast(imgs, tf.float32)
    imgs=tf.image.random_brightness(imgs,50)
    imgs=tf.image.random_contrast(imgs,0.6,1.4)
    imgs=tf.image.random_hue(imgs,0.05)
    imgs = tf.image.per_image_standardization(imgs)

    label = tf.cast(features['label'], tf.float32)
    label = tf.reshape(label, [1])
    return imgl, imgm,imgs, label

def load_image(path):
    img = skimage.io.imread(path)
    resized_img = skimage.transform.resize(img, (image_size, image_size),mode='constant',preserve_range=True)
    return resized_img

def loss(labels, logits):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

def accuracy(labels, logits):
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), "float"))
    return accuracy

def inference(inputs,istrain):
    netinputs=tl.layers.InputLayer(inputs, name='input_layerg')
    if istrain:
        is_training=True
        reuse=None
    else:
        is_training=False
        reuse=True
    with tf.variable_scope("",reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            network = tl.layers.SlimNetsLayer(layer=netinputs, slim_layer=inception_resnet_v2,
                                              slim_args={
                                                  'num_classes': None,
                                                  'is_training': is_training,
                                                   'reuse' : reuse,
                                              },
                                              name='InceptionResnetV2'
                                              )
        network = tl.layers.FlattenLayer(network)
        network = tl.layers.DenseLayer(network, n_units=classnum, name='out')
    return network.outputs,network

if __name__ == '__main__':
    f = open(test_txt)
    lines = f.readlines()
    flist = []
    for line in lines:
        temp = line.strip('\n')
        flist.append(temp)
    f.close()

    train_imgl, train_imgm, train_imgs,train_label = read_and_decode(train_tfrecord)
    img_train_imgl, img_train_imgm, img_train_imgs,img_train_label = tf.train.shuffle_batch([train_imgl, train_imgm, train_imgs,train_label],batch_size=BATCH_SIZE, capacity=BATCH_CAPACITY,min_after_dequeue=MIN_AFTER_DEQU)

    with tf.name_scope('inputlayer'):
        imginputs = tf.placeholder(tf.float32, [None, image_size, image_size, image_channel], 'inputs')
        testinputs = tf.placeholder(tf.float32, [image_size, image_size, image_channel], 'testinputs')
        labels = tf.placeholder(tf.float32, [None, 1], 'label')
        label_oh = tf.one_hot(tf.cast(tf.reshape(labels, [BATCH_SIZE]), tf.int32), classnum ,name='onehot_label')

    dropout_keep_prob = tf.placeholder(tf.float32, None, 'keep_prob')
    learning_rate = tf.placeholder(tf.float32, None, 'learning_rate')

    trainlabel, net = inference(imginputs,True)
    testlabel, _ = inference(tf.expand_dims(tf.image.per_image_standardization(testinputs),axis=0) , False)

    loss_train = loss(label_oh, trainlabel)
    accuracy_train = accuracy(label_oh, trainlabel)
    varlisttrain = variables._all_saveable_objects().copy()
    for _ in range(890):
        del varlisttrain[0]
    varlist = variables._all_saveable_objects().copy()
    for _ in range(2):
        del varlist[-1]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss_train,var_list=varlisttrain)

    init = tf.global_variables_initializer()
    savertemp = tf.train.Saver(max_to_keep=2)
    saver = tf.train.Saver(max_to_keep=2)
    dfgraph = tf.get_default_graph()

    with tf.Session() as sess:
        learnrt = 0.0001
        sess.run(init)
        savertemp = tf.train.Saver(var_list=varlist ,max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=2)
        savertemp.restore(sess, "./inception_resnet_v2_2016_08_30.ckpt")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                acc=0.
                los=0.
                for i in range(train_step):
                    batch_imgm,batch_label=sess.run([img_train_imgm,img_train_label])
                    _,loss_result, accuracy_result = sess.run([train, loss_train, accuracy_train],feed_dict={imginputs: batch_imgm,labels: batch_label,learning_rate: learnrt})
                    acc+=accuracy_result
                    los += loss_result
                    if (i+1)%100==0:
                        print('step:',i+1)
                        print('acc:', acc/100)
                        print('loss:', los/100)
                        acc=0.
                        los=0.
                    if (i + 1) % test_step==0:
                        resultlist=[]
                        for filename in flist:
                            img = skimage.io.imread(imagedir+filename)
                            img = skimage.transform.resize(img, (image_size, image_size), mode='edge', preserve_range=True)
                            test_label = sess.run(testlabel,feed_dict={testinputs: img})
                            t_label=np.argmax(test_label, 1)[0]
                            resultlist.append(filename+' '+str(int(t_label)+1)+'\n')
                        fresult = open(result_txt+str(i + 1)+'steps.csv', 'w')
                        for lin in resultlist:
                            fresult.writelines(lin)
                        fresult.close()
                    if i % 2000 == 0 and i >0:
                        save_path = saver.save(sess, "./my_net/step"+str(i)+".ckpt")

        except tf.errors.OutOfRangeError:
            print('finished')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
