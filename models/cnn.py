import tensorflow as tf 
import tensorlayer as tl 
from tensorlayer.layers import *
import time

class CNN(object):
	def __init__(self, lr=0.0005, l2_reg=None, W_init=tf.truncated_normal_initializer(stddev=0.2),
				batch_size=16, 
				inputs=tf.placeholder(tf.float32, shape=[None, 224, 224, 3]),
				labels=tf.placeholder(tf.float32, shape=[None, 100]),
				n_epoch=100, reuse=False):
		self.W_init = W_init
		self.reuse = reuse
		self.learning_rate = lr
		self.l2_reg = l2_reg
		self.batch_size = batch_size
		self.n_epoch = n_epoch
		self.inputs = inputs
		self.labels = labels
		self.network = self.model(self.inputs)

	def model(self, inputs):
		with tf.variable_scope('mlp', reuse=self.reuse):
			tl.layers.set_name_reuse(self.reuse)
			network = InputLayer(inputs, name='net/in')
			network = Conv2dLayer(network, act=tf.nn.relu, W_init=self.W_init,
								shape=[5, 5, 3, 128], strides=[1, 2, 2, 1], padding='VALID',
								name='net/cnn_layer1')
			network = Conv2dLayer(network, act=tf.nn.relu, W_init=self.W_init,
								shape=[3, 3, 128, 64], strides=[1, 2, 2, 1], padding='VALID',
								name='net/cnn_layer2')
			network = FlattenLayer(network, name='net/flatten')
			network = DenseLayer(network, n_units=1024, act=lambda x: tl.act.lrelu(x, 0.2),
								W_init=self.W_init, name='net/Dense3')
			network = DenseLayer(network, n_units=100, act=tf.identity,
								W_init=self.W_init, name='net/output')
		return network

	def loss(self, label, logits):
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
			if self.l2_reg is not None:
				l2_loss = tf.contrib.layers.l2_regularizer(self.l2_reg)(network.all_params[5]) + tf.contrib.layers.l2_regularizer(self.l2_reg)(network.all_params[7])
				loss = loss + l2_loss
		return loss

	def fit(self, X_train, y_train):
		logits = self.network.outputs
		loss_ = self.loss(self.labels, logits)
		train_param = self.network.all_params
		train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss_, var_list=train_param)
		sess = tf.InteractiveSession()

		start_time = time.time()
		print('Start Training CNN at ', time.asctime(time.localtime(start_time)))
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(self.n_epoch):
			epoch_time = time.time()
			total_loss, n_batch = 0, 0
			for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, self.batch_size, shuffle=True):
				batch_loss, _ = sess.run([loss_, train_op], feed_dict={self.inputs: X_batch, self.labels: y_batch})
				total_loss += batch_loss; n_batch += 1
			print("Epoch %d of %d, loss %f" % (epoch + 1, self.n_epoch, total_loss / n_batch))
			print("Epoch %d takes %d seconds" % (epoch + 1, time.time() - epoch_time))
			
			#if (epoch + 1) % 50 == 0:
			#	tl.files.save_npz(self.network.all_params , name='model_cnn' + str(epoch + 1) + '.npz')
		
		end_time = time.time()
		print('Training takes %d seconds' % (end_time - start_time))


	def predict(self, X):
		pass