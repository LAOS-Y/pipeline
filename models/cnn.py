import tensorflow as tf 
import tensorlayer as tl 
from tensorlayer.layers import *
import time

class CNN(object):
	def __init__(self, lr=0.0005, l2_reg=None, W_init=tf.truncated_normal_initializer(stddev=0.2),
				batch_size=16, n_epoch=100,
				inputs=tf.placeholder(tf.float32, shape=[None, 224, 224, 3]),
				labels=tf.placeholder(tf.float32, shape=[None, 100]),
				sess=tf.InteractiveSession(),
				residual_block_per_group=4, widening_factor=4,
				name='CNN', reuse=False):
		self.W_init = W_init
		self.reuse = reuse
		self.learning_rate = lr
		self.l2_reg = l2_reg
		self.batch_size = batch_size
		self.n_epoch = n_epoch
		self.inputs = inputs
		self.labels = labels
		self.name = name
		self.sess = sess
		# wide res-net
		self.blocks_per_group = residual_block_per_group
		self.widening_factor = widening_factor

		self.network = self.model(self.inputs)
		self.pred_ = tf.nn.softmax(self.network.outputs)

	def _residual_block(self, last_layer, count, nb_filters=16, subsample_factor=1, name_prefix='res/'):
		last_channels = last_layer.outputs.get_shape().as_list()[3]
		if subsample_factor > 1:
			subsample = [1, subsample_factor, subsample_factor, 1]
			short_cut = PoolLayer(last_layer, ksize=subsample, strides=subsample, padding='SAME',
								pool=tf.nn.avg_pool, name=name_prefix+'pool_layer'+str(count))
		else:
			subsample = [1, 1, 1, 1]
			short_cut = last_layer

		def zero_pad_channels(x, pad=0):
			pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
			return tf.pad(x, pattern)

		if nb_filters > last_channels:
			short_cut = LambdaLayer(short_cut, zero_pad_channels,
									fn_args={'pad': nb_filters - last_channels},
									name=name_prefix+'lambda_layer'+str(count))

		y = BatchNormLayer(last_layer, decay=0.999, epsilon=1e-05, 
							is_train=True, name=name_prefix+'norm_layer1'+str(count))
		y = Conv2dLayer(y, act=tf.nn.relu, W_init=self.W_init,
						shape=[3, 3, last_channels, nb_filters], strides=subsample,
						padding='SAME', name=name_prefix+'conv_layer1'+str(count))
		y = BatchNormLayer(last_layer, decay=0.999, epsilon=1e-05, 
							is_train=True, name=name_prefix+'norm_layer2'+str(count))
		last_channels = y.outputs.get_shape().as_list()[3]
		final_channels = short_cut.outputs.get_shape().as_list()[3]
		y = Conv2dLayer(y, act=tf.nn.relu, W_init=self.W_init,
						shape=[3, 3, last_channels, final_channels], strides=subsample,
						padding='SAME', name=name_prefix+'conv_layer2'+str(count))

		out = ElementwiseLayer([y, short_cut], combine_fn=tf.add, 
								name=name_prefix+'merge_layer'+str(count))
		return out

	def model(self, inputs):
		with tf.variable_scope(self.name, reuse=self.reuse):
			tl.layers.set_name_reuse(self.reuse)
			net = InputLayer(inputs, name='net/in')
			net = Conv2dLayer(net, act=tf.nn.relu, W_init=self.W_init,
								shape=[5, 5, 3, 128], strides=[1, 2, 2, 1], padding='SAME',
								name='net/conv_layer1')
			net = Conv2dLayer(net, act=tf.nn.relu, W_init=self.W_init,
								shape=[3, 3, 128, 64], strides=[1, 2, 2, 1], padding='SAME',
								name='net/conv_layer2')
			net = Conv2dLayer(net, act=tf.nn.relu, W_init=self.W_init,
								shape=[3, 3, 64, 32], strides=[1, 2, 2, 1], padding='SAME',
								name='net/cnn_layer3')

			for i in range(self.blocks_per_group):
				nb_filters = 16 * self.widening_factor
				count = i
				net = self._residual_block(net, count, nb_filters=nb_filters, subsample_factor=1)

			for i in range(self.blocks_per_group):
				nb_filters = 32 * self.widening_factor
				if i == 0:
					subsample_factor = 2
				else:
					subsample_factor = 1
				count = i + self.blocks_per_group
				net = self._residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

			for i in range(self.blocks_per_group):
				nb_filters = 64 * self.widening_factor
				if i == 0:
					subsample_factor = 2
				else:
					subsample_factor = 1
				count = i + 2 * self.blocks_per_group
				net = self._residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

			last_channels = net.outputs.get_shape().as_list()[3]

			net = Conv2dLayer(net, act=tf.nn.relu, W_init=self.W_init,
								shape=[3, 3, last_channels, 32], strides=[1, 2, 2, 1], padding='VALID',
								name='net/cnn_layer4')

			net = FlattenLayer(net, name='net/flatten')
			#net = DenseLayer(net, n_units=1024, act=lambda x: tl.act.lrelu(x, 0.2),
			#					W_init=self.W_init, name='net/Dense3')
			net = DenseLayer(net, n_units=100, act=tf.identity,
								W_init=self.W_init, name='net/output')
		return net

	def loss(self, label, logits):
		with tf.name_scope('loss'):
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
		return loss

	def fit(self, X_train, y_train):
		logits = self.network.outputs
		loss_ = self.loss(self.labels, logits)
		train_param = self.network.all_params
		train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss_, var_list=train_param)
		
		start_time = time.time()
		print('Start Training CNN at ', time.asctime(time.localtime(start_time)))
		self.sess.run(tf.global_variables_initializer())
		
		for epoch in range(self.n_epoch):
			epoch_time = time.time()
			total_loss, n_batch = 0, 0
			for X_batch, y_batch in tl.iterate.minibatches(X_train, y_train, self.batch_size, shuffle=True):
				batch_loss, _ = self.sess.run([loss_, train_op], feed_dict={self.inputs: X_batch, self.labels: y_batch})
				total_loss += batch_loss; n_batch += 1
			print("Epoch %d of %d, loss %f" % (epoch + 1, self.n_epoch, total_loss / n_batch))
			print("Epoch %d takes %d seconds" % (epoch + 1, time.time() - epoch_time))
			
			if (epoch + 1) % 50 == 0:
				tl.files.save_npz(self.network.all_params , name='model_cnn' + str(epoch + 1) + '.npz')
		
		end_time = time.time()
		print('Training takes %d seconds' % (end_time - start_time))


	def predict(self, X):
		pred_y = self.sess.run(self.pred_, feed_dict={self.inputs: X})
		return pred_y
