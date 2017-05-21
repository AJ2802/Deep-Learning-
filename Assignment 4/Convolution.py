from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import Accuracy
import tensorflow as tf

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

def convolution(valid_dataset, test_dataset):
	with graph.as_default():

		#input data
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_dataset = tf.constant(valid_dataset);
		tf_test_dataset = tf.constant(test_dataset);
	
		#Variables:
		layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = 0.1)) #how to reduce from num_channels to depth
		layer1_biases = tf.Variable(tf.zeros([depth])) # why constant is 0
		layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = 0.1)) # how to reduce from depth to depth
		layer2_biases = tf.Variable(tf.constant(1.0, shape =[depth]))   #why constant is 1
		layer3_weights = tf.Variable(tf.truncated_normal([image_size//4*image_size//4*depth , num_hidden], stddev = 0.1))#why //4
		layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
		layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev = 0.1))
		layer4_biases = tf.Variable(tf.constant(1.0, shape = [num_labels]))
	

		#Model
		def model(data):
			conv = tf.nn.conv2d(data, layer1_weights, [1,2,2,1], padding ='SAME')
			hidden = tf.nn.relu(conv+layer1_biases)
			conv = tf.nn.conv2d(hidden, layer2_weights, [1,2,2,1], padding = 'SAME')
			hidden = tf.nn.relu(conv + layer2_biases)
			shape = hidden.get_shape().as_list()
			reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]]) #shape[0] ~~batch_size, shape[1]~~out_height, shape[2]~~out_width, shape3~~depth
			hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
			return tf.matmul(hidden, layer4_weights) + layer4_biases
	
		# Training computation
		logits = model(tf_train_dataset)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))
	
	# Optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	#Predictions for the training, validation, and test data
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
		test_prediction = tf.nn.softmax(model(tf_test_dataset))
	
	return graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction
	
	

	
	
def graphRun(graph, optimizer, loss, train_dataset, train_labels, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction, valid_labels, test_labels):
	
	num_steps = 1001
	
	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized')
		for step in range(num_steps):
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset : (offset + batch_size), :, :, :]
			batch_labels = train_labels [offset : (offset + batch_size), :]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
			if (step % 50 == 0):
				print('Minibatch loss at step %d: %f' %(step, l))
				print('Minibatch accuracy: %.1f%%' % Accuracy.accuracy(predictions, batch_labels))
				print('Validation accuracy: %.1f%%' % Accuracy.accuracy(valid_prediction.eval(), valid_labels))
		print('Test accuracy: %.1f%%' % Accuracy.accuracy(test_prediction.eval(), test_labels));
			
