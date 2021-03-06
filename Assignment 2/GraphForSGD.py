from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10
train_subset = 10000

batch_size = 128


def buildingCompGraph(valid_dataset, test_dataset):
	
	graph = tf.Graph()
	
	with graph.as_default():
	
		# Input data. For the training data, we use a placeholder that will be fed
		# at run time with a training minibatch.
	
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
	
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
	
		# Variables
		weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
		biases = tf.Variable(tf.zeros([num_labels]))
	
		# Training computation
		logits = tf.matmul(tf_train_dataset, weights) + biases
		loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels,logits = logits))
	
		# Optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	
		# Predictions for the training, validation, and test data.
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
		test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
	
	return graph, optimizer, loss, tf_train_dataset, tf_train_labels,train_prediction, valid_prediction, test_prediction


