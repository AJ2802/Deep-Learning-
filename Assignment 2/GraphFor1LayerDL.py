from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10
train_subset = 10000

batch_size = 128
num_steps = 3001
hidden_nodes = 1024
def buildingCompGraph(valid_dataset, test_dataset):

	graph = tf.Graph()
	
	with graph.as_default():
	
		#Input data. For the training data, we use a placeholderthat with be fed
		# at run time with a training minibatch
		
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
		
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
		
		#Variables to the 1st layer
		weights1 = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_nodes]));
		biases1 = tf.Variable(tf.zeros([hidden_nodes]))
		
		#Training computation in 1st layer
		logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
		activation = tf.nn.relu(logits1)
		
		#Variables to the 2nd layer
		weights2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
		biases2 = tf.Variable(tf.zeros(num_labels))
		
		#Training computation in 2nd layer
		logits2 = tf.matmul(activation, weights2) + biases2
		loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits2))
		
		# Optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
		
		#Predictions for the training, validation, and test data
		train_prediction = tf.nn.softmax(logits2);
		activation_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1);
		valid_prediction = tf.nn.softmax(tf.matmul(activation_valid, weights2) + biases2);
		activation_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1);
		test_prediction = tf.nn.softmax(tf.matmul(activation_test, weights2) + biases2);
		
		return graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction
