#With gradient descent training, even this much data is prohibitive
#Subset the training data for faster turnaround

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10
train_subset = 10000

def buildingCompGraph(train_dataset, train_labels, valid_dataset, test_dataset):
	graph = tf.Graph();
	with graph.as_default():

		# Input data.
		# Load the training, validation and test data into constants that are
		# attached to the graph
	
		tf_train_dataset = tf.constant(train_dataset[:train_subset,:])
		tf_train_labels = tf.constant(train_labels[:train_subset])
		tf_valid_dataset = tf.constant(valid_dataset)
		tf_test_dataset = tf.constant(test_dataset)
	
		# Variables.
		# These are the parameters that we are going to be training. The weight
		# matrix will be initialized using random values following a (truncated)
		# normal distribution. The biases get initialized to zero.
	
		weights = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
		biases = tf.Variable(tf.zeros([num_labels]))
	
		# Training computation
		# We multiply the inputs with the weight matrix, and add biases. We compute
		# the softmax and cross-entropy (it's one operation in TensorFlow, because
		# it's very common, and it can be optimized). We take the average of this
		# cross-entropy across all training examples: that's our loss.
	
		logits = tf.matmul(tf_train_dataset, weights)+biases;#I dunno how matmul work in tensorflow
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))
	
		# Optimizer
		# We are going to find the minimum of this loss using gradient descent
	
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)   # 0.5 is a learning rate
	
		#Predictions for the training, validation, and test data
		#These are not part of training, but merely here so that we report
		# accuracy figures as we train.
	
		train_prediction = tf.nn.softmax(logits) #Transform original data to another data via softmax function
		valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
		test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
	
		return graph, optimizer, loss, train_prediction, valid_prediction, test_prediction
	
	
	
	
	
