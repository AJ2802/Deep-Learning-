from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import Accuracy
import graphRunDL1Layer

image_size = 28
num_labels = 10

batch_size = 128
num_steps = 3001
hidden_nodes = 1024

keep_prob_hidden = 0.5
keep_prob_input = 1

#Note dropout method is to dropout hidden nodes or input nodes but not connection weights!
def GraphForDropoutDL1Layer(train_dataset, train_labels, valid_dataset, test_dataset):

	graph = tf.Graph()
	
	with graph.as_default():
	
		# Input data. For the training data, we use a placeholder that will be fed
		# at run time with a train in minibatch
		
		tf_train_dataset = tf.placeholder(tf.float32, shape = (batch_size, image_size*image_size))
		tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
		
		tf_valid_dataset = tf.constant(valid_dataset);
		tf_test_dataset = tf.constant(test_dataset);
		multiplier = tf.constant(0.0020691381); #obtained in Q1
		
		tf_train_datasetWithDropOut = tf.nn.dropout(tf_train_dataset, keep_prob = keep_prob_input) #this delete features of an inputs need another prob #thought is dropout needed?
		#Variables to the 1st layer
		weight1 = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_nodes]))
		biases1 = tf.Variable(tf.zeros(hidden_nodes));
		
		#Training computation in the 1st layer
		logits1 = tf.matmul(tf_train_datasetWithDropOut, weight1) + biases1
		activation1WithDropout = tf.nn.dropout(tf.nn.relu(logits1), keep_prob = keep_prob_hidden);
		activation1WithDropout = tf.nn.relu(logits1);
		

		
		#Variables to the 2nd layer
		weight2 = tf.Variable(tf.truncated_normal([hidden_nodes, num_labels]))
		biases2 = tf.Variable(tf.zeros(num_labels))
		
		#Training computation in the 2nd layer
		logits2 = tf.matmul(activation1WithDropout, weight2) + biases2
		
		lossWithDropout = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits2));		
		
		#Optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(lossWithDropout);
		
		#Prediction for the training, validation and test data
		activation1 = tf.nn.relu(tf.matmul(tf_train_dataset, weight1) + biases1)
		train_prediction = tf.nn.softmax(tf.matmul(activation1, weight2) + biases2)
		
		activation_valid1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weight1) + biases1);
		valid_prediction = tf.nn.softmax(tf.matmul(activation_valid1,weight2) + biases2);
		activation_test1 = tf.nn.relu(tf.matmul(tf_test_dataset, weight1) + biases1);
		test_prediction = tf. nn.softmax(tf.matmul(activation_test1,weight2) + biases2)

		return graph, optimizer, lossWithDropout, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction
		
		
def graphRun(graph, optimizer, lossWithDropout,  tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels):

	graphRunDL1Layer.graphRunDL1Layer(graph, optimizer, lossWithDropout,  tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels);
