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
num_steps = 50001 #10001

hidden_nodes_1stLayer = 1024
hidden_nodes_2ndLayer = 512
hidden_nodes_3rdLayer = 256


"""
After several times of trial and error, I conclude that it is very hard to train a deep network when more layers are added.
In order to train a multiple layers of neural network well, there are two things we should pay attention to:
(1) initialization of weights: magnitude of activation values decreases when the propagation moves from a first layer to a last layer.
Therefore, the magnitude of gradient of loss function w.r.t to weight goes down from a first layer to a last layer. 
Hence, the distribution of weight collpases from a first layer to a last layer. Hence, we should initialize weights by using normal distribution with different std in each layer, e.g. a Xavier initialization initializes in ith layer is N(0,\sqrt(2/m_i)) and m_i is the is the dimension of (i-1)th layer..
(2) we need to use smaller learning rate to train the network. Indeed, when layers increases, the loss function will has many local minimums. Hence, otpimizer is easily got trapped in a local minimum region.
(3) change the batch_size do not help training too much in this case.
(4) Regularization can help training more than the dropout in this case.
(5) Increase the training iteration can also helps to improve accuracy.
"""
def GraphForSlowDecayLR2LayerDropout(train_dataset, train_labels, valid_dataset, test_dataset):
	graph = tf.Graph();
	
	with graph.as_default():
	
		# Input data. For the training data, we use a placeholder that will be fed.
		# at run time with a train in minibatch
		
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size))
		tf_train_labels = tf.placeholder(tf.float32, shape = (batch_size, num_labels))
		
		multiplier = tf.placeholder(tf.float32);
		
		tf_valid_dataset = tf.constant(valid_dataset);
		tf_test_dataset = tf.constant(test_dataset);
		
		#Variables to the 1st layer
		weight1 = tf.Variable(tf.truncated_normal([image_size*image_size, hidden_nodes_1stLayer], stddev = np.sqrt(2.0 / (image_size * image_size))))
		biases1 = tf.Variable(tf.zeros(hidden_nodes_1stLayer));
		
		#Training computation in the 1st layer	
		logits1 = tf.matmul(tf_train_dataset, weight1) + biases1;
		activation1 = tf.nn.relu(logits1);
		
		
		#Variables to the 2nd layer
		weight2 = tf.Variable(tf.truncated_normal([hidden_nodes_1stLayer, hidden_nodes_2ndLayer], stddev = np.sqrt(2.0 / (hidden_nodes_1stLayer))))
		biases2 = tf.Variable(tf.zeros(hidden_nodes_2ndLayer))
		
		#Training computation in the 2nd layer
		logits2 = tf.matmul(activation1, weight2) + biases2
		activation2 = tf.nn.relu(logits2)
		
		
		#Variables to the 2nd layer
		weight3 = tf.Variable(tf.truncated_normal([hidden_nodes_2ndLayer, hidden_nodes_3rdLayer], stddev = np.sqrt(2.0 / (hidden_nodes_2ndLayer))))
		biases3 = tf.Variable(tf.zeros(hidden_nodes_3rdLayer))
		
		#Training computation in the 2nd layer
		logits3 = tf.matmul(activation2, weight3) + biases3
		activation3 = tf.nn.relu(logits3)
		
		#Variables to the 3rd layer
		weight4 = tf.Variable(tf.truncated_normal([hidden_nodes_3rdLayer, num_labels], stddev = np.sqrt(2.0 / (hidden_nodes_3rdLayer))))
		biases4 = tf.Variable(tf.zeros(num_labels));
		
		#Training computation in the 3rd layer
		logits4 = tf.matmul(activation3, weight4) + biases4
		Reg = multiplier*(tf.nn.l2_loss(weight1)+ tf.nn.l2_loss(biases1)+ tf.nn.l2_loss(weight2) + tf.nn.l2_loss(biases2)+tf.nn.l2_loss(weight3)+ tf.nn.l2_loss(biases3)+tf.nn.l2_loss(weight4)+ tf.nn.l2_loss(biases4))
		lossWithReg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits4)) + Reg;
		
		#Optimizer
		global_step = tf.Variable(0) # count the number of steps taken
		learning_rate = tf.train.exponential_decay(0.001, global_step, 10000, 0.5, staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(lossWithReg, global_step = global_step);
		
		#Prediction for the training, validation and test data
		train_prediction =tf.nn.softmax(logits4)

		activation1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weight1) + biases1)
		activation2_valid =tf.nn.relu(tf.matmul(activation1_valid, weight2) + biases2)
		activation3_valid =tf.nn.relu(tf.matmul(activation2_valid, weight3) + biases3)
		valid_prediction =tf.nn.softmax(tf.matmul(activation3_valid, weight4) + biases4)
		
		activation1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weight1) + biases1)
		activation2_test =tf.nn.relu(tf.matmul(activation1_test, weight2) + biases2)
		activation3_test =tf.nn.relu(tf.matmul(activation2_test, weight3) + biases3)
		test_prediction =tf.nn.softmax(tf.matmul(activation3_test, weight4) + biases4)
		
		return graph, optimizer, lossWithReg, multiplier, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction
	
	
def graphRun(graph, optimizer, lossWithReg, multiplier, tf_train_dataset,  tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels):

	optimal_multiplier = 0.0020691381; #obtained in Q1
	graphRunDL1Layer.graphRunDL1Layer(graph, optimizer, lossWithReg, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels, multiplier=multiplier, optimal_multiplier=optimal_multiplier, num_steps = num_steps, batch_size = batch_size);

		 
