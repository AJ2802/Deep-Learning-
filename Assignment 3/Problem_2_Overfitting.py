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
factor = 5;
def graphRun(graph, optimizer, lossWithReg, multiplier, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels):

	batch_size = 128
	num_training_examples = factor*128;
	small_train_dataset = train_dataset[0:num_training_examples]; 
	small_train_labels = train_labels[0:num_training_examples];

	optimal_multiplier = 0.0020691381; #obtained in Q1
	graphRunDL1Layer.graphRunDL1Layer(graph, optimizer, lossWithReg, tf_train_dataset, tf_train_labels, small_train_dataset, small_train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels,  multiplier, optimal_multiplier);
	"""with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run();
		print("Initialized");
		
		for step in range(num_steps):
		
			# Pick an offset within training data, which has been randomized.
			# Note: we could use better randomization across epochs.
			
			offset = (step * batch_size) % (small_train_labels.shape[0] - batch_size)
			
			# Generate a minibatch
			
			batch_data = small_train_dataset[offset : offset + batch_size, :];
			batch_labels = small_train_labels[offset : offset + batch_size, :];
			optimal_multiplier = 0.0020691381; #obtained in Q1
			feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, multiplier: optimal_multiplier}
			
			_, l ,predictions = session.run([optimizer, lossWithReg, train_prediction], feed_dict = feed_dict)
			
			if (step % 500 == 0):
				print("offset %d" %offset);
				print("Minibatch loss at step %d: %f" %(step, l))
				print("Minibatch accuracy: %.1f%%" % Accuracy.accuracy(predictions, batch_labels));
				print("Validation accuracy: %.1f%%" % Accuracy.accuracy(valid_prediction.eval(), valid_labels));
		print("Test accuracy: %.1f%%" % Accuracy.accuracy(test_prediction.eval(), test_labels));"""
