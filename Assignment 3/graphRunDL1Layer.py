from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import Accuracy

image_size = 28
num_labels = 10

batch_size = 128
num_steps = 3001


def graphRunDL1Layer(graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels, multiplier=None, optimal_multiplier=None, num_steps = num_steps, batch_size = batch_size):

	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run();
		print("Initialized");
		
		for step in range(num_steps):
		
			# Pick an offset within training data, which has been randomized.
			# Note: we could use better randomization across epochs.
			
			offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
			
			# Generate a minibatch
			
			batch_data = train_dataset[offset : offset + batch_size, :]
			batch_labels = train_labels[offset : offset + batch_size, :]
			
			if multiplier==None: 
				feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			else:
				feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, multiplier: optimal_multiplier}
			
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
			
			if (step % 500 == 0):
				print("offset %d" %offset);
				print("Minibatch loss at step %d: %f" %(step, l))
				print("Minibatch accuracy: %.1f%%" % Accuracy.accuracy(predictions, batch_labels));
				print("Validation accuracy: %.1f%%" % Accuracy.accuracy(valid_prediction.eval(), valid_labels));
				
		print("Test accuracy: %.1f%%" %Accuracy.accuracy(test_prediction.eval(), test_labels));
