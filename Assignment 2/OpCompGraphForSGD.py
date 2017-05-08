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

def accuracy(predictions, labels):
	return (100.0*np.sum(np.argmax(predictions, 1) == np.argmax(labels,1))/predictions.shape[0])
	
def graphRun(graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels):
	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print("Initialized")
		for step in range(num_steps):
			# Pick an offset within the training data, which has been randomized.
			# Note: we could use better randomization across epochs.
			offset = (step*batch_size) % (train_labels.shape[0] - batch_size)   
			#offset is not in ascending order. You move around the whole train_dataset. Adv: add more variation in a train data. Remark: it is even better if offset is chosen randomly.
			
			# Generate a minibatch.
			batch_data = train_dataset[offset:(offset + batch_size), :]
			batch_labels = train_labels[offset:(offset + batch_size),:]
		
			# Prepare a dictionary telling the session where to feed the minibatch
			# The key of the dictionary is the placeholder node of the graph to be fed,
			# and the value is the numpy array to feed to it.
		
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		
			_,l,predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
			if ( step % 500 == 0):
				print("offset %d" % offset);
				print("Minibatch loss at step %d: %f" % (step, l))
				print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
				print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
			
		print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
