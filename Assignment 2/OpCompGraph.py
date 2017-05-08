from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10
train_subset = 10000

num_steps = 801

def accuracy(predictions, labels):
	return (100.0*np.sum(np.argmax(predictions, 1) == np.argmax(labels,1))/predictions.shape[0])
	
def graphRun(graph, optimizer, loss, train_labels, valid_labels, test_labels, train_prediction, valid_prediction, test_prediction):
	with tf.Session(graph= graph) as session:
		# This is a one-time operations which ensures the parameters get initialized as
		# described in the graph: random weights for the matrix, zeros for the
		# biases.
	
		tf.global_variables_initializer().run() #implement initialization
		print("Initialized")
		for step in range(num_steps):
			# Run the computations. We tell .run() that we want to run the optimizer,
			# and get the loss value and the training predictions returned as numpy
			# arrays [Note: train_prediction is just an array]
		
			_, l, predictions = session.run([optimizer, loss, train_prediction]) 
			#I do not think entry in prediction is just zero or one. I think prediction is a matrix and each row, an entry is between 0 and 1.
			#l is a loss value
			#prediction is a prediction matrix
			if (step % 100 == 0):
				print('Loss at step %d: %f' % (step, l))
				print('Training accuracy: %.1f%%' %accuracy(predictions, train_labels[:train_subset,:]))
			    #%% mean print "%"
			    
				#Calling .eval() on valid_prediction is basically like calling run(), but
				#just to get that one numpy array (I think valid_prediction.eval() is a matrix, same as test_prediction.eval()). Note that it recomputes all its graph
				#dependencies.
				print('Validation accuracy: %.1f%%' %accuracy(valid_prediction.eval(), valid_labels))
		print('Test accuracy: %.1f%%' %accuracy(test_prediction.eval(), test_labels))
