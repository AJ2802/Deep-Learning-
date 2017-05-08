#With gradient descent training, even this much data is prohibitive
#Subset the training data for faster turnaround

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import Accuracy

image_size = 28
num_labels = 10
train_subset = 10000

batch_size = 128
num_multiplier = 1;
num_steps = 3001;
partition = 20;
multiplier_range = np.logspace(-4,0,partition) # or np.logspace(-4,1,partition)
multiplier_valid = np.zeros((partition,3));
def GraphForRegLog(train_dataset, train_labels, valid_dataset, test_dataset):
	graph = tf.Graph()
	with graph.as_default():
	
		# Input data
		# Load the training, validation and test data into constant that are
		# attached to the graph
		
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size*image_size));
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		multiplier = tf.placeholder(tf.float32);
		
		tf_valid_dataset = tf.constant(valid_dataset);
		tf_test_dataset = tf.constant(test_dataset);
		
		
		
		#Variables
		weight = tf.Variable(tf.truncated_normal([image_size*image_size, num_labels]))
		biases = tf.Variable(tf.zeros([num_labels]))
		
		#Training Computation
		logits = tf.matmul(tf_train_dataset, weight) + biases
		regularization = multiplier*(tf.nn.l2_loss(weight))
		lossWithoutRegularization = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits))
		lossWithRegularization = lossWithoutRegularization + regularization
		
		#Optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(lossWithRegularization);
		
		#Predictions for the training, validation, and test data
		train_prediction = tf.nn.softmax(logits) + regularization;
		valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weight) + biases) ;
		test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weight) + biases) ;
		
	return graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction
	
def graphRun(graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels):
	
	for index, multiplier_value in enumerate(multiplier_range): 
		with tf.Session(graph=graph) as session:
			tf.global_variables_initializer().run();
			max_valid_acc = float('-inf');
			minStep = 0;
			print();
			print("Initialized");
			for step in range(num_steps):
				# Pick an offset within the training data, which has been randomized.
				# Note: we could use better randomization across epochs.
				offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
			
				# Generate a minibatch
				batch_data = train_dataset[offset:(offset + batch_size),:]
				batch_labels = train_labels[offset:(offset+batch_size),:]
			
				# Prepare a dictionary telling the session where to feed the minibatch
				# The key of the dictionary is the placeholder node of the graph to be fed,
				# and the value is the numpy array to feed it.
				
				feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, multiplier: multiplier_value};
			
				_, l, predictions = session.run([optimizer, lossWithRegularization, train_prediction], feed_dict = feed_dict)
	

				
				if ( step % 500 == 0):
					print("offset %d" %offset, " multiplier %.10f" %multiplier_value);
					print("Minibatch loss at step %d: %f" %(step, l))
					print("Minibatch accuracy: %.1f%%" % Accuracy.accuracy(predictions, batch_labels))
					valid_acc = Accuracy.accuracy(valid_prediction.eval(), valid_labels)
					print("Validation accuracy: %.1f%%" %valid_acc )
					
					
					
					if valid_acc > max_valid_acc:
						max_valid_acc = valid_acc;
						minStep = step
		
		multiplier_valid[index,:] = [multiplier_value, max_valid_acc, step];
		
	print("index [multiplier_value, max_valid_acc, step]");
	for index in range(partition):
		print("index: %d, multiplier value: %.10f, max_valid_acc: %.2f%%, step: %d" %(index,(float)(multiplier_valid[index,0]), (float)(multiplier_valid[index,1]), (int)(multiplier_valid[index,2])));

	plt.semilogx(multiplier_valid[:,0], multiplier_valid[:,1]);
	plt.grid(True);
	plt.title("Valdaition Accuracy against multiplier values Problem 1 LogisticWithReg")
	plt.show();

	
	opindex = np.argmax(multiplier_valid[:,1]);
	opMultiplier_value  = multiplier_valid[opindex, 0];
	opStep = multiplier_valid[opindex, 2];		
	
	print("The optimal multiplier is %.10f" %opMultiplier_value)
	print("")
	#For the test case
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run();
		print("Initialized");
		for step in range((int)(opStep)):
			# Pick an offset within the training data, which has been randomized.
			# Note: we could use better randomization across epochs.
			offset = (step*batch_size) % (train_labels.shape[0] - batch_size)
			
			# Generate a minibatch
			batch_data = train_dataset[offset:(offset + batch_size),:]
			batch_labels = train_labels[offset:(offset+batch_size),:]
			
			# Prepare a dictionary telling the session where to feed the minibatch
			# The key of the dictionary is the placeholder node of the graph to be fed,
			# and the value is the numpy array to feed it.
			
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, multiplier: opMultiplier_value};
			
			session.run([optimizer], feed_dict = feed_dict)
									

		print("Test accuracy: %.1f%%" % Accuracy.accuracy(test_prediction.eval(), test_labels))
