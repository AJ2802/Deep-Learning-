from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

data_index = 0;

#skip_window :  the number of words to the left and to the right of a target word
def generate_batch(data, batch_size, skip_window):
	global data_index #note data_index should not initialized to be zero coz we want our batch of words has some randomness.
	
	batch = np.ndarray(shape=(batch_size, skip_window*2), dtype = np.int32)
	labels = np.ndarray(shape=(batch_size,1), dtype = np.int32)
	
	span = 2 * skip_window + 1 #[skip_window target skip_window] note (left, target word, right)
	buffer = collections.deque(maxlen=span)
	
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	for i in range(batch_size):
		buffer_list = list(buffer);
		labels[i,0] = buffer_list.pop(skip_window);
		batch[i] = buffer_list;
		
		buffer.append(data[data_index]) # give another source and another label word
		data_index = (data_index + 1) % len(data)
		
		#buffer contains (source word (word next to the target wor), target word) and the target word will renew in every num_skips times of addition.
	return batch, labels
	
def graphForText(batch_size, embedding_size, skip_window, valid_size, valid_window, valid_examples, num_sampled, vocabulary_size):
	
	graph = tf.Graph()

	with graph.as_default(), tf.device('/cpu:0'):

		# Input data.
		# CBOW model: (source = words near centered words,  label = centered words)
		# For example, the dataset is "the quick brown fox jumped over the lazy dog"
		# If the window size is 1, then # skip-gram model try to predict 'quick' from 'the' and 'brown', 'brown' from 'quick' and 'fox'.
		# Therefore, DATASET becomes (['the' 'brown'], quick), (['quick', 'fox'], brown)... of (input, output) pairs. It is the DATASET we are going to train and make prediction.
		# Each word is represented as an integer (i,e. the page on the count where the word shows up)
		# Batch full of integers representating the source words near centered words
		# The other is a target centered word [ highly possible word given the source context words

		train_dataset = tf.placeholder(tf.int32, shape = [batch_size, 2*skip_window])
		train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
		valid_dataset = tf.constant(valid_examples, dtype = tf.int32)
	
		#Variables.
		# start with a random embedding matrix
		embeddings = tf.Variable( tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		# Weight and Baises is for softmax, use Xavier's initialization
		softmax_weights= tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0/ math.sqrt(embedding_size)))
		softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
	
		# Model.
		# Look up embeddings for inputs.
		# Look up a vector for each of the source words in the batch.
		embed = tf.nn.embedding_lookup(embeddings, train_dataset) # Note embedding_size = batch_size #embedding_lookup avoid changing of dimension of embeddings
		# Compute the softmax loss, using a sample of the negative labels each time.
		# Look inputs = tf.reduce_sum(embeds, 1) coz = we get the average vector after embedding 2*skip_window words
		loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = softmax_weights, biases = softmax_biases, inputs = tf.reduce_mean(embed, 1), labels = train_labels, num_sampled = num_sampled, num_classes = vocabulary_size))
	
		# Optimizer.
		# Note: The optimizer will optimize the softmax_weights AND the embeddings.
		# This is because the embeddings are defined as a variable quantity and the 
		# optimizer's 'minimize' method will by default modify all variable quantities
		# that contribute to the tensor it is passed.
		# See docs on 'tf.train.Optimizer.minimize()' for more details.
		optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
	
		# Compute the similarity between minibatch examples and all embeddings.
		# We use the cosine distance:
		# The validation set just to indicate which row is chosen so no tf.reduce_mean is required.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		# e.g. valid_dataset = [20, 11]
		# valid_embeddings should be 20th row and 11th row of the tranpose of the normalized embeddings
		# HERE
		similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
		
		return graph, train_dataset, train_labels, normalized_embeddings, valid_examples, optimizer, loss, similarity
		
def graphRun(graph, data, train_dataset, train_labels, batch_size, skip_window, optimizer, loss, normalized_embeddings, reverse_dictionary, valid_size, valid_examples , similarity):
	num_steps = 100001
	
	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized');
		average_loss = 0;
		for step in range(num_steps):
			batch_data, batch_labels = generate_batch(data, batch_size, skip_window)
			feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
			_, l = session.run([optimizer, loss], feed_dict = feed_dict)
			average_loss += l
			if step % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print('Average loss at step %d: %f' % (step, average_loss))
				average_loss = 0
			
			# note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 10000 == 0:
				sim = similarity.eval();
				
				for i in range(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]] #valid_word is the target word
					top_k = 8 # number of nearest neighbors
					nearest = (-sim[i,:]).argsort()[1: top_k+1]
					log = 'Nearest to %s:' % valid_word
					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)
		final_embeddings = normalized_embeddings.eval()
		
		return final_embeddings;
