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

import Batch_Generation


def graphForText(batch_size, embedding_size, skip_window, num_skips, valid_size, valid_window, valid_examples, num_sampled, vocabulary_size):
	
	graph = tf.Graph()

	with graph.as_default(), tf.device('/cpu:0'):

		# Input data.
		# skip-gram model: (source = centered words, label = words near centered words)
		# For example, the dataset is "the quick brown fox jumped over the lazy dog"
		# If the window size is 1, then # skip-gram model try to predict 'the' and 'brown' from 'quick', 'quick' and 'fox' from 'brown'
		# Therefore, DATASET becomes (quick, the), (quick, brown), (brown, quick),... of (input, output) pairs. It is the DATASET we are going to train and make prediction.
		# Each word is represented as an integer (i,e. the page on the count where the word shows up)
		# Batch full of integers representating the source centered words
		# The other is a target words around the centered word [ highly possible word given the source context words

		train_dataset = tf.placeholder(tf.int32, shape = [batch_size])
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
		embed = tf.nn.embedding_lookup(embeddings, train_dataset) # Note embedding_size = batch_size
		# Compute the softmax loss, using a sample of the negative labels each time.
		loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = softmax_weights, biases = softmax_biases, inputs = embed, labels = train_labels, num_sampled = num_sampled, num_classes = vocabulary_size))
	
		# Optimizer.
		# Note: The optimizer will optimize the softmax_weights AND the embeddings.
		# This is because the embeddings are defined as a variable quantity and the 
		# optimizer's 'minimize' method will by default modify all variable quantities
		# that contribute to the tensor it is passed.
		# See docs on 'tf.train.Optimizer.minimize()' for more details.
		optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
	
		# Compute the similarity between minibatch examples and all embeddings.
		# We use the cosine distance:
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		# e.g. valid_dataset = [20, 11]
		# valid_embeddings should be 20th row and 11th row of the tranpose of the normalized embeddings
		similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
		
		return graph, train_dataset, train_labels, normalized_embeddings, valid_examples, optimizer, loss, similarity
		
def graphRun(graph, data, train_dataset, train_labels, batch_size, num_skips, skip_window, optimizer, loss, normalized_embeddings, reverse_dictionary, valid_size, valid_examples , similarity):
	num_steps = 100001
	
	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized');
		average_loss = 0;
		for step in range(num_steps):
			batch_data, batch_labels = Batch_Generation.generate_batch(data, batch_size, num_skips, skip_window)
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
