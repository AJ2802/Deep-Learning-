#Reference : https://github.com/hankcs/udacity-deep-learning/blob/master/6_lstm.py

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

def logprob(predictions, labels):
	""" Log-probability of the true labels in a predicted batch. """
	#Check this example
	#p = np.matrix([[1,2,3],[4,5,6]])
	#p[p<3]= 100.
	#Output is [100 100 3]
	#			[4   5  6]
	predictions[predictions < 1e-10] = 1e-10 #1e-10 is assigned to an entry of the predictions matrix as long as the entry is <1e-10. It avoids a numerical error of log 0.
	return np.sum(np.multiply(labels, -np.log(predictions)))/labels.shape[0] #a kind of cross entropy function
	
def sample_distribution(distribution):
	""" Sample one element from a distribution assumed to be an array of normalized probabilities."""
	r = random.uniform(0,1)
	s = 0
	for i in range(len(distribution)):
		s += distribution[i]
		if s >= r:
			return i
	return len(distribution) - 1 #Does this line run in runtime?
	
def sample(prediction, vocabulary_size):
	""" Turn a (column) prediction into 1-hot encoded samples."""
	"""not absolute/deterministic way to turn probability vector to a 1-hot encoded vector"""
	p = np.zeros(shape=[1, vocabulary_size], dtype = np.float)
	p[0, sample_distribution(prediction[0])] = 1.0
	return p

def one_hot_voc(prediction, size):
	p = np.zeros(shape = [1, size], dtype = np.float)
	p[0, prediction[0]] = 1.0
	return p
	
def random_distribution(vocabulary_size):
	""" Generate a random column of probabilities."""
	b = np.random.uniform(0.0, 1.0, size = [1, vocabulary_size])
	return b/np.sum(b,1)[:, None] #[:,None] makes a row to be a column
