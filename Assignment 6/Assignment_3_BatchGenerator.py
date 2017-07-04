##Reference : https://github.com/hankcs/udacity-deep-learning/blob/master/6_lstm.py

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import Transformation

class BatchGenerator(object):
	def __init__(self, text, batch_size, num_unrollings, vocabulary_size, first_letter):
		self._text = text; 
		self._text_size = len(text)
		self._batch_size = batch_size #batch_size = 64
		self._num_unrollings = num_unrollings
		self._vocabulary_size = vocabulary_size
		self._first_letter = first_letter
		segment = self._text_size // num_unrollings #??
		self._cursor = [ offset*segment for offset in range(batch_size)] #[1st segment, 2nd segment,..., batch_size^th segment] #This serves like a shuffle coz sequential sampling is not a good idea? So each row in a batch is a new training example.
		self._last_batch = self._next_batch(0)
		
	def _next_batch(self, step):
		"""Generate a single batch from the current cursor position in the data."""
		batch =''
		for b in range(self._num_unrollings):
			self._cursor[step] %= self._text_size
			batch += self._text[self._cursor[step]]
			self._cursor[step] += 1
		return batch #batch store a string of _num_unrolling many characters
		
	
	def next(self):
		"""Generate the next array of batches from the data. The array consists of the last batch of the previous array, followed by num_unrollings new ones."""
		batches = [self._last_batch]
		for step in range(self._batch_size):
			batches.append(self._next_batch(step))
		self._last_batch = batches[-1]
		return batches #batches is [last_batch, new 1st batch,...., new batch_size^th batch]. batches[i][:] means i^th batch which is a sentence of num_unrollings many words in the i^th segment
		
	def characters(probabilities, first_letter):
		"""Turn a 1-hot encoding or a probability distribution over the possible characters back into its (most likely) character representation."""
		#probabilities is matrix which each row contains 26 entries (' ' +'a-z')
		#declare first_letter later
		return [Transformation.id2char(c, first_letter) for c in np.argmax(probabilities, 1)] #return a letter which is in high probability in each row. And the output is a column vector of letters.
	
	
	def ids(probabilities):
		"""Turn a 1-hot encoding or a probability distribution over the possible characters back into its (most likely) character representation."""
		return [str(c)  for c in np.argmax(probabilities, 1)]
		#what is the argmax of probabilities?
	
	
	def batches2string(batches, first_letter):
		"""Convert a sequence of batches back into their (most likely) string representation."""
		s = ['']*batches[0].shape[0] #number of row in a batch = batch_size
	
		for b in batches:
			s = [''.join(x) for x in zip(s, ids(b))]
			# characters(b, first_letter) is a vector of batch_size number of entries
			#x is in a form of (one of entries in the list s, one character in characters(b, first_letter))
			#join is an operator to joint the entry in the list s and the character in x together.
		return s # s is a list of batch_size entries and each ith entry transform batches to characters (maybe words) in the ith segment. 
	
def rev_id(forward, first_letter):
	temp = forward.split(' ') # temp is a list of elements and each element is a word.
	backward = []
	for i in range(len(temp)):
		backward += temp[i][::-1]+' ' #temp[i] is a list [::-1] mean starts from the end toward to the start by taking -1 step each
	return list(map(lambda x: Transformation.char2id(x, first_letter), backward[:-1])) #backward[:-1] is from the begining to the end
	#The output is a list of characters (characters of all word in the list is in reverse order.
	
	
	
	
