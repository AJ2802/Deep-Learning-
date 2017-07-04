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
		self._first_letter = first_letter
		self._text_bigram = self._text_size//2
		self._vocabulary_size = vocabulary_size
		segment = self._text_bigram // batch_size
		self._cursor = [ offset*segment for offset in range(batch_size)] #[1st segment, 2nd segment,..., batch_size^th segment] #This serves like a shuffle coz sequential sampling is not a good idea? So each row in a batch is a new training example.
		self._last_batch = self._next_batch()
		

	
	def _next_batch(self):
		"""Generate a single batch from the current cursor position in the data."""
		batch = np.zeros(shape = self._batch_size, dtype = np.int32);

		for b in range(self._batch_size):
			char_pointer = self._cursor[b] * 2
			char1st = Transformation.char2id(self._text[char_pointer], self._first_letter)
			if char_pointer == self._text_size-1:
				char2nd = 0;
			else:
				char2nd = Transformation.char2id(self._text[char_pointer + 1], self._first_letter)
	
			batch[b] = char1st * self._vocabulary_size + char2nd
			
			self._cursor[b] = (self._cursor[b] + 1) % self._text_bigram

		return batch 
		
	def next(self):
		"""Generate the next array of batches from the data. The array consists of the last batch of the previous array, followed by num_unrollings new ones."""
		batches = [self._last_batch]
		for step in range(self._num_unrollings): #num_unrollings = 10;
			batches.append(self._next_batch())
		self._last_batch = batches[-1]
		return batches #batches is a list of batch and the length of batches is unrollings and each batch is a vector of size batch_size and each ith entry in the vector is an integer representing a bigram of text in the ith segment pointed by cursor i.

def bi2str(encoding, vocabulary_size, first_letter):
	quotient = encoding // vocabulary_size;
	remainder = encoding % vocabulary_size;
	bigram = Transformation.id2char(quotient, first_letter) + Transformation.id2char(remainder, first_letter)
	return bigram

def bigrams(encodings, vocabulary_size, first_letter):
	return [bi2str(encoding, vocabulary_size, first_letter) for encoding in encodings]
	
def bibatches2string(batches, vocabulary_size, first_letter):
	"""Convert a sequence of batches back into their (most likely) string representation."""
	s = ['']*batches[0].shape[0] #number of row in a batch = batch_size
	for encodings in batches:
	#encodings is a batch which is a vector of batch_size and each ith entry in the vector is an integer representing a bigram of text in the ith segment pointed by cursor i.
		s = [''.join(x) for x in zip(s, bigrams(encodings, vocabulary_size, first_letter))]
	return s 



