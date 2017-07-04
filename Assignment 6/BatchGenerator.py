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
		segment = self._text_size // batch_size
		self._cursor = [ offset*segment for offset in range(batch_size)] #[1st segment, 2nd segment,..., batch_size^th segment] #This serves like a shuffle coz sequential sampling is not a good idea? So each row in a batch is a new training example.
		self._last_batch = self._next_batch()

	
	def _next_batch(self):
		"""Generate a single batch from the current cursor position in the data."""
		batch = np.zeros(shape = (self._batch_size, self._vocabulary_size), dtype = np.float)
		for b in range(self._batch_size):
			batch[b, Transformation.char2id(self._text[self._cursor[b]], self._first_letter)] = 1.0; #self._text[self._cursor[b]] is a letter in text at the location self._cursor[b].
			#the above line indicate the letter in text at at the location self._cursor[b] by label 1 in the id(letter)^th column in bth row of the matrix batch.
			self._cursor[b] = (self._cursor[b] + 1)%self._text_size #update the cursor in the bth segment to point to the next letter in the segment.
		return batch #A batch contain one letter of each of the batch_size segment in each row.
		
	def next(self):
		"""Generate the next array of batches from the data. The array consists of the last batch of the previous array, followed by num_unrollings new ones."""
		batches = [self._last_batch]
		for step in range(self._num_unrollings): #num_unrollings = 10;
			batches.append(self._next_batch())
			#batch[b, char2id(self._text[self._cursor[b]])] has the same row but self._cursor in the bth segment pointing to the next word in the segment.
		self._last_batch = batches[-1]
		return batches #batches is [last_batch, new 1st batch,...., new num_unrollings^th batch]. batches[:][i,:] form a meaningful sentence in a i^th segment.
		#Note each batch in batches contains a letter in each segment. Each batch is a matrix of batch_size number of row and |{a-z, ' '}| number of column.
		
def characters(probabilities, first_letter):
	"""Turn a 1-hot encoding or a probability distribution over the possible characters back into its (most likely) character representation."""
	#probabilities is matrix which each row contains 26 entries (' ' +'a-z')
	#declare first_letter later
	return [Transformation.id2char(c, first_letter) for c in np.argmax(probabilities, 1)] #return a letter which is in high probability in each row. And the output is a column vector of letters.
		
def batches2string(batches, first_letter):
	"""Convert a sequence of batches back into their (most likely) string representation."""
	s = ['']*batches[0].shape[0] #number of row in a batch = batch_size
	
	for b in batches:
		s = [''.join(x) for x in zip(s, characters(b, first_letter))]
		# characters(b, first_letter) is a vector of batch_size number of entries
		#x is in a form of (one of entries in the list s, one character in characters(b, first_letter))
		#join is an operator to joint the entry in the list s and the character in x together.
	return s # s is a list of batch_size entries and each ith entry transform batches to characters (maybe words) in the ith segment. 
		

		
