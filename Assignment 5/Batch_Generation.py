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
def generate_batch(data, batch_size, num_skips, skip_window):
	global data_index #note data_index should not initialized to be zero coz we want our batch of words has some randomness.
	assert batch_size % num_skips == 0 #assert means check
	assert num_skips <= 2 * skip_window
	
	batch = np.ndarray(shape=(batch_size), dtype = np.int32)
	labels = np.ndarray(shape=(batch_size,1), dtype = np.int32)
	
	span = 2 * skip_window + 1 #[skip_window target skip_window] note (left, target word, right)
	buffer = collections.deque(maxlen=span)
	
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	
	for i in range(batch_size // num_skips):
		target = skip_window # target label at the center of the buffer
		targets_to_avoid = [ skip_window ]
		for j in range(num_skips): #note num_skip has to be <= 2*skip_window, otherwise nothing is added to batch and labels
			while target in targets_to_avoid:
				target = random.randint(0, span -1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] =buffer[skip_window]
			labels[i * num_skips +j, 0] = buffer[target]   
			#(source = centered word, label = word near the centered word)
		
		buffer.append(data[data_index]) # give another source and another label word
		data_index = (data_index + 1) % len(data)
		
		#buffer contains (source word (word next to the target wor), target word) and the target word will renew in every num_skips times of addition.
	return batch, labels
	

