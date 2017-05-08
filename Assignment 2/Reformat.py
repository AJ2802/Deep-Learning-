from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

image_size = 28
num_labels = 10

#data as a flat matrix (1 d array)
#labels as float 1-hot encodings
def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
	#Map 0 to [1.0, 0.0, 0.0, ...] , 1 to [0.0, 1.0, 0.0, ...]
	labels = (np.arange(num_labels)==labels[:,None]).astype(np.float32)
	return dataset, labels


