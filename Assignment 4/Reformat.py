from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size,image_size,num_channels)).astype(np.float32)
	# Map 1 to [0.0, 1.0, 0.0,...], 2 to [0.0, 0.0, 1.0,...]\
	#Check label dimension is a row vector or column vector
	print("label in Reformat dim", labels.shape);
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels
	
