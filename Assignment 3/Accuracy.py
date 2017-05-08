from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

def accuracy(predictions, labels):
	return (100*np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0]);
