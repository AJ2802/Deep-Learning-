# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 23:56:54 2017

@author: AJ
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def reload(pickle_file):
	with open(pickle_file, 'rb') as f: #b means in binary , be careful if the file is in jpeg or exe
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		valid_dataset = save['valid_dataset']
		valid_labels = save['valid_labels']
		test_dataset = save['test_dataset']
		test_labels = save['test_labels']
		del save #hint to help gc free up memory
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
	
	
