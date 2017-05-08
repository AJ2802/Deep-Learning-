# -*- coding: utf-8 -*-

"""
Created on Sun Mar 26 17:35:25 2017

@author: AJ
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root='.' #Change me to store data elsewhere


np.random.seed(133)

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
