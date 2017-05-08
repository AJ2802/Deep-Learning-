# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:18:18 2017

@author: AJ
"""

#These are al the modules we ' ll be using Later. Make sure you can import them
#before proceeding further.

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

data_root='.' #Change me to store data elsewhere

import Download
import Unzip
import problem_1_pickle
import problem_2_image
import problem_3
import problem_4_randomize
import problem_5_DuplicatedData
import problem_6_logisticReg

# Config the matplotlib backend as plotting inline in IPython
#%matplotlib inline

print("Hello")
train_filename = Download.maybe_download('notMNIST_large.tar.gz', 247336696);
test_filename = Download.maybe_download('notMNIST_small.tar.gz', 8458043);

train_folders = Unzip.maybe_extract(train_filename);
test_folders = Unzip.maybe_extract(test_filename);

# Store train data and test data in binary format for fast masipulation.
train_datasets = problem_1_pickle.maybe_pickle(train_folders, 45000)
test_datasets = problem_1_pickle.maybe_pickle(test_folders, 1800)

problem_2_image.pickle_image(train_datasets, [0,1,2])
problem_2_image.pickle_image(test_datasets, [0,1,2])

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = problem_3.merge_datasets(train_datasets, train_size, valid_size)
_,_, test_dataset,test_labels = problem_3.merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = problem_4_randomize.randomize(train_dataset, train_labels)
test_dataset, test_labels = problem_4_randomize.randomize(test_dataset, test_labels)
valid_dataset, valid_labels = problem_4_randomize.randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(data_root,'notMNIST.pickle')

"""Compress and combine all datasets into one pickle file"""
try:
    f = open(pickle_file, 'wb')
    save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
            }
    
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

dup_test, santData_test, santLabel_test, dup_valid, santData_valid, santLabel_valid = problem_5_DuplicatedData.sant_Data(pickle_file)

print("dup_test", dup_test);
print("dup_valid", dup_valid);

matrix, acc = problem_6_logisticReg.logistic(pickle_file, 10000)

print("matrix", matrix);
print("matrix dimension", np.shape(matrix));
print("acc", acc);

print("This time test data is santized")
matrix, acc = problem_6_logisticReg.logistic(pickle_file, 10000, santData_test, santLabel_test)
print("matrix", matrix)
print("matrix dimension", np.shape(matrix));
print("acc", acc);