#These are all the modules we' ll be using later. Make sure you can import them
#before proceeding further

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import Reload
import Reformat
import Accuracy
import Convolution
import Problem1_Convolution_max_pool
import Problem2_Convolution_best_perform
import time

pickle_file = 'notMNIST.pickle'

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = Reload.reload(pickle_file)
train_dataset, train_labels = Reformat.reformat(train_dataset, train_labels)
valid_dataset, valid_labels = Reformat.reformat(valid_dataset, valid_labels)
test_dataset, test_labels = Reformat.reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


start = time.time()
graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Convolution.convolution(valid_dataset, test_dataset)

Convolution.graphRun(graph, optimizer, loss, train_dataset, train_labels, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction, valid_labels, test_labels)
end = time.time()
print("Time for conv with stride 2 and without max pool is", end-start); 



start = time.time()
graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Problem1_Convolution_max_pool.convolution(valid_dataset, test_dataset)

Problem1_Convolution_max_pool.graphRun(graph, optimizer, loss, train_dataset, train_labels, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction, valid_labels, test_labels)
end = time.time()
print("Time for conv with a stride 1 and a 2-by-2 max pool with a stride 2 (problem 1) is", end-start); 


start = time.time()
graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Problem2_Convolution_best_perform.convolution(valid_dataset, test_dataset)

Problem2_Convolution_best_perform.graphRun(graph, optimizer, loss, train_dataset, train_labels, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction, valid_labels, test_labels)
end = time.time()
print("Time for conv with 3 different filters with max pool, Xavier initializationn, dropout and regularization (problem 2) is", end-start); 
