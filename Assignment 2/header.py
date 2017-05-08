# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 23:53:07 2017

@author: AJ
"""

#These are all modules we 'll using later. Make sure you can import them 
#before proceeding further

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import Reload
import Reformat
import ComputationGraph
import OpCompGraph
import GraphForSGD
import OpCompGraphForSGD
import GraphFor1LayerDL
import OpGraphFor1LayerDL


image_size = 28
num_labels = 10

pickle_file = 'notMNIST.pickle'
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = Reload.reload(pickle_file);
#print('Training set', train_dataset.shape, train_labels.shape);
#print('Validation set', valid_dataset.shape, valid_labels.shape);
#print('Test set', test_dataset.shape, test_labels.shape);

#data as a flat matrix (1 d array)
#labels as float 1-hot encodings
train_dataset, train_labels = Reformat.reformat(train_dataset, train_labels)
valid_dataset, valid_labels = Reformat.reformat(valid_dataset, valid_labels)
test_dataset, test_labels = Reformat.reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


print()
print("Prediction by Gradient Descent with no hidden layer")
graph, optimizer, loss, train_prediction, valid_prediction, test_prediction = ComputationGraph.buildingCompGraph(train_dataset, train_labels, valid_dataset, test_dataset);

OpCompGraph.graphRun(graph, optimizer, loss, train_labels, valid_labels, test_labels, train_prediction, valid_prediction,test_prediction);


#Now we switch to stochastic gradient descent training instead, which is much faster.
#The graph will be similar, except that instead of holding all the training data into a constant node,
#we create a Placeholder node which will be fed actual data at every call of session.run()
#Review what is a stochastic gradient decent then you will know why we need a Placeholder node 
#instead of constant node.
print()
print("Prediction by Stochastic Gradient Descent with no hidden layer")
graph, optimizer, loss, tf_train_dataset, tf_train_labels,train_prediction, valid_prediction, test_prediction = GraphForSGD.buildingCompGraph(valid_dataset, test_dataset)

OpCompGraphForSGD.graphRun(graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels)



print()
print("Prediction by Stochastic Gradient Descent with 1 hidden layer")
graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = GraphFor1LayerDL.buildingCompGraph(valid_dataset, test_dataset)
OpCompGraphForSGD.graphRun(graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels);
#OpGraphFor1LayerDL.graphRun(graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_dataset,train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels);

