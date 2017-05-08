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

import Reload
import Reformat
import Accuracy
import Problem_1_Reg_Logistic;
import Problem_1_Reg_DL1Layer;
import Problem_2_Overfitting;
import Problem_3_Dropout
import Problem_4_SlowDecay_Dropout_DL2Layer
pickle_file = 'notMNIST.pickle'

train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = Reload.reload(pickle_file);
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = Reformat.reformat(train_dataset, train_labels)
valid_dataset, valid_labels = Reformat.reformat(valid_dataset, valid_labels)
test_dataset, test_labels = Reformat.reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Problem_1_Reg_Logistic.GraphForRegLog(train_dataset, train_labels, valid_dataset, test_dataset);

Problem_1_Reg_Logistic.graphRun(graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels)

graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Problem_1_Reg_DL1Layer.GraphForRegDL1Layer(train_dataset, train_labels, valid_dataset, test_dataset);

Problem_1_Reg_DL1Layer.graphRun(graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels)

Problem_2_Overfitting.graphRun(graph, optimizer, lossWithRegularization, multiplier, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels)

graph, optimizer, lossWithDropout, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Problem_3_Dropout.GraphForDropoutDL1Layer(train_dataset, train_labels, valid_dataset, test_dataset)

Problem_3_Dropout.graphRun(graph, optimizer, lossWithDropout,  tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels)

graph, optimizer, lossWithReg, multiplier, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction = Problem_4_SlowDecay_Dropout_DL2Layer.GraphForSlowDecayLR2LayerDropout(train_dataset, train_labels, valid_dataset, test_dataset)

Problem_4_SlowDecay_Dropout_DL2Layer.graphRun(graph, optimizer, lossWithReg, multiplier, tf_train_dataset, tf_train_labels, train_dataset, train_labels, train_prediction, valid_prediction, valid_labels, test_prediction, test_labels)
