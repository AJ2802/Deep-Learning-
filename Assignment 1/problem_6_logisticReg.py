# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:57:50 2017

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

image_size = 28



def score (actual,pred):
    length = len(actual);
    accuracy = 0;
    for i in range(length):
        if actual[i]==pred[i]:
            accuracy = accuracy + 1;
    
    return accuracy/length*100
       
    
def logistic(pickle_file, num_train, santData_test=None, santLabel_test=None):
    with open(pickle_file, 'rb') as fid:
        images = pickle.load(fid);
        trainX = images['train_dataset'][0:num_train].reshape(num_train,image_size*image_size)
        trainY = images['train_labels'][0:num_train]
        reg = LogisticRegression()
        reg.fit(trainX,trainY);
        
        if santData_test==None:
            lengthTest = len(images['test_dataset']);
            testX = images['test_dataset'].reshape(lengthTest,image_size*image_size)
            testY = images['test_labels']
        else:
            lengthTest = len(santData_test);
            testX = santData_test.reshape(lengthTest,image_size*image_size)
            testY = santLabel_test
        
        predY = reg.predict(testX);
        accuracy = score(predY, testY);
        # OR accuracy = reg.score(testX,testY)*100;
        #print("score", reg.score(testX, testY))
        
    return reg.coef_, accuracy;
            