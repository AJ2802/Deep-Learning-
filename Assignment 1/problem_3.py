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

image_size = 28 #Pixel width and height
pixel_depth = 255.0 #Number of levels per pixel.


def make_array(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size),dtype = np.float32)
        labels = np.ndarray(nb_rows,dtype = np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_array(valid_size, image_size)
    train_dataset, train_labels = make_array(train_size, image_size)
    vsize_per_class = valid_size // num_classes #take quotient
    tsize_per_class = train_size // num_classes #take quotient
    
    start_v, start_t =0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                #let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class,:,:]
                    valid_dataset[start_v:end_v,:,:] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v +=vsize_per_class
                
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t: end_t, :, :] = train_letter
                train_labels[start_t: end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
                
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    
    return valid_dataset, valid_labels, train_dataset, train_labels


                
