# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 18:09:34 2017

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

num_classes = 10
np.random.seed(133)


image_size = 28 #Pixel width and height
pixel_depth = 255.0 #Number of levels per pixel.

""" show image from pickle files"""

def pickle_image(pickle_files, image_index):
     
    for pickle_file in pickle_files:
        with open(pickle_file,'rb') as fid:
            images = pickle.load(fid)
            for index in image_index:
                image = images[index,:,:]
                plt.imshow(image)
                plt.show()
        fid.close()