# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 23:45:24 2017

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

def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder,image)
        try:
            image_data = (ndimage.imread(image_file).astype(float)-
                          pixel_depth/2)/pixel_depth
            if image_data.shape !=(image_size, image_size):
                raise Exception('Inexpected image shape: %s' %str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file,':',e,'- it\'s ok, skipping.')
        
    dataset = dataset[0:num_images,:,:]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d<%d' % 
                        (num_images, min_num_images))
        
    print('Full dataset tensor:', dataset.shape)
    print('Mena:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset
    
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            #You may override by setting force=True
            print('%s already present -Skipping pickling.' % set_filename)
        else:
            print('Pickling %s.' %set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename,'wb') as f:
                    pickle.dump(dataset,f, pickle.HIGHEST_PROTOCOL)
            except Exception  as e:
                print('Unable to save data to', set_filename, ':', e)
                
    return dataset_names
    
            
    
    