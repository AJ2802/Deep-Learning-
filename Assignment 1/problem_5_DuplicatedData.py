# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:38:57 2017

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
import hashlib

data_root='.' #Change me to store data elsewhere

image_size = 28

#sourceData is data you wanna remove duplicate
#refData is data you wanna check if data in sourceData overlap with refData
def extract_overlap_hash(sourceData, refData):
    hashData_source = np.array([hashlib.sha256(image).hexdigest() for image in sourceData]);
    hashData_ref    = np.array([hashlib.sha256(image).hexdigest() for image in refData]);
    overlap={}
    
    for source_index, source_hashImg in enumerate(hashData_source):
        hash_index = np.where(hashData_ref==source_hashImg);                    
        if len(hash_index[0])>0:
            overlap[source_index] = hash_index;
                   
    return overlap;
                
def remove__duplicated_data(overlap, sourceData, sourceLabel):
    
    index = list(overlap.keys());
    santData_source =np.delete(sourceData,index, 0);
    santLabel_source = np.delete(sourceLabel, index, None);
                              
    return santData_source, santLabel_source

def duplicateCount(overlap):
    return len(overlap);

            
def sant_Data(pickle_file):
    with open (pickle_file ,'rb') as fig:
        wholeData = pickle.load(fig);
        
        sourceData = wholeData['test_dataset']; 
        sourceLabel = wholeData['test_labels'];
        refData = wholeData['train_dataset'];
        refLabel = wholeData['train_labels'];
        
        overlap = extract_overlap_hash(sourceData, refData);
        santData_test, santLabel_test = remove__duplicated_data(overlap, sourceData, sourceLabel);
        dup_test = duplicateCount(overlap);
                            
        sourceData = wholeData['valid_dataset']; 
        sourceLabel = wholeData['valid_labels'];
        refData = wholeData['train_dataset'];
        refLabel = wholeData['train_labels'];
                            
        overlap = extract_overlap_hash(sourceData, refData);
        santData_valid, santLabel_valid = remove__duplicated_data(overlap, sourceData, sourceLabel);
        dup_valid = duplicateCount(overlap)
        
    return dup_test, santData_test, santLabel_test, dup_valid, santData_valid, santLabel_valid