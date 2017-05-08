# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:25:27 2017

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

def download_progress_hook(count, blockSize, totalSize):
    """ A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    
    global last_percent_reported
    percent = int (count*blockSize*100/totalSize)
    
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
            
        last_percent_reported = percent
        
def maybe_download(filename, expected_bytes,force=False):
    """ Donwload a file if not present, and make sure it's the right size"""

    dest_filename = os.path.join(data_root, filename)
    print("dest_filename",dest_filename);
    if force or not os.path.exists(dest_filename):
        print('Attempting to download', filename)
        filename, _=urlretrieve(url+filename, dest_filename, reporthook = download_progress_hook)
        print('\n Download Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception('Failed to verify ' + dest_filename +'. Can you get to it with a browser?')
    return dest_filename

