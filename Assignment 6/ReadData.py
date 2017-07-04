from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		name = f.namelist()[0]
		#f.namelist() = ['text8']
		#f.namelist()[0] = text8;
		data = tf.compat.as_str(f.read(name))
		#tf.compat.as_str convert input to string and separate each words (default separator is space).
		#.split()-->default separator is space
		#e.g. if split(",")-->separator is ,
		#Note tf.compat.as_str(f.read(name)) return as a list [] form
	return data
	

