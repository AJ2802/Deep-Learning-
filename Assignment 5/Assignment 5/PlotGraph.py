from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE



def plot(embeddings, labels):
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	pylab.figure(figsize = (15, 15)) # in inches
	for i, label in enumerate(labels):
		x, y = embeddings[i,:]
		pylab.scatter(x,y)
		pylab.annotate(label, xy=(x, y), xytext= (5, 2), textcoords = 'offset points', ha = 'right', va = 'bottom')
	pylab.show()
