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


def build_dataset(words, vocabulary_size):
	vocabulary_size = vocabulary_size #first 50000 most frequent words in the "words" dataset.
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size-1)) #count store (common word, frequency), note the len(count) = vocabulary_size
	dictionary = dict()
	i = 0;
	for word, _ in count:
		dictionary[word] = len(dictionary); #you can think of dictionary map word to the page on the count that shows up the word.
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0 #dictionary['UNK']
			unk_count = unk_count + 1
		data.append(index) # data store the page on the count that shows up the words.
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) #zip try to combine (dictionary value and dictionary keys) as 2-tuple and reverse dictionary is map from the page on the count that shows up the word to the word.
	
	return data, count, dictionary, reverse_dictionary
	
