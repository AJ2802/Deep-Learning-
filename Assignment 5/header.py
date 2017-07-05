#These are all the modules we'll be using later. Make sure you cna import them
#before proceeding further

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

import Download
import ReadData
import Vocabulary
import Batch_Generation
import Train
import PlotGraph
import CBOW

filename = Download.maybe_download('text8.zip', 31344016);
words = ReadData.read_data(filename);
print('Data size %d' % len(words));

vocabulary_size = 50000 #first 50000 most frequent words in the "words" dataset.
data, count, dictionary, reverse_dictionary = Vocabulary.build_dataset(words, vocabulary_size);
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words #Hint to reduce memory

#Skip gram
print('data:', [reverse_dictionary[di] for di in data[:8]]) #data show first 8 words in the words.

for num_skips, skip_window in [(2,1), (4,2)]:
	data_index = 0
	batch, labels = Batch_Generation.generate_batch(data, batch_size = 8, num_skips = num_skips, skip_window = skip_window)
	print('\n with num_skips = %d and skip_window =%d:' % (num_skips, skip_window))
	print(' batch:', [reverse_dictionary[bi] for bi in batch])
	print(' labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
	

batch_size = 128
embedding_size = 128 #Dimension of the embedding vector
skip_window = 1 # How many words to consider left and right
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size)) # is a array of randomly pick valid_size number of elt in [0,...,valid_window -1]
num_sampled = 64 # Number of negative examples to sample.

graph, train_dataset, train_labels, normalized_embeddings, valid_examples, optimizer, loss, similarity = Train.graphForText(batch_size, embedding_size, skip_window, num_skips, valid_size, valid_window, valid_examples, num_sampled, vocabulary_size)

final_embeddings = Train.graphRun(graph, data, train_dataset, train_labels, batch_size, num_skips, skip_window, optimizer, loss, normalized_embeddings, reverse_dictionary, valid_size, valid_examples , similarity)

num_points = 400

tsne = TSNE(perplexity=30, n_components = 2, init = 'pca', n_iter = 5000)
two_d_embeddings = tsne.fit_transform (final_embeddings[1: num_points+1,:]);

words = [reverse_dictionary[i] for i in range(1, num_points + 1)] #first 400 common words in words which is deleted before
PlotGraph.plot(embeddings = two_d_embeddings, labels = words)


#CBOW

print('data:', [reverse_dictionary[di] for di in data[:8]])
for skip_window in [2,4]:
    data_index = 0
    batch, labels = CBOW.generate_batch(data, batch_size=8, skip_window=skip_window)
    print('\nwith skip_window = %d:' % (skip_window))
    print('    batch:', [[reverse_dictionary[words] for words in bi ] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
    
batch_size = 128
embedding_size = 128 #Dimension of the embedding vector
skip_window = 1 # How many words to consider left and right
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size)) # is a array of randomly pick valid_size number of elt in [0,...,valid_window -1]
num_sampled = 64 # Number of negative examples to sample.

graph, train_dataset, train_labels, normalized_embeddings, valid_examples, optimizer, loss, similarity = CBOW.graphForText(batch_size, embedding_size, skip_window, valid_size, valid_window, valid_examples, num_sampled, vocabulary_size)

final_embeddings = CBOW.graphRun(graph, data, train_dataset, train_labels, batch_size, skip_window, optimizer, loss, normalized_embeddings, reverse_dictionary, valid_size, valid_examples , similarity)

num_points = 400

tsne = TSNE(perplexity=30, n_components = 2, init = 'pca', n_iter = 5000)
two_d_embeddings = tsne.fit_transform (final_embeddings[1: num_points+1,:]);

words = [reverse_dictionary[i] for i in range(1, num_points + 1)] #first 400 common words in words which is deleted before
PlotGraph.plot(embeddings = two_d_embeddings, labels = words)
