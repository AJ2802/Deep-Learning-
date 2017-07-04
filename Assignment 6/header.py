# There are all the modules we'll using later. Make sure you can import them
# before proceeding further
#Reference : https://github.com/hankcs/udacity-deep-learning/blob/master/6_lstm.py

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import Download
import ReadData
import Transformation

import BatchGenerator
import OutputLabels
import LSTM
import Assignment_1

import Assignment_2_BatchGenerator
import Assignment_2_OutputLabels
import Assignment_2_LSTM

import Assignment_3_BatchGenerator
import Assignment_3_LSTM

filename = Download.maybe_download('text8.zip', 31344016);

text = ReadData.read_data(filename);
print('Data size %d' % len(text))

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1 #[a-z]+' ' string.ascii_lowercase ->'abcde...z'
first_letter = ord(string.ascii_lowercase[0]) #Given a string of length one, return an integer representing the Unicode code point of the character e.g.ord('a')=97 and chr(97)='a'.

print(Transformation.char2id('a', first_letter), Transformation.char2id('z', first_letter), Transformation.char2id(' ', first_letter), Transformation.char2id('ï', first_letter))
#a -> 1
print(Transformation.id2char(1, first_letter), Transformation.id2char(26, first_letter), Transformation.id2char(0, first_letter));
# 1 -> a

#Function to generate a training batch for the LSTM model.
batch_size = 32
num_unrollings = 10; #number of characters in each chunk of words

train_batches = BatchGenerator.BatchGenerator(train_text, batch_size, num_unrollings, vocabulary_size, first_letter); #Note vocabulary_size is the total number of english alphabet + a space character.
valid_batches = BatchGenerator.BatchGenerator(valid_text, 1,1, vocabulary_size, first_letter);

print(BatchGenerator.batches2string(train_batches.next(), first_letter)) #return a list of chunks of words, the length of the list is batch_size, the length of each chunk of words is num_unrollings + 1
print(BatchGenerator.batches2string(train_batches.next(), first_letter))
print(BatchGenerator.batches2string(valid_batches.next(), first_letter))
print(BatchGenerator.batches2string(valid_batches.next(), first_letter))

#Simple LSTM Model
num_nodes = 64
graph, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input = LSTM.graphForLSTM(num_nodes, vocabulary_size, batch_size, num_unrollings)
LSTM.graphRun(graph, vocabulary_size, num_unrollings, first_letter, train_batches, valid_batches, valid_size, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input)


#Assignment 1
#Simple LSTM Model
graph, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input = Assignment_1.graphForLSTM(num_nodes, vocabulary_size, batch_size, num_unrollings)
Assignment_1.graphRun(graph, vocabulary_size, num_unrollings, first_letter, train_batches, valid_batches, valid_size, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input)

#Assignment 2

bigram_vocabulary_size = vocabulary_size * vocabulary_size

train_batches =Assignment_2_BatchGenerator.BatchGenerator(train_text, 8, 8, vocabulary_size, first_letter);
valid_batches = Assignment_2_BatchGenerator.BatchGenerator(valid_text, 1, 1, vocabulary_size, first_letter);

print(Assignment_2_BatchGenerator.bibatches2string(train_batches.next(), vocabulary_size, first_letter))
print(Assignment_2_BatchGenerator.bibatches2string(train_batches.next(), vocabulary_size, first_letter))
print(Assignment_2_BatchGenerator.bibatches2string(valid_batches.next(), vocabulary_size, first_letter))
print(Assignment_2_BatchGenerator.bibatches2string(valid_batches.next(), vocabulary_size, first_letter))
#Simple LSTM Model
train_batches =Assignment_2_BatchGenerator.BatchGenerator(train_text, batch_size, num_unrollings, vocabulary_size, first_letter);
valid_batches = Assignment_2_BatchGenerator.BatchGenerator(valid_text, 1, 1, vocabulary_size, first_letter);

graph, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input, tf_train_data, keep_prob = Assignment_2_LSTM.graphForLSTM(num_nodes, bigram_vocabulary_size, batch_size, num_unrollings)
Assignment_2_LSTM.graphRun(graph, vocabulary_size, num_unrollings, first_letter, train_batches, valid_batches, valid_size, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input, tf_train_data, keep_prob, bigram_vocabulary_size)


"""
#Assignment 3
train_batches = Assignment_3_BatchGenerator.BatchGenerator(train_text, batch_size, num_unrollings, vocabulary_size, first_letter)
valid_batches = Assignment_3_BatchGenerator.BatchGenerator(valid_text, 1, num_unrollings, vocabulary_size, first_letter)
batches = train_batches.next()
train_sets = []
batch_encs = list(map(lambda x: list(map(lambda y: Transformation.char2id(y, first_letter), list(x))), batches)) #batches is a list of batch_size many batch. i^th batch is a list of num_unrollings many characters in i^th segment
batch_decs = list(map(lambda x: Assignment_3_BatchGenerator.rev_id(x, first_letter), batches))

print('x=', ''.join([Transformation.id2char(x, first_letter) for x in batch_encs[0]]))
print('y=', ''.join([Transformation.id2char(x, first_letter) for x in batch_decs[0]]))


Assignment_3_LSTM.graphRun(vocabulary_size, batch_size, valid_size, train_batches, valid_batches, first_letter)
"""




