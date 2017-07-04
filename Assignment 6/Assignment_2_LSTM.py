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

import Assignment_2_BatchGenerator
import Assignment_2_OutputLabels

num_nodes = 512
embedding_size = 128 

def graphForLSTM(num_nodes, bigram_vocabulary_size, batch_size, num_unrollings):
	graph = tf.Graph();
	with graph.as_default():
	
		# Parameters:
		x = tf.Variable(tf.truncated_normal([embedding_size, num_nodes*4], -0.1, 0.1)) #Be careful the dimension between x and m
		m = tf.Variable(tf.truncated_normal([num_nodes, num_nodes*4], -0.1, 0.1))
		biases = tf.Variable(tf.zeros([1, num_nodes * 4]))

		#Variables saving state across unrollings. What is the use of saved_output and saved_state
		saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable = False)
		saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable = False)
		
		#Classifier weights and biases
		w = tf.Variable(tf.truncated_normal([num_nodes, bigram_vocabulary_size], -0.1, 0.1)) #logistic regression to classify output/state to words 
		b = tf.Variable(tf.zeros([bigram_vocabulary_size]))
		
		# embedding for all possible bigrams
		
		embeddings = tf.Variable(tf.random_uniform([bigram_vocabulary_size, embedding_size], -1.0 ,1.0))
		
		#one hot encoding for labels in
		np_one_hot = np.zeros((bigram_vocabulary_size, bigram_vocabulary_size))
		np.fill_diagonal(np_one_hot,1)
		bigram_one_hot = tf.constant(np.reshape(np_one_hot, -1), dtype = tf.float32, shape=[bigram_vocabulary_size, bigram_vocabulary_size]) # bigram_one_hot is a bigram_vocabulary_size by bigram_vocabulary_size matrix with diagonal 1 and all other entries zero.		
		keep_prob = tf.placeholder(tf.float32)
		
		
		#Definition of the cell computation.
		def lstm_cell(i ,o , state): #i is a batch which is batch_size*vocabulary_size dimension o is a matrix of batch_size*num_nodes, state is a matrix of batch_size*num_nodes
			"""Create a LSTM cell. See e.g. : http://arxiv.org/pdf/1402.1128v1.pdf
			Note that in this formulation, we omit the various connections between the
			previous state and the gates."""
			
			i = tf.nn.dropout(i, keep_prob)
			overall = tf.matmul(i,x) + tf.matmul(o,m) + biases
			overall_input, overall_output, overall_update, overall_output = tf.split(overall, 4, 1) 
			input_gate = tf.sigmoid(overall_input)
			forget_gate = tf.sigmoid(overall_output)
			state = forget_gate * state + input_gate*tf.tanh(overall_update)
			output_gate = tf.sigmoid(overall_output)
			
			output = output_gate*tf.tanh(state)
			
			output = tf.nn.dropout(output, keep_prob)
			
			return output, state #Eq. 5 or 11 labelled by m_t in the paper and the state c_t
		
		# Input data. [num_unrollings, batch_size] -> one hot encoding removed, we send just bigram ids
		tf_train_data = tf.placeholder(tf.int32, shape=[num_unrollings + 1, batch_size]) #batches is a list with length unrollings + 1 and batches contain a batch which a vector of size batch_size.
		train_data = list()
		#tf_train_data is a list of batch which can be thought of a matrix  row is unrollings + 1 and col is batch_size
		#tf.split(tf_train_data, num_rollings + 1, 0) is to split into [0th row, 1st row,....,num_rollings^th row] (row of tf_train_data)
		#i first is 0th row, and then 1st row, ...
		
		for i in tf.split(axis = 0, num_or_size_splits = num_unrollings + 1, value = tf_train_data):
			train_data.append(tf.squeeze(i)) #train_data is now a list of an integer with range from 0 to bigram_vocabulary_size and the length of the list is num_rollings + 1
		train_inputs = train_data[: num_unrollings]
		train_labels = list()
		for l in train_data[1:]:
			train_labels.append(tf.gather(bigram_one_hot,l)) #it returns a one hot encoding, like if l[0]=10, it return [0,...,1,...,0] where 1 is at 10^th position
		
		# Unrolled LSTM loop.
		outputs = list() 
		output = saved_output
		state = saved_state
		for i in train_inputs: #train_input is a list of batch.
			output, state = lstm_cell(tf.nn.embedding_lookup(embeddings, i), output, state) #Note that the second argumnet of embedding_lookup is a list of integer (indices of 1st argument or I think it is rows of 1st argument.)
			outputs.append(output) #outputs stores num_unrollings number of output m_t.
			
		# State saving across unrollings.
		with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]): #get last output m_{t-1} and last state c_{t-1}
			# Classifier: #predict the next character ('a-z'+' ') from the current input (current character) and state (signal from last hidden layer) 
			              #convert each row in an output(batch_size*num_nodes) to a character (a-z + ' ').
			logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b) #xw_plus_b is to compute matmul(x, weights)+ biases # Note concat(outputs, 0) is a list of matrix. #see Eq.(6) or (15) in the paper
			                                                      #logits is y_t in the paper
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels = tf.concat(train_labels, 0), logits = logits))
			
		# Optimizer.
		global_step = tf.Variable(0) #number of iterations
		learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase = True)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(loss)) #gradient = \partial loss \partial v and v is variable
		# Note how to prevent derivatives explosion in their approach.
		#First find all derivatives FORUMULA (tensorflow is a symbolic computation) after a clip. The output is a symbolic operation/formula but not a numerical value.
		gradients, _ = tf.clip_by_global_norm(gradients, 1.25) #prevent from derivatives explosion
		optimizer = optimizer.apply_gradients(zip(gradients, v), global_step = global_step)
		
		#Predictions.
		train_prediction = tf.nn.softmax(logits) #id for characters
		
		#Sampling and validation eval: batch 1, no unrolling.
		sample_input = tf.placeholder(tf.int32, shape=[1])
		
		saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
		saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
		
		reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])), saved_sample_state.assign(tf.zeros([1, num_nodes]))) #For validation
		
		embedd_sample_input = tf.nn.embedding_lookup(embeddings, sample_input);
		sample_output, sample_state = lstm_cell(embedd_sample_input, saved_sample_output, saved_sample_state) 
		#sample_output->m_t, sample_state->c_t, sample_input->x_t, saved_sample_output->m_{t-1}, saved_sample_state -> c_{t-1}
		
		with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]): #finish calculation of previous sample_output and sample_state
			sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
			
			
		return graph, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input, tf_train_data, keep_prob
		
def graphRun(graph, vocabulary_size, num_unrollings, first_letter, train_batches, valid_batches, valid_size, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input, tf_train_data, keep_prob, bigram_vocabulary_size):
	num_steps = 7001
	summary_frequency = 100
		
	bi_onehot = np.zeros((bigram_vocabulary_size, bigram_vocabulary_size))
	np.fill_diagonal(bi_onehot, 1);

	def bi_one_hot (encodings): #It is give a one-hot encoding representation of bigram integer. For a0, it will be [0,...,1,...0] where 1 is at vocabulary_size^th location.
		return [bi_onehot[e] for e in encodings]
	
	

	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized');
		mean_loss = 0
		for step in range(num_steps):
			batches = train_batches.next()
			feed_dict = dict()
			feed_dict={tf_train_data: batches, keep_prob: 0.6}
			
			_, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict = feed_dict) #Information of the last batch in this LSTM reccurent model is stored to compute loss, optimizer and predictions in the current batch. 
			
			
			#Finish the whole iteration of train_data (batches) to compute loss , optimizer, loss in one iteration of the step loop. You can think of the recurrent neural network uses LSTM by around unrollings number of times. And this recurrent neural network predicts unrolling many characters in each iteration. [Note the number of LSTM is changing since it is based on the unrolling. Like when the validation set is sampled, this recurrent network uses LSTM by 79 times]
			mean_loss += l # total loss of the recurrent nueral network.
			if step % summary_frequency == 0 :
				if step > 0:
					mean_loss = mean_loss / summary_frequency
				# The mean loss is an estimate of the loss over the last few batches.
				print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
				mean_loss = 0
				
				labels = list(batches)[1:]
				labels = np.concatenate([bi_one_hot(l) for l in labels])
				
				#labels = np.concatenate(list(batches)[1:])
				
				print('Minibatch perplexity: %.2f' %float(np.exp(Assignment_2_OutputLabels.logprob(predictions, labels)))) #a kind of cross entropy function
				if (step % (summary_frequency * 10)) == 0:
					# Generate some samples/ generate a validation set based on our current training.
					print('='*80)
					for _ in range(5):
						
						feed = np.argmax(Assignment_2_OutputLabels.sample(Assignment_2_OutputLabels.random_distribution(bigram_vocabulary_size), bigram_vocabulary_size))
						sentence = Assignment_2_BatchGenerator.bi2str(feed, vocabulary_size, first_letter);
						
						reset_sample_state.run() #reset the sample state each time after we generate a validation set of sample.
						for _ in range(79):
							prediction = sample_prediction.eval({sample_input: [feed], keep_prob: 1.0})
							
							feed = np.argmax( Assignment_2_OutputLabels.sample(prediction, bigram_vocabulary_size))
							sentence += Assignment_2_BatchGenerator.bi2str(feed, vocabulary_size, first_letter);
						
						print(sentence) #print the whole sentence starting from a random choice of the first letter. The sentence should be more and more make sense or at least contains more and more legit words along iterations of step.
					print('='*80);
				# Measure validation set perplexity.
				reset_sample_state.run()
				valid_logprob = 0
				for _ in range(valid_size):
					b = valid_batches.next()
					predictions = sample_prediction.eval({sample_input: [feed], keep_prob: 1.0})
					valid_logprob = valid_logprob + Assignment_2_OutputLabels.logprob(predictions, Assignment_2_OutputLabels.one_hot_voc(b[1],bigram_vocabulary_size))
				print('Validation set perplexity: %2f' %float(np.exp(valid_logprob/valid_size)))
		
		
		
		 
