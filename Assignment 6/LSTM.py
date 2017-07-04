from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import BatchGenerator
import OutputLabels

num_nodes = 64


def graphForLSTM(num_nodes, vocabulary_size, batch_size, num_unrollings):
	graph = tf.Graph();
	with graph.as_default():
	
		# Parameters:
		
		# Input gate: input, previous output, and bias.
		ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1)) #Note vocabulary_size is the total number of english alphabet + a space character.
		im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		ib = tf.Variable(tf.zeros([1, num_nodes]))
		
		# Forget gate: input, previous output, and bias.
		fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		fb = tf.Variable(tf.zeros([1, num_nodes]))
		
		# Memory cell: input, state, and bias. #What are states?
		cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		cb = tf.Variable(tf.zeros([1, num_nodes]))
		
		# Output gate: input, previous output, and bias.
		ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		ob = tf.Variable(tf.zeros([1, num_nodes]))
		
		#Variables saving state across unrollings. What is the use of saved_output and saved_state
		saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable = False)
		saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable = False)
		
		#Classifier weights and biases
		w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1)) #logistic regression to classify output/state to words 
		b = tf.Variable(tf.zeros([vocabulary_size]))
		
		#Definition of the cell computation.
		def lstm_cell(i ,o , state): #i is a batch which is batch_size*vocabulary_size dimension o is a matrix of batch_size*num_nodes, state is a matrix of batch_size*num_nodes
			"""Create a LSTM cell. See e.g. : http://arxiv.org/pdf/1402.1128v1.pdf
			Note that in this formulation, we omit the various connections between the
			previous state and the gates."""
			input_gate = tf.sigmoid(tf.matmul(i,ix) + tf.matmul(o,im) +ib) #Eq. (1) or Eq.(7) of the paper except  no c_{t-1}.
			forget_gate = tf.sigmoid(tf.matmul(i,fx) + tf.matmul(o,fm) + fb) #Eq. (2) or Eq.(8) of the paper except no c_{t-1}.
			update = tf.matmul(i,cx) + tf.matmul(o,cm) + cb #see the second term in Eq.(3) or (9) in the RHS of the paper
			state = forget_gate*state + input_gate*tf.tanh(update) #Eq. (3) or Eq.(9) of the paper. state is the c_t which is the state in the memory cell.
			output_gate = tf.sigmoid(tf.matmul(i,ox) + tf.matmul(o,om)+ob) #Eq. 4 or 10 of the paper
			return output_gate*tf.tanh(state), state #Eq. 5 or 11 labelled by m_t in the paper and the state c_t
		
		# Input data.
		train_data = list()
		for _ in range(num_unrollings + 1):
			train_data.append(tf.placeholder(tf.float32, shape = [batch_size, vocabulary_size])) # train_data(a batches) contains a list of batch. len(train_data) = num_unrolling + 1
		train_inputs = train_data[:num_unrollings] #number of characters in each chunk of words
		train_labels = train_data[1:] #labels are inputs shifted by one time step.
		#Suppose train_data=[0,1,2,3,4,5] and num_unrollings = 5
		#then, train_inputs = [0,1,2,3,4] and train_labels = [1,2,3,4,5]. See the whole sequence shifted by 1.
		
		# Unrolled LSTM loop.
		outputs = list() 
		output = saved_output
		state = saved_state
		for i in train_inputs: #train_input is a list of batch.
			output, state = lstm_cell(i, output, state)
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
		#First find all derivatives FORUMULA (tensorflow is a symobolic computation) after a clip. The output is a symbolic operation/formula but not a numerical value.
		gradients, _ = tf.clip_by_global_norm(gradients, 1.25) #prevent from derivatives explosion
		optimizer = optimizer.apply_gradients(zip(gradients, v), global_step = global_step)
		
		#Predictions.
		train_prediction = tf.nn.softmax(logits) #id for characters
		
		#Sampling and validation eval: batch 1, no unrolling.
		sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
		
		saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
		saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
		
		reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])), saved_sample_state.assign(tf.zeros([1, num_nodes]))) #For validation
		
		sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state) 
		#sample_output->m_t, sample_state->c_t, sample_input->x_t, saved_sample_output->m_{t-1}, saved_sample_state -> c_{t-1}
		
		with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]): #finish calculation of previous sample_output and sample_state
			sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
			
			
		return graph, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input
		
def graphRun(graph, vocabulary_size, num_unrollings, first_letter, train_batches, valid_batches, valid_size, train_data, optimizer, loss, train_prediction, learning_rate, sample_prediction, reset_sample_state, sample_input):
	num_steps = 7001
	summary_frequency = 100
	
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized');
		mean_loss = 0
		for step in range(num_steps):
			batches = train_batches.next()
			feed_dict = dict()
			for i in range (num_unrollings + 1):
				feed_dict[train_data[i]] = batches[i] #note feed_dict is a dictionary. Both train_data and batches are lists of batch. #len(train_data)=len(batches) = unrollings + 1
			_, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict = feed_dict) #Information of the last batch in this LSTM reccurent model is stored to compute loss, optimizer and predictions in the current batch. 
			#Finish the whole iteration of train_data (batches) to compute loss , optimizer, loss in one iteration of the step loop. You can think of the recurrent neural network uses LSTM by around unrollings number of times. And this recurrent neural network predicts unrolling many characters in each iteration. [Note the number of LSTM is changing since it is based on the unrolling. Like when the validation set is sampled, this recurrent network uses LSTM by 79 times]
			mean_loss += l # total loss of the recurrent nueral network.
			if step % summary_frequency == 0 :
				if step > 0:
					mean_loss = mean_loss / summary_frequency
				# The mean loss is an estimate of the loss over the last few batches.
				print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
				mean_loss = 0
				labels = np.concatenate(list(batches)[1:])
				print('Minibatch perplexity: %.2f' %float(np.exp(OutputLabels.logprob(predictions, labels)))) #a kind of cross entropy function
				if (step % (summary_frequency * 10)) == 0:
					# Generate some samples/ generate a validation set based on our current training.
					print('='*80)
					for _ in range(5):
						feed =  OutputLabels.sample(OutputLabels.random_distribution(vocabulary_size), vocabulary_size)
						sentence =  BatchGenerator.characters(feed, first_letter)[0] #Randomly generate 1st letter (may not be meaningful and may not be even a legit words)
						reset_sample_state.run() #reset the sample state each time after we generate a validation set of sample.
						for _ in range(79):
							prediction = sample_prediction.eval({sample_input: feed})
							feed = OutputLabels.sample(prediction, vocabulary_size) #Trasnform a prob distribution to one hot encoding and update the feed (input).
							sentence += BatchGenerator.characters(feed, first_letter)[0]#Map from id to character
						print(sentence) #print the whole sentence starting from a random choice of the first letter. The sentence should be more and more make sense or at least contains more and more legit words along iterations of step.
					print('='*80);
				# Measure validation set perplexity.
				reset_sample_state.run()
				valid_logprob = 0
				for _ in range(valid_size):
					b = valid_batches.next()
					predictions = sample_prediction.eval({sample_input: b[0]})
					valid_logprob = valid_logprob + OutputLabels.logprob(predictions, b[1])
				print('Validation set perplexity: %2f' %float(np.exp(valid_logprob/valid_size)))
		
		
		
		 
