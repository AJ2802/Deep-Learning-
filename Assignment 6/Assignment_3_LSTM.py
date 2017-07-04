##Reference : https://github.com/hankcs/udacity-deep-learning/blob/master/6_lstm.py

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import math

#from tensorflow.models.rnn.translate import seq2seq_model
import seq2seq_model
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

import Assignment_3_BatchGenerator
import OutputLabels
import Transformation

num_nodes = 64


def create_model(forward_only, vocabulary_size, batch_size):
	model = seq2seq_model.Seq2SeqModel(source_vocab_size = vocabulary_size, target_vocab_size = vocabulary_size, buckets=[(20, 20)], size = 256, num_layers = 4, max_gradient_norm = 5.0, batch_size = batch_size, learning_rate = 1.0, learning_rate_decay_factor = 0.9, use_lstm = True, forward_only = forward_only);
	return model
	
	
	

def graphRun(vocabulary_size, batch_size, valid_size, train_batches, valid_batches, first_letter):
	
	with tf.Session() as sess:
		model = create_model(False, vocabulary_size, batch_size);
		sess.run(tf.initialize_all_variables())
		num_steps = 30001
		
		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		step_ckpt = 100
		valid_ckpt = 500
		
		for step in range(1, num_steps):
			model.batch_size = batch_size
			batches = train_batches.next()
			train_sets = []
			batch_encs = list(map(lambda x: list(map(lambda y: Transformation.char2id(y, first_letter), list(x))), batches)) #batches is a list of batch_size many batch. i^th batch is a list of num_unrollings many characters in i^th segment
			batch_decs = list(map(lambda x: Assignment_3_BatchGenerator.rev_id(x, first_letter), batches))
			for i in range(len(batch_encs)):
				train_sets.append((batch_encs[i], batch_decs[i])) #batch_encs[i] and batch_decs[i] comes i^th batch #Training Data: batch_encs[i] and Label: batch_decs[i] 
			#Get a batch and make a step
			encoder_inputs, decoder_inputs, target_weights = model.get_batch([train_sets],0)
			_, step_loss,_ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, forward_only = False)
			
			loss += step_loss / step_ckpt
			
			#Once in a while, we save checkpoint, print statistics, and run evals.
			if step % step_ckpt == 0:
				# Print statistics for the previous epoch.
				perplexity = math.exp(loss) if loss < 300 else float('inf')
				
				#Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and loss > max(previous_losses[-3:]): #(max of last three losses)
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)
				
				loss = 0.0
				
				if step % valid_ckpt == 0:
					v_loss = 0.0
					model.batch_size = 1
					batches = ['the quick brown fox']
					test_sets = []
					batch_encs = list(map(lambda x: list(map(lambda y: Transformation.char2id(y, first_letter), list(x))), batches))
					test_sets.append((batch_encs[0],[]))
					#Get a 1-element batch to feed the sentence to the model.
					encoder_inputs, decoder_inputs, target_weights = model.get_batch([test_sets],0)
					#get output logits for the sentence
					_, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, forward_only = True)
					#This is a greedy decoder - outputs are just argmaxes of output_logits.
					outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
					print('>>>>>>>', batches[0],'->', ''.join(map(lambda x:Transformation.id2char(x, first_letter), outputs))) #Note batches[0]='the quick brown fox'
					#uses to calcualte perplexity based on valid data set"
					for _ in range(valid_size):
						model.batch_size = 1
						v_batches = valid_batches.next()
						valid_sets = []
						v_batch_encs = list(map(lambda x: list(map(lambda y: Transformation.char2id(y, first_letter), list(x))), v_batches))
						v_batch_decs = list(map(lambda x: Assignment_3_BatchGenerator.rev_id(x, first_letter), v_batches))
						for i in range(len(v_batch_encs)):
							valid_sets.append((v_batch_encs[i], v_batch_decs[i]))
						encoder_inputs, decoder_inputs, target_weights = model.get_batch([valid_sets],0)
						_, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)
						v_loss += eval_loss / valid_size
						
					eval_ppx = math.exp(v_loss) if v_loss<300 else float('inf')
					print(" valid eval: perplexity %.2f" %(eval_ppx))
					
		# reuse variable -> subdivide into two boxes
		model.batch_size = 1 # We decode one sentence at a time
		batches = ['the quick brown fox']
		test_sets = []
		batch_encs = list(map(lambda x: list(map(lambda y: Transformation.char2id(y, first_letter), list(x))), batches))
		test_sets.append((batch_encs[0],[]))
		#Get a 1-element batch to feed the sentence to the model.
		encoder_inputs, decoder_inputs, target_weights = model.get_batch([test_sets],0)
		#Get output logits for the sentence.
		_,_,output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)
		# This is a greedy decoder - outputs are just argmaxes of output_logits.
		outputs = [int(np.argmax(logit, axis = 1)) for logit in output_logits]
		print('##:', outputs)
		#If there is an EOS symbol in outputs, cut them at that point.
		if Transformation.char2id('!', first_letter) in outputs:
			outputs = outputs[:outputs.index(Transformation.char2id('!', first_letter))]
		print(batches[0], '->', '.'.join(map(lambda x: Transformation.id2char(x, first_letter), outputs)))
						
					
					
					
					
					
					
