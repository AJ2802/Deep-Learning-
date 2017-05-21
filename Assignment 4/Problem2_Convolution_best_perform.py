from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import Accuracy
import tensorflow as tf

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 128

unit_patch_size = 1;

small_patch_size = 3


graph = tf.Graph()

def convolution(valid_dataset, test_dataset):
	with graph.as_default():

		#input data
		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
		tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		tf_valid_dataset = tf.constant(valid_dataset);
		tf_test_dataset = tf.constant(test_dataset);
	
		#Variables:
		model_1_layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev = np.sqrt(2.0 / (patch_size * patch_size * num_channels)))) 
		model_1_layer1_biases = tf.Variable(tf.zeros([depth])) # why constant is 0
		
		model_1_layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev = np.sqrt(2.0 / (patch_size * patch_size * depth // 4)))) 
		model_1_layer2_biases = tf.Variable(tf.constant(1.0, shape =[depth]))   #why constant is 1
		
		layer3_weights = tf.Variable(tf.truncated_normal([image_size//4 *image_size//4 *depth * 3, num_hidden], stddev = np.sqrt(2.0 / (image_size * image_size * depth // 4 // 4 * 3 ))))		
		layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
		
		layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev = np.sqrt(2.0/num_hidden)))
		layer4_biases = tf.Variable(tf.constant(1.0, shape = [num_labels]))
	
		model_2_layer1_weights = tf.Variable(tf.truncated_normal([unit_patch_size, unit_patch_size, num_channels, depth], stddev = np.sqrt(2.0 / (unit_patch_size * unit_patch_size * num_channels))))
		model_2_layer1_biases = tf.Variable(tf.zeros([depth]))
		model_2_layer2_weights = tf.Variable(tf.truncated_normal([unit_patch_size, unit_patch_size, depth, depth], stddev = np.sqrt(2.0 / (unit_patch_size * unit_patch_size  * depth // 4 )))) 
		model_2_layer2_biases = tf.Variable(tf.constant(1.0, shape =[depth]))   #why constant is 1
		
		model_3_layer1_weights = tf.Variable(tf.truncated_normal([small_patch_size, small_patch_size, num_channels, depth], stddev = np.sqrt(2.0 / (small_patch_size * small_patch_size * num_channels )))) 
		model_3_layer1_biases = tf.Variable(tf.zeros([depth]))   
		model_3_layer2_weights = tf.Variable(tf.truncated_normal([small_patch_size, small_patch_size, depth, depth], stddev = np.sqrt(2.0 / (small_patch_size * small_patch_size * depth // 4)))) 
		model_3_layer2_biases = tf.Variable(tf.constant(1.0, shape =[depth]))   #why constant is 1

		#Model
		def convModel(data, weights_layer1, biases_layer1, weights_layer2, biases_layer2, keep_prob_hidden):
			hidden = tf.nn.relu(tf.nn.conv2d(data, weights_layer1, [1,1,1,1], padding ='SAME')+ biases_layer1)			
			max_pool = tf.nn.max_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			dropout = tf.nn.dropout(max_pool, keep_prob = keep_prob_hidden);
			
			hidden = tf.nn.relu(tf.nn.conv2d(dropout, weights_layer2, [1,1,1,1], padding ='SAME')+ biases_layer2)			
			max_pool = tf.nn.max_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			dropout = tf.nn.dropout(max_pool, keep_prob = keep_prob_hidden);
			
			return dropout;
			
			
			
			
		def model(data, keep_prob_hidden):
			
			#hidden = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1,1,1,1], padding ='SAME')+ layer1_biases)			
			#max_pool = tf.nn.max_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			#dropout = tf.nn.dropout(max_pool, keep_prob = keep_prob_hidden);
			
			#hidden= tf.nn.conv2d(max_pool, layer2_weights, [1,1,1,1], padding = 'SAME')
			#max_pool = tf.nn.max_pool(hidden, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
			#dropout = tf.nn.dropout(max_pool, keep_prob = keep_prob_hidden);
			
			model1 = convModel(data, model_1_layer1_weights, model_1_layer1_biases,  model_1_layer2_weights, model_1_layer2_biases, keep_prob_hidden);
			model2 = convModel(data, model_2_layer1_weights, model_2_layer1_biases,  model_2_layer2_weights, model_2_layer2_biases, keep_prob_hidden);
			model3 = convModel(data, model_3_layer1_weights, model_3_layer1_biases,  model_3_layer2_weights, model_3_layer2_biases, keep_prob_hidden);
			
			

			shape1 = model1.get_shape().as_list()
			reshape1 = tf.reshape(model1, [shape1[0], shape1[1]*shape1[2]*shape1[3]]) #shape[0] ~~batch_size, shape[1]~~out_height, shape[2]~~out_width, shape3~~depth
					
		
			shape2 = model2.get_shape().as_list()
			reshape2 = tf.reshape(model2, [shape2[0], shape2[1]*shape2[2]*shape2[3]])
		
			shape3 = model3.get_shape().as_list()
			reshape3 = tf.reshape(model3, [shape3[0], shape3[1]*shape3[2]*shape3[3]])
		
			reshape = tf.concat((reshape1, reshape2, reshape3), axis=1)

			hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
			dropout = tf.nn.dropout(hidden, keep_prob = keep_prob_hidden);
			
			return tf.matmul(hidden, layer4_weights) + layer4_biases
	
		# Training computation
		logits = model(tf_train_dataset, 0.5)
		Reg =  0.002 *(tf.nn.l2_loss(layer3_biases)+ tf.nn.l2_loss(layer3_biases)+ tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases))
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf_train_labels, logits = logits)) + Reg
	
	# Optimizer
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(0.001, global_step, 10000, 0.5, staircase=True) #staircase stands for integer division of global step /decay_step (an parameter right after global_step)
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step);
		
		#optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	#Predictions for the training, validation, and test data
		train_prediction = tf.nn.softmax(logits)
		valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1))
		test_prediction = tf.nn.softmax(model(tf_test_dataset, 1))
	
	return graph, optimizer, loss, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction
	
	

	
	
def graphRun(graph, optimizer, loss, train_dataset, train_labels, tf_train_dataset, tf_train_labels, train_prediction, valid_prediction, test_prediction, valid_labels, test_labels):
	
	num_steps = 30001
	
	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized')
		for step in range(num_steps):
			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train_dataset[offset : (offset + batch_size), :, :, :]
			batch_labels = train_labels [offset : (offset + batch_size), :]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)
			if (step % 50 == 0):
				print('Minibatch loss at step %d: %f' %(step, l))
				print('Minibatch accuracy: %.1f%%' % Accuracy.accuracy(predictions, batch_labels))
				print('Validation accuracy: %.1f%%' % Accuracy.accuracy(valid_prediction.eval(), valid_labels))
		print('Test accuracy: %.1f%%' % Accuracy.accuracy(test_prediction.eval(), test_labels));
			
