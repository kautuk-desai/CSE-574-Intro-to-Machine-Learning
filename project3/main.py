import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from utilities import Utilities

def main():
	# initialize the utilites class. donwloads mnist data and initializes input variable x,
	# predicted output label valriable y_
	num_iterations = 20000
	batch_size = 50
	learning_rate = 0.5
	utility_obj = Utilities(num_iterations, batch_size, learning_rate)
	
	# Read the USPS data from proj3_images folder and store for further use
	utility_obj.get_usps_data()

	# create the logistic regression model, train using mnist data and test it using mnist and usps data set
	logistic_regression(utility_obj)

	# create single layer neural network model, train using mnist and test it using mnist and usps
	num_neurons = 100
	# single_layer_nn(utility_obj,num_neurons)

	# create convolutional neural network model, train using mnist and test it using mnist and usps
	# train_cnn(utility_obj)


def logistic_regression(utility_obj):
	W = tf.Variable(tf.zeros([784, 10]))  # weights
	b = tf.Variable(tf.zeros([10])) # bias

	y = tf.matmul(utility_obj.x_input, W) + b # logistic regression model
	alphas = np.arange(0.05, 1.05, 0.05)
	accuracy = np.zeros(len(alphas))

	for index, value in enumerate(alphas):
		print('########## Logistic Regression Model Training ##########')
		utility_obj.learning_rate = value
		logistic_reg_accuracy, logistic_reg_sess = utility_obj.compute_model_training(y)
		accuracy[index] = logistic_reg_sess.run(logistic_reg_accuracy, feed_dict={utility_obj.x_input: utility_obj.usps_test_images,
		utility_obj.y_labels: utility_obj.usps_test_labels})

	plt.plot(alphas, accuracy)
	plt.xticks(alphas)
	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy')
	plt.title('Logistic Regression (USPS)')
	plt.savefig('./plots/logistic_regression_alpha_vs_accuracy_usps.png')

	print('MNIST Logistic Regression Accuracy =', logistic_reg_sess.run(
		logistic_reg_accuracy, feed_dict={utility_obj.x_input: utility_obj.mnist.test.images,
		utility_obj.y_labels: utility_obj.mnist.test.labels}))

	print('USPS Logistic Regression Accuracy =', logistic_reg_sess.run(
		logistic_reg_accuracy, feed_dict={utility_obj.x_input: utility_obj.usps_test_images,
		utility_obj.y_labels: utility_obj.usps_test_labels}))
	logistic_reg_sess.close()

def single_layer_nn(utility_obj, num_neurons):
	#### the below code is used for ploting the graph by varying the number of neurons in step of 10
	# arr_size = int((num_neurons) / 10)
	# accuracy = [0]*arr_size
	# neurons = [0]*arr_size
	# index = 0

	# for i in range(10, num_neurons+10, 10):
	# 	hidden_layer = tf.layers.dense(utility_obj.x_input, num_neurons, activation=tf.nn.relu)
	# 	h_1 = tf.layers.dense(hidden_layer, 10) # number of classes in which the data is to be classified
	# 	print('num of neurons', i)
	# 	print('########## Sinle layer NN training ##########')
	# 	snn_accuracy, snn_sess = utility_obj.compute_model_training(h_1)
	# 	index = int((i - 10) / 10)
	# 	neurons[index] = i
	# 	accuracy[index] = snn_sess.run(snn_accuracy, feed_dict={utility_obj.x_input: utility_obj.mnist.test.images, utility_obj.y_labels: utility_obj.mnist.test.labels})

	# print('neurons=', neurons)
	# print('accuracy=', accuracy)
	# plt.plot(neurons, accuracy)
	# plt.title('Single Layer NN (MNIST)')
	# plt.ylabel('Accuracy')
	# plt.xlabel('Number of Neurons')
	# plt.savefig('./plots/snn_neurons_vs_accuracy.png')

	hidden_layer = tf.layers.dense(utility_obj.x_input, num_neurons, activation=tf.nn.relu)
	h_1 = tf.layers.dense(hidden_layer, 10) # number of classes in which the data is to be classified
	print('########## Sinle layer NN training ##########')
	snn_accuracy, snn_sess = utility_obj.compute_model_training(h_1)

	print('Single Layer NN MNIST Accuracy = ',snn_sess.run(snn_accuracy,
		feed_dict={utility_obj.x_input: utility_obj.mnist.test.images, utility_obj.y_labels: utility_obj.mnist.test.labels}))

	print('USPS Single Layer NN Accuracy =', snn_sess.run(snn_accuracy,
		feed_dict={utility_obj.x_input: utility_obj.usps_test_images, utility_obj.y_labels: utility_obj.usps_test_labels}))
	snn_sess.close()


def train_cnn(utility_obj):
	W_conv1 = utility_obj.weight_variable([5, 5, 1, 32])
	b_conv1 = utility_obj.bias_variable([32])
	x_image = tf.reshape(utility_obj.x_input, [-1, 28, 28, 1])
	h_conv1 = tf.nn.relu(utility_obj.conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = utility_obj.max_pool_2x2(h_conv1)

	# second layer in convolutional neural network
	W_conv2 = utility_obj.weight_variable([5, 5, 32, 64])
	b_conv2 = utility_obj.bias_variable([64])

	h_conv2 = tf.nn.relu(utility_obj.conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = utility_obj.max_pool_2x2(h_conv2)

	W_fc1 = utility_obj.weight_variable([7 * 7 * 64, 1024])
	b_fc1 = utility_obj.bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# drop random neurons to avoid overfitting
	
	h_fc1_drop = tf.nn.dropout(h_fc1, utility_obj.keep_prob)

	# we now perform the prediction using the basic logistic approach to compute the softmax
	# and then evaluating the output
	W_fc2 = utility_obj.weight_variable([1024, 10])
	b_fc2 = utility_obj.bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	utility_obj.learning_rate = 1e-4
	utility_obj.optimizer = tf.train.AdamOptimizer(utility_obj.learning_rate)
	utility_obj.nn = 'cnn'

	cnn_accuracy, cnn_sess = utility_obj.compute_model_training(y_conv)

	## Testing the model
	print('CNN MNIST Accuracy = ', cnn_sess.run(cnn_accuracy,
		feed_dict={utility_obj.x_input: utility_obj.mnist.test.images,
		utility_obj.y_labels: utility_obj.mnist.test.labels, utility_obj.keep_prob: 1.0}))

	print('CNN USPS Accuracy = ', cnn_sess.run(cnn_accuracy,
		feed_dict={utility_obj.x_input: utility_obj.usps_test_images,
		utility_obj.y_labels: utility_obj.usps_test_labels, utility_obj.keep_prob: 1.0}))

	cnn_sess.close()


if __name__ == '__main__':
	main()
