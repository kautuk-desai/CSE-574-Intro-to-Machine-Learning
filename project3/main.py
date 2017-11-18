import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import PIL.ImageOps as ImageOps
from utilities import Utilities


def main():
	# initialize the utilites class. donwloads mnist data and initializes input variable x,
	# predicted output label valriable y_
	num_iterations = 20000
	batch_size = 50
	utility_obj = Utilities(num_iterations, batch_size)
	
	# Read the USPS data from proj3_images folder and store for further use
	utility_obj.get_usps_data()

	# create the logistic regression model, train using mnist data and test it using mnist and usps data set
	logistic_regression(utility_obj)

	# create single layer neural network model, train using mnist and test it using mnist and usps
	num_neurons = 100
	single_layer_nn(utility_obj,num_neurons)

	# create convolutional neural network model, train using mnist and test it using mnist and usps
	# train_cnn(utility_obj)


def logistic_regression(utility_obj):
	W = tf.Variable(tf.zeros([784, 10]))  # weights
	b = tf.Variable(tf.zeros([10])) # bias

	y = tf.matmul(utility_obj.x_input, W) + b # logistic regression model
	print('########## Logistic Regression Model Training ##########')
	logistic_reg_accuracy, logistic_reg_sess = utility_obj.compute_model_training(y)

	print('MNIST Logistic Regression Accuracy =', logistic_reg_sess.run(
		logistic_reg_accuracy, feed_dict={utility_obj.x_input: utility_obj.mnist.test.images, utility_obj.y_labels: utility_obj.mnist.test.labels}))

	print('USPS Logistic Regression Accuracy =', logistic_reg_sess.run(
		logistic_reg_accuracy, feed_dict={utility_obj.x_input: utility_obj.usps_test_images, utility_obj.y_labels: utility_obj.usps_test_labels}))
	logistic_reg_sess.close()

def single_layer_nn(utility_obj, num_neurons):
	hidden_layer = tf.layers.dense(utility_obj.x_input, num_neurons, activation=tf.nn.relu)
	h_1 = tf.layers.dense(hidden_layer, 10) # number of classes in which the data is to be classified

	print('########## Sinle layer NN training ##########')
	snn_accuracy, snn_sess = utility_obj.compute_model_training(h_1)

	print('Single Layer NN MNIST Accuracy = ', snn_sess.run(
		snn_accuracy, feed_dict={utility_obj.x_input: utility_obj.mnist.test.images, utility_obj.y_labels: utility_obj.mnist.test.labels}))

	print('USPS Single Layer NN Accuracy =', snn_sess.run(
		snn_accuracy, feed_dict={utility_obj.x_input: utility_obj.usps_test_images, utility_obj.y_labels: utility_obj.usps_test_labels}))
	snn_sess.close()


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train_cnn(x, y_):
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# second layer in convolutional neural network
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# drop random neurons to avoid overfitting
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# we now perform the prediction using the basic logistic approach to compute the softmax
	# and then evaluating the output
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as cnn_sess:
		cnn_sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if (i % 100 == 0):
				train_accuracy = accuracy.eval(
					feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))

			train_step.run(
				feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

		print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
	main()
