import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

def main():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	y = tf.matmul(x, W) + b
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=y))
	# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# this is for mini batch stochastic gradient descent
	for i in range(1000):
		x_train_batch, y_train_batch = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {x: x_train_batch, y_: y_train_batch})

	predicted_output = tf.argmax(y, 1)
	expected_output = tf.argmax(y_, 1)

	correct_prediction = tf.equal(predicted_output, expected_output)
	# correct_prediction = tf.Print(correct_prediction, [correct_prediction])

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# USPS data
	usps_data_count = 1500
	usps_per_digit_data = 150
	per_digit_label_counter = 0
	usps_test_images = np.ndarray([usps_data_count + 1, 784])
	image_label = 0
	usps_test_labels = np.ndarray([usps_data_count + 1, 10])

	for i in range(usps_data_count, 0, -1):
		file_path = './proj3_images/Test/test_' + "{0:0=4d}".format(i) + '.png'
		img = mpimg.imread(file_path)
		image = np.resize(img, (28,28))
		flattened_vector = image.flatten()
		usps_test_images[i] = flattened_vector

		if (per_digit_label_counter == usps_per_digit_data):
			image_label += 1
			per_digit_label_counter = 0

		per_digit_label_counter += 1
		label = np.zeros(10)
		label[image_label] = 1.0
		usps_test_labels[i] = label
		


	# print(len(usps_test_images))
	# print(len(usps_test_images[0]))
	# print(len(usps_test_labels))
	# print(len(usps_test_labels[0]))

	ip = usps_test_images[1:1501]
	# print('ip:',ip)
	op = usps_test_labels[1:1501]
	# print('label: ', op)

	print('predicted_output: ', sess.run(predicted_output, feed_dict = {x: ip, y_: op}))
	print('expected_output: ', sess.run(expected_output, feed_dict = {x: ip, y_: op}))
	print('Classification accuracy: ', sess.run(accuracy, feed_dict = {x: ip, y_: op}))



if __name__ == '__main__':
	main()