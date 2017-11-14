import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.image as mpimg
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
	img = mpimg.imread('./proj3_images/Test/test_0003.png')

	image = np.resize(img, (28,28))
	image = image.flatten()

	label = np.zeros(10)
	label[9] = 1;

	print('Classification error rate: ', sess.run(accuracy, feed_dict = {x: [image], y_: [label]}))



if __name__ == '__main__':
	main()