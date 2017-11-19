import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image


class Utilities:

	def __init__(self, num_iterations, batch_size, learning_rate):
		self.num_iterations = num_iterations
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
		print('mnist train images: ', len(self.mnist.train.images))
		self.y_labels = tf.placeholder(tf.float32, [None, 10])
		self.x_input = tf.placeholder(tf.float32, [None, 784])  # input images in vector shape of 784
		self.keep_prob = tf.placeholder(tf.float32) # used for cnn neurons droping probability
		self.nn = ''

	def normalize(self, data):
		row_sums = data.sum(axis=1)
		# increase the dimension of the existing array by one more dimension
		norm_matrix = np.divide(data, row_sums[:, np.newaxis])
		normalized_data = 1 - norm_matrix
		normalized_data[normalized_data < 1] = 0
		return normalized_data

	def get_usps_data(self):
		usps_data_count = 1500
		usps_per_digit_data = 150
		per_digit_label_counter = 0
		usps_test_images = np.ndarray([usps_data_count + 1, 784])
		image_label = 0
		usps_test_labels = np.ndarray([usps_data_count + 1, 10])
		required_size = (28, 28)

		for i in range(usps_data_count, 0, -1):
			file_path = './proj3_images/Test/test_' + "{0:0=4d}".format(i) + '.png'
			img = Image.open(file_path)
			img = img.resize(required_size)

			image = np.asarray(img)
			normalized_data = self.normalize(image)
			# plt.imsave("./temp/test_" + "{0:0=4d}".format(i) + ".png", normalized_data, cmap=cm.gray)
			flattened_vector = normalized_data.flatten()
			usps_test_images[i - 1] = flattened_vector

			if (per_digit_label_counter == usps_per_digit_data):
				image_label += 1
				per_digit_label_counter = 0

			per_digit_label_counter += 1
			label = np.zeros(10)
			label[image_label] = 1
			usps_test_labels[i - 1] = label

		self.usps_test_images = usps_test_images
		self.usps_test_labels = usps_test_labels

	def compute_model_training(self, prediction):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=self.y_labels, logits=prediction))
		train_step = self.optimizer.minimize(cross_entropy)

		correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y_labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		accuracy = tf.multiply(accuracy, 100)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		# this is for mini batch stochastic gradient descent
		for i in range(self.num_iterations):
			batch = self.mnist.train.next_batch(self.batch_size)
			if (i % 5000 == 0):
				if (self.nn == 'cnn'):
					train_accuracy = accuracy.eval(feed_dict={self.x_input: batch[0], self.y_labels: batch[1], self.keep_prob: 0.5})
				else:
					train_accuracy = accuracy.eval(feed_dict={self.x_input: batch[0], self.y_labels: batch[1]})

				print('step %d, training accuracy %g' % (i, train_accuracy))

			if (self.nn == 'cnn'):
				train_step.run(feed_dict={self.x_input: batch[0], self.y_labels: batch[1], self.keep_prob: 0.5})
			else:
				train_step.run(feed_dict={self.x_input: batch[0], self.y_labels: batch[1]})

		return accuracy, sess

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)


	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)


	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
