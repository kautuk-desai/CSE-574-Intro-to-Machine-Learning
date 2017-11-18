import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import PIL.ImageOps as ImageOps


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 10])

    # train_cnn(x, y_)

    single_layer_nn(x, y_, mnist)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # this is for mini batch stochastic gradient descent
    for i in range(1000):
        x_train_batch, y_train_batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: x_train_batch, y_: y_train_batch})

    predicted_output = tf.argmax(y, 1)
    expected_output = tf.argmax(y_, 1)

    correct_prediction = tf.equal(predicted_output, expected_output)
    # correct_prediction = tf.Print(correct_prediction, [correct_prediction])

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy = tf.multiply(accuracy, 100)

    # USPS data
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
        normalized_data = normalize(image)
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

    # print(len(usps_test_images))
    # print(usps_test_images[0])
    # print(len(usps_test_labels))
    # print(len(usps_test_labels[0]))

    print('predicted_output: ', sess.run(predicted_output, feed_dict={x: usps_test_images, y_: usps_test_labels}))
    print('expected_output: ', sess.run(expected_output, feed_dict={x: usps_test_images, y_: usps_test_labels}))
    print('Classification accuracy: ', sess.run(accuracy, feed_dict={x: usps_test_images, y_: usps_test_labels}))


def normalize(data):
    row_sums = data.sum(axis=1)
    norm_matrix = np.divide(data, row_sums[:,np.newaxis])  # increase the dimension of the existing array by one more dimension
    normalized_data = 1 - norm_matrix
    normalized_data[normalized_data < 1] = 0
    return normalized_data

def single_layer_nn(x, y_, mnist):
	hidden_layer = tf.layers.dense(x, 100, activation=tf.nn.relu)
	h_1 = tf.layers.dense(hidden_layer, 10)

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_1))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(h_1, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print('######## Sinle layer NN training ########')
	with tf.Session() as snn_sess:
		snn_sess.run(tf.global_variables_initializer())
		for i in range(20000):
			batch = mnist.train.next_batch(50)
			if (i % 1000 == 0):
				train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
				print('step %d, training accuracy %g' % (i, train_accuracy))

			train_step.run(feed_dict={x: batch[0], y_: batch[1]})

		print('Single Layer MNIST Accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



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

    ## second layer in convolutional neural network
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    ## drop random neurons to avoid overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## we now perform the prediction using the basic logistic approach to compute the softmax
    ## and then evaluating the output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as cnn_sess:
        cnn_sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if (i % 100 == 0):
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main()
