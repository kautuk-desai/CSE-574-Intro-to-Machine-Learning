from PIL import Image
import numpy as np
import tensorflow as tf


class Utilities(object):
    """docstring for ClassName"""

    def __init__(self, file_path):
        super(Utilities, self).__init__()
        self.file_path = file_path
        self.image_size = (28, 28)
        self.x_input = tf.placeholder(tf.float32, [None, 784])  # input images in vector shape of 784
        self.y_labels = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)  # used for cnn neurons droping probability
        self.num_iterations = 300
        self.batch_size = 5000
        self.learning_rate = 1e-4
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def load_images(self, num_images, file_names):
        data = [[]] * num_images
        for i in range(num_images):
            im = Image.open(self.file_path + file_names[i])
            im = im.convert(mode='L')
            resized_im = im.resize(self.image_size)
            # resized_im.show()
            flattened_im = np.asarray(resized_im).flatten()
            data[i] = flattened_im

        return data

    def compute_model_training(self, prediction, training_data, training_label):
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
            input_batch = training_data[i*self.batch_size : (i+1)*self.batch_size]
            output_label = training_label[i*self.batch_size : (i+1)*self.batch_size]
            if (i % 5000 == 0):
                train_accuracy = accuracy.eval(feed_dict={self.x_input: input_batch, self.y_labels: output_label, self.keep_prob: 0.5})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={self.x_input: input_batch, self.y_labels: output_label, self.keep_prob: 0.5})

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
