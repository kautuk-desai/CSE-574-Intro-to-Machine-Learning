import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])

	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	y = tf.matmul(x, W) + b
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=y))
	# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_* tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

	
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for i in range(1000):
		x_train_batch, y_train_batch = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict = {x: x_train_batch, y_: y_train_batch})


	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))



if __name__ == '__main__':
	main()