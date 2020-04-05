""" LeNet like CNN using tensorflow 1
"""
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.datasets import fashion_mnist
tf.disable_eager_execution()


def defineCNN(inp):
	""" Deines a CNN for image classification
	"""
	conv1 = tf.layers.conv2d(inp, 32, 5, activation=tf.nn.relu)
	max1 = tf.layers.max_pooling2d(conv1, 2, 2)
	conv2 = tf.layers.conv2d(max1, 64, 5, activation=tf.nn.relu)
	max2 = tf.layers.max_pooling2d(conv2, 2, 2)
	shape = (-1, 
			max2.shape[1] * max2.shape[2] * max2.shape[3])
	fc1 = tf.reshape(max2, shape)
	fc1 = tf.layers.dense(fc1, 1024)
	dropout1 = tf.layers.dropout(fc1, 0.5)
	out = tf.layers.dense(dropout1, 10, activation=tf.nn.softmax)

	return out


def train(epochs, batch_size=32):
	""" Train the CNN and output loss and accuracy
		at eac epoch
	"""
	inp = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
	labels = tf.placeholder(tf.int64, (None,))
	logits = defineCNN(inp)
	loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
	global_step = tf.train.get_or_create_global_step()
	optimizer = tf.train.AdamOptimizer().minimize(loss, global_step)

	predictions = tf.argmax(logits, axis=1)
	correct_predictions = tf.equal(labels, predictions)
	accuracy = tf.reduce_mean(
		tf.cast(correct_predictions, tf.float32)
	)

	init = tf.global_variables_initializer()

	(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
	X_train = X_train / 255
	X_test = X_test / 255
	X_train = np.expand_dims(X_train, -1)
	X_test = np.expand_dims(X_test, -1)

	num_batches = X_train.shape[0] // batch_size
	print("Batch size: ", batch_size)
	print("Number of batches per epoch", num_batches)

	sess = tf.Session()
	sess.run(init)

	saver = tf.train.Saver()
	writer = tf.summary.FileWriter("log/graph_loss", tf.get_default_graph())
	validation_writer = tf.summary.FileWriter("log/validation_loss")

	accuracy_summary = tf.summary.scalar("accuracy", accuracy)
	loss_summary = tf.summary.scalar("loss", loss)

	for epoch in range(epochs):
		# for batch in range(num_batches):
		for batch in range(num_batches):
			start_index = epoch * batch_size
			end_index = (epoch + 1) * batch_size

			loss_value, _, step = sess.run(
				[loss, optimizer, global_step],
				feed_dict={
					inp: X_train[start_index:end_index],
					labels: Y_train[start_index:end_index]
				}
			)

		saver.save(sess, "log/model")
		# Calculate train and validation accuracy
		start_index = 0
		end_index = 128
		train_accuracy, train_loss, \
			train_accuracy_summary, train_loss_summary = \
		sess.run(
			[accuracy, loss, accuracy_summary, loss_summary],
			feed_dict={inp: X_train[start_index:end_index],
						labels: Y_train[start_index:end_index]}
		)
		print("Epoch :", epoch + 1)
		print("\tTraining accuracy: {} Training loss: {}".format(
			train_accuracy, train_accuracy)
		)

		val_accuracy, val_loss, \
			val_accuracy_summary, val_loss_summary = \
		sess.run(
			[accuracy, loss, accuracy_summary, loss_summary],
			feed_dict={inp: X_test[start_index:end_index],
						labels: Y_test[start_index:end_index]}
		)
		print("\tValidation accuracy: {} Validation loss: {}".format(
			val_accuracy, val_loss)
		)

		writer.add_summary(train_accuracy_summary, step)
		writer.add_summary(train_loss_summary, step)
		validation_writer.add_summary(val_accuracy_summary, step)
		validation_writer.add_summary(val_loss_summary, step)

		writer.flush()
		validation_writer.flush()

	sess.close()


if __name__ == "__main__":
	train(epochs=10, batch_size=32)

