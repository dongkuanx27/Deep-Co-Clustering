import tensorflow as tf


def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)

    initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
    initial = initializer(shape)

    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

