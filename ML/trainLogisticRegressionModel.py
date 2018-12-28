import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def costFunction(x, theta, y, regConst):
    m = int(x.get_shape()[0])
    return - (1/m) * tf.reduce_sum(tf.math.multiply(y,tf.math.log(sigmoid(theta,x))) + tf.math.multiply(1-y,tf.math.log(1-sigmoid(theta,x)))) + regConst/(2*m) * tf.reduce_sum(tf.math.square(theta[1:theta.get_shape()[0]]))

def sigmoid(theta, x):
    m = x.get_shape()[0]
    xWithOnes = tf.concat([tf.ones([m,1], tf.float32), x], 1)
    exponent = tf.matmul(xWithOnes, theta)
    return tf.math.sigmoid(-exponent)

games_matrix = tf.get_variable("games_matrix", shape=[10292,3], dtype = tf.float32, initializer = tf.zeros_initializer)
results_matrix = tf.get_variable("results_matrix", shape=[10292,3], dtype = tf.int32, initializer = tf.zeros_initializer)
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Restore variables from disk.
  saver.restore(sess, "../models/dataMatrices.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("games_matrix : %s" % games_matrix.eval())
  print("results_matrix : %s" % results_matrix.eval())
  theta = tf.get_variable("theta", shape = [4,1], dtype = tf.float32, initializer = tf.zeros_initializer)
  sess.run(tf.variables_initializer([theta]))
  # print(sigmoid(theta, games_matrix).eval())
  y = tf.get_variable("y", shape = [10292,1], dtype = tf.float32, initializer = tf.ones_initializer)
  sess.run(tf.variables_initializer([y]))
  print(costFunction(games_matrix, theta, y, 1).eval())
