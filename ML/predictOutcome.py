import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# From results.txt:
# RESULTS------------------------------
# Optimal Home Win Model is in: hw2 
# theta: [[ 0.10498991]
#  [-0.94905062]
#  [ 0.74825934]
#  [ 0.84990819]] 
# cost: 5.2527308180314325 
# Optimal Draws Model is in: d10 
# theta: [[ 0.33612894]
#  [-0.14059028]
#  [-0.06648189]
#  [ 0.61908273]] 
# cost: 1.3382759314324848 
# Optimal Home Loss Model is in: hl0 
# theta: [[-1.77296921]
#  [-0.17580398]
#  [ 0.09614677]
#  [ 0.26889068]] 
# cost: 1.111997144379274 

def sigmoid(theta, x):
  m = x.get_shape()[0]
  xWithOnes = tf.concat([tf.ones([m,1], tf.float64), x], 1)
  exponent = tf.matmul(xWithOnes, theta)
  return 1 / (1 + tf.math.exp(-exponent))

# To apply one vs all logistic regression, we run a data point against each model and pick the one with the highest precision
# using our hypothesis function, aka the sigmoid function (this will return a value between 0 and 1)

HomeWinsModel = tf.Variable(
  [[ 0.10498991],
    [-0.94905062],
    [ 0.74825934],
    [ 0.84990819]], dtype = tf.float64)
DrawsModel = tf.Variable(
  [[ 0.33612894],
    [-0.14059028],
    [-0.06648189],
    [ 0.61908273]], dtype = tf.float64)
HomeLossModel = tf.Variable(
  [[-1.77296921],
    [-0.17580398],
    [ 0.09614677],
    [ 0.26889068]], dtype = tf.float64)

# test_games_matrix = tf.get_variable("test_games_matrix", shape=[2292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
# test_results_matrix = tf.get_variable("test_results_matrix", shape=[2292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
# training_games_matrix = tf.get_variable("training_games_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)
# training_results_matrix = tf.get_variable("training_results_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)

# saver = tf.train.Saver({ 
#   "training_games_matrix": training_games_matrix, 
#   "training_results_matrix": training_results_matrix, 
#   "test_games_matrix" : test_games_matrix,
#   "test_results_matrix" : test_results_matrix 
# })
  
with tf.Session() as sess:
  # saver.restore(sess, "../models/dataMatrices.ckpt")
  # print("Model restored.")
  # # Check the values of the variables
  # print("test_games_matrix : %s" % test_games_matrix.eval())
  # print("test_results_matrix : %s" % test_results_matrix.eval())
  # print("training_games_matrix : %s" % training_games_matrix.eval())
  # print("training_results_matrix : %s" % training_results_matrix.eval())
  