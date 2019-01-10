import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from predictOutcome import predictor

games_matrix = tf.get_variable("games_matrix", shape=[10292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
results_matrix = tf.get_variable("results_matrix", shape=[10292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
training_games_matrix = tf.get_variable("training_games_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)
training_results_matrix = tf.get_variable("training_results_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)
test_games_matrix = tf.get_variable("test_games_matrix", shape=[2292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
test_results_matrix = tf.get_variable("test_results_matrix", shape=[2292,3], dtype = tf.float64, initializer = tf.zeros_initializer)

# Add ops to save and restore all the variables.
saver = tf.train.Saver({ 
  "training_games_matrix": training_games_matrix, 
  "training_results_matrix": training_results_matrix, 
  "test_games_matrix" : test_games_matrix,
  "test_results_matrix" : test_results_matrix,
  "games_matrix" : games_matrix,
  "results_matrix" : results_matrix 
})

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

game = tf.Variable(
  [[17],
    [1],
    [0.999999]], dtype = tf.float64)

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
  print("training_games_matrix : %s" % training_games_matrix.eval())
  print("training_results_matrix : %s" % training_results_matrix.eval())
  print("test_games_matrix : %s" % test_games_matrix.eval())
  print("test_results_matrix : %s" % test_results_matrix.eval())
  predict = predictor(HomeWinsModel, DrawsModel, HomeLossModel)
  predicted_on_training_set = predict.predictMultipleLogisticRegression(training_games_matrix)
  predicted_on_test_set = predict.predictMultipleLogisticRegression(test_games_matrix)
  training_acc, t_acc_up = tf.metrics.accuracy(training_results_matrix, predicted_on_training_set)
  test_acc, test_acc_up = tf.metrics.accuracy(test_results_matrix, predicted_on_test_set)
  sess.run(tf.local_variables_initializer())
  sess.run(t_acc_up)
  sess.run(test_acc_up)
  print("training set predictions: %s" % predicted_on_training_set.eval())
  print("accuracy on training set: %s" % training_acc.eval())
  print("test set predictions: %s" % predicted_on_test_set.eval())
  print("accuracy on test set: %s" % test_acc.eval())