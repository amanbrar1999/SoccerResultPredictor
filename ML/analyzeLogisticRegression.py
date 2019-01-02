import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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