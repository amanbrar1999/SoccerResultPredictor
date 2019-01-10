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

class predictor:

  def __init__(self, homeWins, draws, homeLoss):
    # we will concatenate the individual models into a single matrix to make our calculation code a bit neater
    self.LogisticRegressionModel = tf.transpose(tf.concat([homeWins, draws, homeLoss], 1))
  
  @staticmethod
  def sigmoidSingle(theta, x):
    xWithOnes = tf.concat([tf.ones([1,1], tf.float64), x], 0)
    exponent = tf.matmul(theta, xWithOnes)
    return 1 / (1 + tf.math.exp(-exponent))
  
  @staticmethod
  def sigmoid(theta, x):
    m = x.get_shape()[0]
    xWithOnes = tf.concat([tf.ones([m,1], tf.float64), x], 1)
    exponent = tf.matmul(xWithOnes, tf.transpose(theta))
    return 1 / (1 + tf.exp(-exponent))
  
  def predictSingleLogisticRegression(self, game):
    # assuming game is entered as a 3x1 tensor
    return predictor.sigmoidSingle(self.LogisticRegressionModel, game)
  
  @staticmethod
  def switchRow(num):
    if num == 0:
      return [ 1., 0., 0. ]
    elif num == 1: 
      return [ 0., 1., 0. ]
    elif num == 2:
      return [ 0., 0., 1. ]

  def predictMultipleLogisticRegression(self, game):
    # game will be entered as a m x 3 matrix
    results = predictor.sigmoid(self.LogisticRegressionModel, game)
    # return tf.transpose([tf.argmax(results, axis = 1)])
    max_index_list =  tf.argmax(results, axis = 1)
    array = np.array(max_index_list.eval())
    mapping = lambda x: [ 1., 0., 0. ] if x == 0 else ([ 0., 1., 0. ] if x == 1 else [ 0., 0., 1. ])
    return tf.constant(np.array([mapping(x) for x in array]))
