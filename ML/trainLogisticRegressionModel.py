import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def costFunction(x, theta, y, regConst):
    m = float(int(x.get_shape()[0]))
    return -(1/m) * tf.reduce_sum(tf.math.multiply(y,tf.math.log(sigmoid(theta,x))) + tf.math.multiply(1-y,tf.math.log(1-sigmoid(theta,x)))) # + regConst/(2*m) * tf.reduce_sum(tf.math.square(theta[1:theta.get_shape()[0]]))

def oneVAllHomeWinsMatrix(results_matrix):
  return tf.reshape(results_matrix[:,0],[results_matrix.get_shape()[0],1])

def oneVAllDrawsMatrix(results_matrix):
  return tf.reshape(results_matrix[:,1],[results_matrix.get_shape()[0],1])

def oneVAllHomeLossMatrix(results_matrix):
  return tf.reshape(results_matrix[:,2],[results_matrix.get_shape()[0],1])

def sigmoid(theta, x):
  m = x.get_shape()[0]
  xWithOnes = tf.concat([tf.ones([m,1], tf.float64), x], 1)
  exponent = tf.matmul(xWithOnes, theta)
  return 1 / (1 + tf.math.exp(-exponent))

def gradientDescent(x, theta, y, gradConst, regConst):
  m = float(int(x.get_shape()[0]))
  xWithOnes = tf.concat([tf.ones([m,1], tf.float64), x], 1)
  newTheta = theta - gradConst*(1/m)*tf.reduce_sum(tf.multiply(sigmoid(theta,x)-y,xWithOnes))
  return newTheta

games_matrix = tf.get_variable("games_matrix", shape=[10292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
results_matrix = tf.get_variable("results_matrix", shape=[10292,3], dtype = tf.float64, initializer = tf.zeros_initializer)
training_games_matrix = tf.get_variable("training_games_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)
training_results_matrix = tf.get_variable("training_results_matrix", shape=[8000,3], dtype = tf.float64, initializer = tf.zeros_initializer)

# Add ops to save and restore all the variables.
saver = tf.train.Saver({ 
  "training_games_matrix": training_games_matrix, 
  "training_results_matrix": training_results_matrix, 
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
  # since our gradient descent is susceptible to local minima, we will run it several times and pick out the theta with
  # the smallest cost function value
  optimalHomeWinsTheta = tf.get_variable("optimalHomeWinsTheta", shape = [4,1], dtype = tf.float64, initializer = tf.zeros_initializer)
  optimalDrawsTheta = tf.get_variable("optimalDrawsTheta", shape = [4,1], dtype = tf.float64, initializer = tf.zeros_initializer)
  optimalHomeLossTheta = tf.get_variable("optimalHomeLossTheta", shape = [4,1], dtype = tf.float64, initializer = tf.zeros_initializer)
  optimalHWCost = 1000
  optimalDCost = 1000
  optimalHLCost = 1000
  optimalHWIndex = 0
  optimalDIndex = 0
  optimalHLIndex = 0
  homeWinsTheta = tf.get_variable("homeWinsTheta", shape = [4,1], dtype = tf.float64, initializer = tf.random_normal_initializer)
  drawsTheta = tf.get_variable("drawsTheta", shape = [4,1], dtype = tf.float64, initializer = tf.random_normal_initializer)
  homeLossTheta = tf.get_variable("homeLossTheta", shape = [4,1], dtype = tf.float64, initializer = tf.random_normal_initializer)
  sess.run(tf.variables_initializer([optimalHomeWinsTheta, optimalDrawsTheta, optimalHomeLossTheta, homeWinsTheta, drawsTheta, homeLossTheta]))
  for i in range(12):
    print("------------------------------------------------On Run: %s" % i)
    sess.run(tf.assign(homeWinsTheta,tf.random_normal(shape = [4,1], dtype = tf.float64)))
    sess.run(tf.assign(drawsTheta,tf.random_normal(shape = [4,1], dtype = tf.float64)))
    sess.run(tf.assign(homeLossTheta,tf.random_normal(shape = [4,1], dtype = tf.float64)))
    for j in range(25):
      sess.run(tf.assign(homeWinsTheta, gradientDescent(training_games_matrix, homeWinsTheta, oneVAllHomeWinsMatrix(training_results_matrix), .0007, 0.1)))
      sess.run(tf.assign(drawsTheta, gradientDescent(training_games_matrix, drawsTheta, oneVAllDrawsMatrix(training_results_matrix), .0007, 0.1)))
      sess.run(tf.assign(homeLossTheta, gradientDescent(training_games_matrix, homeLossTheta, oneVAllHomeLossMatrix(training_results_matrix), .0007, 0.1)))
      newHWCost = costFunction(training_games_matrix, homeWinsTheta, oneVAllHomeWinsMatrix(training_results_matrix),0.1).eval()
      newDCost = costFunction(training_games_matrix, drawsTheta, oneVAllDrawsMatrix(training_results_matrix),0.1).eval()
      newHLCost = costFunction(training_games_matrix, homeLossTheta, oneVAllHomeLossMatrix(training_results_matrix),0.1).eval()
      print('Home Wins -------------------')
      print(homeWinsTheta.eval())
      print(newHWCost)
      print('Draws -------------------')
      print(drawsTheta.eval())
      print(newDCost)
      print('Home Loss -------------------')
      print(homeLossTheta.eval())
      print(newHLCost)
      if newHWCost < optimalHWCost:
        optimalHWCost = newHWCost
        optimalHWIndex = i
        sess.run(tf.assign(optimalHomeWinsTheta,homeWinsTheta))
      if newDCost < optimalDCost:
        optimalDCost = newDCost
        optimalDIndex = i
        sess.run(tf.assign(optimalDrawsTheta,drawsTheta))
      if newHLCost < optimalHLCost:
        optimalHLCost = newHLCost
        optimalHLIndex = i
        sess.run(tf.assign(optimalHomeLossTheta,homeLossTheta))
    file = open("../models/thetas/homeWins/hw" + str(i) + ".txt", "w") 
    file.write(str(homeWinsTheta.eval()) + ", cost: " + str(newHWCost)) 
    file.close() 
    file = open("../models/thetas/draws/d" + str(i) + ".txt", "w") 
    file.write(str(drawsTheta.eval()) + ", cost: " + str(newDCost)) 
    file.close() 
    file = open("../models/thetas/homeLoss/hl" + str(i) + ".txt", "w") 
    file.write(str(homeLossTheta.eval()) + ", cost: " + str(newHLCost)) 
    file.close() 
  print("RESULTS------------------------------")
  print("Optimal Home Win Model is in: hw%s" % optimalHWIndex)
  print("theta: %s" % optimalHomeWinsTheta.eval())
  print("cost: %s" % optimalHWCost)
  print("Optimal Draws Model is in: d%s" % optimalDIndex)
  print("theta: %s" % optimalDrawsTheta.eval())
  print("cost: %s" % optimalDCost)
  print("Optimal Home Loss Model is in: hl%s" % optimalHLIndex)
  print("theta: %s" % optimalHomeLossTheta.eval())
  print("cost: %s" % optimalHLCost)
  file = open("../models/thetas/results.txt", "w") 
  file.write("RESULTS------------------------------\n")
  file.write("Optimal Home Win Model is in: hw%s \n" % optimalHWIndex)
  file.write("theta: %s \n" % optimalHomeWinsTheta.eval())
  file.write("cost: %s \n" % optimalHWCost)
  file.write("Optimal Draws Model is in: d%s \n" % optimalDIndex)
  file.write("theta: %s \n" % optimalDrawsTheta.eval())
  file.write("cost: %s \n" % optimalDCost)
  file.write("Optimal Home Loss Model is in: hl%s \n" % optimalHLIndex)
  file.write("theta: %s \n" % optimalHomeLossTheta.eval())
  file.write("cost: %s \n" % optimalHLCost)
  file.close() 