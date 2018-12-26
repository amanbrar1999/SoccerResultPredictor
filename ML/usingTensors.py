import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a = tf.get_variable("a", initializer = tf.constant([[1,2],[3,4]], dtype = tf.int32))
b = tf.get_variable("b", initializer = tf.constant([[1],[1]], dtype = tf.int32))
c = tf.get_variable("c", initializer = tf.matmul(a,b), dtype = tf.int32)


# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Save the variables to disk.
  save_path = saver.save(sess, "../DataSets/model.ckpt")
  print("Model saved in path: %s" % save_path)