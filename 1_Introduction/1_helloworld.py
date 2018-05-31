from __future__ import print_function
import tensorflow as tf

# Simple hello world using TensorFlow
# The operation is added as a node to the default graph.
# The value returned by the constructor represents the output of the Constant operation
hello = tf.constant('Hello, TensorFlow!') # a Constant op

# Session is a class for running TF operations
sess = tf.Session()

# Run the op
print(sess.run(hello))
