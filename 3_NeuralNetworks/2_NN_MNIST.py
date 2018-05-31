'''
This is a simple example of using MNIST data and
do training with simple DNN (2-hidden-layers).
There are two ways to do the training in this code:
    1. one way is the normal way with individual events
    2. another way is to use the average event for each class
        to train a NN for that class

MNIST database of handwritten digits: http://yann.lecun.com/exdb/mnist/.
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)


# Training settings.
learning_rate = 0.1
training_epochs = int(20) # each epoch uses the whole dataset
batch_size = int(50000)
display_step = int(1)


# Define NN.
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# Input.
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
X_ave = tf.placeholder("float", [num_input])
Y_ave = tf.placeholder("float", [num_classes])


# Store layers weight & bias.
weights = {
    'out': tf.Variable(tf.random_normal([num_input, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create linear model.
def neural_net(x):
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.relu( tf.matmul(x, weights['out']) )
    #out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer


# Construct model.
logits = neural_net(X)
prediction = tf.nn.softmax(logits)


# Define loss (cross entropy) and optimizer.
# tf.reduce_sum computes the sum of elements across dimensions of a tensor.
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model.
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables.
# This initializer is an op that initializes global variables in the graph.
init = tf.global_variables_initializer()

# Store vectors for each batch, calculate average.
def average_data(data,size):
    ave = data[0]
    for s in range(1,size):
        ave = ave + data[s]
    ave = tf.divide(ave, size)

    average = []
    for n in range(size):
        average.append( ave )
    return average

# calculate average event for one class
def average_one(data,size):
    ave = data[0]
    for s in range(1,size):
        ave = ave + data[s]
    ave = tf.divide(ave, size)
    return ave


# Main loop ---------------------------------------------------------------------
# Use "with...as" statement to try safely.
# A Session object encapsulates the environment in which
# Operation objects are executed and 
# Tensor objects are evaluated.
with tf.Session() as sess:

    # session.run() runs operations and
    # evaluates tensors in fetches.
    sess.run(init)
    '''
    #############################################################################
    # Training with normal events.
    # Each epoch uses the whole training dataset.
    for epoch in range(1, training_epochs+1):
        # train.next_batch() returns a tuple of two arrays.
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop).
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if epoch % display_step == 0 or epoch == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
            print("Step " + "{:5d}".format(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    #############################################################################
    '''
    '''
    #############################################################################
    # Training with average input for each class (10 NN for 10 classes)
    counter = 0
    num = int(0)
    num_test = int(9)
    print("Handwritten digit: ", num)

    for epoch in range(1, training_epochs+1):
        batch_class = int(0)
        batches_X = []
        batches_Y = []
        batches = mnist.train.next_batch(batch_size)

        for b in range(batch_size):
            # select event only for that class
            if ( tf.argmax(batches[1][b]).eval()==num ):
                batch_class += 1
                batches_X.append(batches[0][b]) # each element has size of 784
                batches_Y.append(batches[1][b]) # each element has size of 10
                counter += 1
                
        #print (len(batches_X))
        #print (batches_Y)
        average_X = average_data(batches_X,batch_class) 
        average_Y = average_data(batches_Y,batch_class) 
        #print (average_X)
        #print (average_Y)

        # Run optimization op (backprop).
        sess.run(train_op, feed_dict={X: average_X, Y: average_Y})

        if epoch % display_step == 0 or epoch == 1:
            # Calculate batch loss and accuracy
            loss, acc, pred = sess.run([loss_op, accuracy, prediction], feed_dict={X: average_X, Y: average_Y})
            print("Step " + "{:5d}".format(epoch) + ", Minibatch Loss= " + "{:.9f}".format(loss) + \
                    ", Training Accuracy= " + "{:.3f}".format(acc) + \
                    ", Prediction= ", pred )

    # Calculate accuracy on test data.
    print("Test on digit: ", num_test)
    batch_class_test = int(0)
    batches_X_test = []
    batches_Y_test = []
    batches_test = mnist.test.next_batch(batch_size)
    for t in range(batch_size):
        if ( tf.argmax(batches_test[1][t]).eval()==num_test ):
            batch_class_test += 1
            batches_X_test.append(batches_test[0][t]) # each element is a tensor of 784*batch_size
            batches_Y_test.append(batches_test[1][t]) # each element is a tensor of 10*batch_size
    acc_test = sess.run(accuracy, feed_dict={X: batches_X_test, Y: batches_Y_test})
    print ("Testing events: ", batch_size)
    print ("Testing accuracy: ", acc_test)

    print("No.%d Optimization Finished!"%(num))
    #############################################################################
    '''

    #############################################################################
    # Training with 10 average events (1 NN for 10 classes)
    counter = 0

    # a list of average events
    # the list has a length of 10
    # each element is the average event for each class
    # calculation based on one batch
    average_X = []
    average_Y = []
    batches = mnist.train.next_batch(batch_size)

    # loop over all classes
    for num in range(10):
        # calculate average event for each class
        batch_class = int(0)
        batches_X = []
        batches_Y = []

        for b in range(batch_size):
            if ( tf.argmax(batches[1][b]).eval()==num ):
                batch_class += 1
                batches_X.append(batches[0][b]) 
                batches_Y.append(batches[1][b]) 
                counter += 1
        # append one average event to the list   
        average_X.append( average_one(batches_X,batch_class) )
        average_Y.append( average_one(batches_Y,batch_class) )
        
    #print (average_X)
    #print (average_Y)

    for epoch in range(1, training_epochs+1):
        # Run optimization op (backprop).
        # use 10 average events to do training
        sess.run(train_op, feed_dict={X: average_X, Y: average_Y})

        if epoch % display_step == 0 or epoch == 1:
            # Calculate batch loss and accuracy
            loss, acc, pred = sess.run([loss_op, accuracy, prediction], feed_dict={X: average_X, Y: average_Y})
            print("Step " + "{:5d}".format(epoch) + ", Minibatch Loss= " + "{:.9f}".format(loss) + \
                    ", Training Accuracy= " + "{:.3f}".format(acc) + \
                    ", Prediction= ", pred )

    # Calculate accuracy on test data.
    acc_test = sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels})
    print ("Testing accuracy: ", acc_test)

    print("Optimization Finished!")
    #############################################################################
    
