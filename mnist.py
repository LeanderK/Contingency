from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contingency

import argparse
import sys
import tempfile

import tensorflow as tf
import numpy as np
from numpy import random as nprandom

from sklearn.metrics import roc_curve, auc

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
max_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# Adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Create the neural network
def conv_net(x_dict, n_classes, dropout, is_training, should_reuse):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=should_reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# Define the model function
def model_fn(features, labels, num_classes, is_training, should_reuse):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits = conv_net(features, num_classes, dropout, is_training=is_training, should_reuse=should_reuse)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    correct_prediction = tf.equal(pred_classes, tf.cast(labels, dtype=tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Evaluate the accuracy of the model
    #acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    
    return (loss_op, pred_probas, accuracy)

class MNistContingency(contingency.Contingency):
    def __init__(self, learning_rate, num_classes):
        if num_classes > max_classes:
            raise ValueError('Max classes is ', max_classes, ' not ', num_classes)
        contingency.Contingency.__init__(self, learning_rate, learning_rate, num_input, num_classes, model_fn)

def run(run_fn, mnist_cont, batch_size, num_adversarial, num_steps): 
    images = tf.placeholder(tf.float32, shape=[None, mnist_cont.num_input], name="images")
    labels = tf.placeholder(tf.float32, shape=[None], name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")
    with tf.Session() as session:
        (acc_eval, pred_eval, train_fn) = run_fn(images, labels, is_training)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        session.run(tf.local_variables_initializer())
        # Build the Estimator

        for iteration in range(num_steps):
            (training, cont_training) = mnist_cont.next_batch(batch_size, (batch_size - num_adversarial), mnist.train)
            (a, cont) = train_fn(iteration, session, training, cont_training)
            # a = session.run(accEval, feed_dict={images.name: train_im, labels.name: train_la})
            if iteration % 50 == 0:
                print("Training Accuracy in iteration ", iteration, ":", a)

        (eval_rel_im, eval_rel_la) = mnist_cont.relabel(mnist.test)
        (eval_valid_im, eval_valid_la) = mnist_cont.only_valid(mnist.test.images, mnist.test.labels)
        #a = session.run(acc_eval, feed_dict={images: eval_rel_im, labels: eval_rel_la, is_training.name: False})
        #print("Final Accuracy on all relabeled classes", iteration, ":", a)
        a = session.run(acc_eval, feed_dict={images: eval_valid_im, labels: eval_valid_la, is_training.name: False})
        print("Final Accuracy on only valid classes", iteration, ":", a)
        (unex_im, unex_la) = mnist_cont.unexpected_data(mnist.test)
        a = session.run(acc_eval, feed_dict={images: unex_im, labels: unex_la, is_training.name: False})
        print("Final Accuracy on unexpected data", iteration, ":", a)

        #TODO is this right?
        # from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        # Compute ROC curve and ROC area for each class
        # BEWARE: the zero class here is unexpected input, not the zero class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        (data, labels) = mnist_cont.relabel_roc(mnist.test)
        pred = session.run(pred_eval, 
                feed_dict={images: data, is_training.name: False})
        pred_roc = 1 - pred[:,0]
        fpr[0], tpr[0], _ = roc_curve(labels, pred_roc)
        roc_auc[0] = auc(fpr[0], tpr[0])

        return (fpr, tpr, roc_auc, pred)