from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# Adapted from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import tensorflow as tf
import numpy as np
from numpy import random as nprandom

# Training Parameters
learning_rate = 0.001
num_steps = 300
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units


# Create the neural network
def conv_net(x_dict, n_classes, dropout, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet'):
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
def model_fn(features, labels, is_training):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits = conv_net(features, num_classes, dropout, is_training=is_training)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # Evaluate the accuracy of the model
    acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    
    return (loss_op, acc, acc_op)

def withoutContingency(features, labels, is_training):
    (loss_op, acc, acc_op) = model_fn(features, labels, is_training)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                global_step=tf.train.get_global_step())

    def trainWithout(iteration, session, train_images, train_labels):
        a, a_op, t = session.run([acc, acc_op, train_op], feed_dict={
                features.name: train_images,
                labels.name: train_labels,
                is_training.name: True})  
        return a
       
    return (acc, trainWithout)

def withRandomContingency(features, labels):
    (loss_op, acc, acc_op) = model_fn(features, labels, is_training)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                global_step=tf.train.get_global_step())

    def trainingRandomStep(iteration, session, train_images, train_labels):
        randomImages = nprandom.random(num_input, 5)
        labels = np.zeroes(5)
        resultingImg = train_images.concatenate(randomImages)
        resultingLab = train_labels.concatenate(labels)
        a, a_op, t = session.run([acc, acc_op, train_op], feed_dict={
                features.name: resultingImg,
                labels.name: resultingLab,
                is_training.name: True})  
        return a
    return (acc, trainingRandomStep)

def run(model_fn): 
    images = tf.placeholder(tf.float32, shape=[None, num_input], name="images")
    labels = tf.placeholder(tf.float32, shape=[None], name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")
    with tf.Session() as session:
        (acc_eval, train_fn) = model_fn(images, labels, is_training)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        session.run(tf.local_variables_initializer())
        # Build the Estimator

        for iteration in range(num_steps):
            (train_im, train_la) = only_zero_one(*mnist.train.next_batch(batch_size))
            a = train_fn(iteration, session, train_im, train_la)
            # a = session.run(accEval, feed_dict={images.name: train_im, labels.name: train_la})
            if iteration % 100 == 0:
                print("Training Accuracy in iteration ", iteration, ":", a)

        (eval_im, eval_la) = relabel(mnist.test)
        a = session.run(acc_eval, feed_dict={images: eval_im, labels: eval_la, is_training.name: False})
        print("Final Accuracy ", iteration, ":", a)

def only_zero_one(images, labels):
    indices = np.where(labels <= 1 )
    return (images[indices], labels[indices])

def relabel(dataset):
    indices = np.where(dataset.labels != 1 )
    relabel = np.copy(dataset.labels)
    relabel[indices] = 0
    return (dataset.images, relabel)