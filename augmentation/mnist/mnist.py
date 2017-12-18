from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from augmentation import augmentation, augmentation_data

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

max_dist = 3
#max_dist = mnist.MNISTAugmentation.mnist_max_dist(mnist.MNISTAugData(num_classes))

# Create the neural network
def conv_net(x_dict, n_classes, dropout, is_training, should_reuse):
    summaries = []
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=should_reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu, name='conv1')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv2')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024, name='fc1')
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes, name='out')

        for tensor in tf.global_variables():
            name = (tensor.name + '_histogram').replace(':', '_')
            summ = tf.summary.histogram(name, tensor)
            summaries.append(summ)

    return (out, summaries)

# Define the model function
def model_fn(features, labels, num_classes, is_training, should_reuse):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    (logits, summaries) = conv_net(features, num_classes, dropout, is_training=is_training, should_reuse=should_reuse)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    correct_prediction = tf.equal(pred_classes, tf.cast(labels, dtype=tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summaries.append(tf.summary.scalar("accuracy", accuracy))

    # Evaluate the accuracy of the model
    #acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # Define loss and optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32))
    summaries.append(tf.summary.histogram("crosse-entropy", cross_entropy))
    loss_op = tf.reduce_mean(cross_entropy)
    summaries.append(tf.summary.scalar("loss", loss_op))

    return {'loss_op':loss_op, 'pred_op':pred_probas, 'acc_op':accuracy, 'summ_op': tf.summary.merge(summaries)}

class MNISTAugmentation(augmentation.Augmentation):
    def __init__(self, learning_rate_adv, num_adversarial, num_adversarial_train, aug_data, gen_aug_labels):
        augmentation.Augmentation.__init__(self, learning_rate_adv, num_adversarial, num_adversarial_train
                                        , num_input, model_fn, aug_data, max_dist, gen_aug_labels)
    @staticmethod
    def mnist_max_dist(aug_data):
        return augmentation.Augmentation.calc_max_dist(aug_data.get_valid_training_data()[0], num_input)

class MNISTAugData(augmentation_data.AugmentationData):
    def __init__(self, num_classes):
        if num_classes > max_classes:
            raise ValueError('Max classes is ', max_classes, ' not ', num_classes)
        augmentation_data.AugmentationData.__init__(self, mnist.train.images, mnist.train.labels, mnist.test.images
                                                , mnist.test.labels, mnist.validation.images, mnist.validation.labels
                                                , num_classes, num_input)

def set_up(aug_obj):
    images = tf.placeholder(tf.float32, shape=[None, num_input], name="images")
    labels = tf.placeholder(tf.float32, shape=[None, aug_obj.getAugmentationData().get_num_classes()], name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")
    return (images, labels, is_training)

def run(run_fn, learning_rate, num_adversarial, aug_obj, batch_size, num_steps): 
    (images, labels, is_training) = set_up(aug_obj)
    aug_data_obj = aug_obj.getAugmentationData()
    with tf.Session() as session:
        model = run_fn(features = images, learning_rate = learning_rate, labels = labels, is_training = is_training)
        summary_writer = tf.summary.FileWriter('tensorboard/train',
                                            session.graph)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        session.run(tf.local_variables_initializer())

        for iteration in range(num_steps):
            (aug_data, aug_lbls) = model['train_fn'](iteration, session, batch_size, batch_size)
            # a = session.run(accEval, feed_dict={images.name: train_im, labels.name: train_la})
            if iteration % 50 == 0:
                (test_data, test_lbls) = aug_data_obj.next_test_batch(batch_size)
                feed_dict = {images: test_data, labels: test_lbls, is_training.name: False}
                summarize(session, model, feed_dict, iteration, summary_writer)

        return aug_obj.eval(session, model, images, labels, is_training)

def summarize(session, model, tf_dict, iteration, summary_writer):
        (summ, acc) = session.run(
            [model['summ_op'], model['acc_op']], 
            feed_dict=tf_dict
        )
        print("Training Accuracy in iteration ", iteration, ":", acc)
        summary_writer.add_summary(summ, iteration)
        summary_writer.flush()