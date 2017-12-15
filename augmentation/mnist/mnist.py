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
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32))
    summaries.append(tf.summary.histogram("crosse-entropy", cross_entropy))
    loss_op = tf.reduce_mean(cross_entropy)
    summaries.append(tf.summary.scalar("loss", loss_op))

    return {'loss_op':loss_op, 'pred_op':pred_probas, 'acc_op':accuracy, 'summ_op': tf.summary.merge(summaries)}

class MNistAugmentation(augmentation.Augmentation):
    def __init__(self, learning_rate_adv, num_adversarial, num_adversarial_train, aug_data):
        augmentation.Augmentation.__init__(self, learning_rate_adv, num_adversarial, num_adversarial_train
                                        , num_input, model_fn, aug_data)
    def mnist_max_dist(aug_data):
        return augmentation.Augmentation.calcMaxDist(aug_data.get_valid_training_data()[0], num_input)

class MNISTContData(augmentation_data.AugmentationData):
    def __init__(self, num_classes):
        if num_classes > max_classes:
            raise ValueError('Max classes is ', max_classes, ' not ', num_classes)
        augmentation_data.AugmentationData.__init__(self, mnist.train.images, mnist.train.labels, mnist.test.images
                                                , mnist.test.labels, mnist.validation.images, mnist.validation.labels
                                                , num_classes, num_input)

def run(run_fn, learning_rate, num_adversarial, aug_data_obj, batch_size, num_steps): 
    images = tf.placeholder(tf.float32, shape=[None, num_input], name="images")
    labels = tf.placeholder(tf.float32, shape=[None], name="labels")
    is_training = tf.placeholder(tf.bool, name="is_training")
    with tf.Session() as session:
        model = run_fn(features = images, learning_rate = learning_rate, labels = labels, is_training = is_training)
        summary_writer = tf.summary.FileWriter('tensorboard/train',
                                            session.graph)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        session.run(tf.local_variables_initializer())
        # Build the Estimator

        for iteration in range(num_steps):
            (training, aug_training) = aug_data_obj.next_batch(batch_size, (batch_size - num_adversarial))
            (aug_data, aug_lbls) = model['train_fn'](iteration, session, training, aug_training)
            aug_data_obj.add_to_augmentation(aug_data, aug_lbls)
            # a = session.run(accEval, feed_dict={images.name: train_im, labels.name: train_la})
            if iteration % 50 == 0:
                (test_data, test_lbls) = aug_data_obj.next_test_batch(batch_size)
                (summ, acc) = session.run(
                    [model['summ_op'], model['acc_op']], 
                    feed_dict={images: test_data, labels: test_lbls, is_training.name: False}
                )
                print("Training Accuracy in iteration ", iteration, ":", acc)
                summary_writer.add_summary(summ, iteration)
                summary_writer.flush()

        (eval_valid_im, eval_valid_la) = aug_data_obj.get_valid_training_data()
        #a = session.run(model['acc_op'], feed_dict={images: eval_rel_im, labels: eval_rel_la, is_training.name: False})
        #print("Final Accuracy on all relabeled classes", iteration, ":", a)
        a = session.run(model['acc_op'], feed_dict={images: eval_valid_im, labels: eval_valid_la, is_training.name: False})
        print("Final Accuracy on only valid classes:", a)
        (unex_im, unex_la) = aug_data_obj.unexpected_data()
        a = session.run(model['acc_op'], feed_dict={images: unex_im, labels: unex_la, is_training.name: False})
        print("Final Accuracy on unexpected data:", a)
        (rand_im, rand_la) = aug_data_obj.generate_random(eval_valid_la.shape[0])
        a = session.run(model['acc_op'], feed_dict={images: rand_im, labels: rand_la, is_training.name: False})
        print("Final Accuracy on random data:", a)

        #TODO is this right?
        # from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        (data, labels) = aug_data_obj.relabel_roc()
        pred = session.run(model['pred_op'], 
                feed_dict={images: data, is_training.name: False})
        pred_roc = 1 - pred[:,0]
        fpr[0], tpr[0], _ = roc_curve(labels, pred_roc)
        roc_auc[0] = auc(fpr[0], tpr[0])

        return (fpr, tpr, roc_auc, pred, model)