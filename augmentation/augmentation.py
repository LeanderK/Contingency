from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import itertools
import math
from datetime import datetime 

import augmentation_data

import tensorflow as tf
import numpy as np
from numpy import random as nprandom

from sklearn.metrics import roc_curve, auc

class Augmentation:

    def __init__(self, learning_rate_adv, num_adversarial, num_adversarial_train, 
                num_input, model_fn, cont_data, max_dist):
        """
        Parameters
        ----------
        learning_rate_adv : int
            learning rate for the adversarial (if used)
        num_adversarial : int
            the number of adversarial examples generated per iteration
        num_adversarial_train : int
            adversarial training iterations (if used)
        cont_data : AugmentationData
            the data to train on
        """
        # Training Parameters
        self.learning_rate_adv = learning_rate_adv
        self.num_adversarial = num_adversarial
        self.num_adversarial_train = num_adversarial_train

        # Network Parameters
        self.num_input = num_input

        self.model_fn = model_fn

        self.augmentation_data = cont_data
        self.num_classes = cont_data.get_num_classes()
        (train_data, train_labels) = cont_data.get_valid_training_data()
        self.max_dist = max_dist

        with tf.name_scope('augmentation'):
            #from IPython.core.debugger import Tracer; Tracer()() 
            data = tf.convert_to_tensor(train_data, dtype=tf.float32, name="data")
            #calculate max euclidian distance in the training data
            self.max_image = tf.reduce_max(data,axis=0)
            self.min_image = tf.reduce_min(data,axis=0)
            #share of zero-labels in the dataset
            share_zeros = (np.where(train_labels == 0)[0].shape[0])/(train_labels.shape[0])
            self.loss_random_prediction = -np.log(share_zeros)

    def withoutAugmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('without_augmentation'):
            model = self.model_fn(features, labels, self.num_classes
                                                , is_training, False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(model['loss_op'],
                                        global_step=tf.train.get_global_step())

            def trainWithout(iteration, session, training, cont_training):
                (train_images, train_labels) = training
                t = session.run([train_op], feed_dict={
                        features.name: train_images,
                        labels.name: train_labels,
                        is_training.name: True})  
                empty_imges = np.empty(shape=(0, self.num_input))
                empty_labels = np.empty(shape=(0))
                empty = (empty_imges, empty_labels)
                return (empty)
            return {
                'acc_op':model['acc_op'], 
                'pred_op':model['pred_op'], 
                'train_fn':trainWithout, 
                'summ_op':model['summ_op']
            }

    def withRandomAugmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('with_random_augmentation'):
            return self.internalWithAugmentation(learning_rate, features, labels, is_training, 0)

    def withAugmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('with_adversarial_augmentation'):
            return self.internalWithAugmentation(learning_rate, features, labels, is_training, self.num_adversarial_train)

    def internalWithAugmentation(self, learning_rate, features, labels, is_training, num_adversarial_train):
        model = self.model_fn(features, labels, self.num_classes, is_training, False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        cont_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.num_input], name="cont_batch")
        cont_batch_la = tf.placeholder(dtype=tf.float32, shape=[None], name="cont_batch_labels")

        #scalar that controls augmentation 
        cont_beta = tf.placeholder(dtype=tf.float32, shape=[], name="cont_beta")
        cont_model = self.model_fn(cont_batch, cont_batch_la, self.num_classes, is_training, True)

        loss_with_cont = model['loss_op'] + cont_beta * cont_model['loss_op']
        train_op = optimizer.minimize(loss_with_cont,
                                    global_step=tf.train.get_global_step())

        #generating the contingengy
        gen_images = tf.get_variable(dtype=tf.float32, shape=[self.num_adversarial, self.num_input], name="gen_images")
        gen_labels = tf.placeholder(dtype=tf.float32, shape=[self.num_adversarial], name="gen_labels")

        gen_model = self.model_fn(gen_images, gen_labels, self.num_classes, is_training, True)
        distance = tf.reduce_mean(self.pairwiseL2Norm(gen_images, features), axis=1)
        adversial_fitness = gen_model['loss_op'] - (self.loss_random_prediction * self.max_dist)/distance
        #TODO maybe switch to Adam and reset it for each s(classifier-)training step? 
        optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_adv)
        adv_op = optimizer2.minimize(-1 * adversial_fitness,
                                    global_step=tf.train.get_global_step(),
                                    var_list=gen_images)

        def trainingContStep(iteration, session, training, cont_training):
            (train_images, train_labels) = training
            (cont_img, cont_labels) = cont_training

            # generates the congingency
            randomImages = nprandom.random((self.num_adversarial, self.num_input))
            zerolabels = np.zeros(self.num_adversarial)
            session.run(gen_images.assign(randomImages))

            for iteration in range(num_adversarial_train):
                #TODO does this work?
                a = session.run(adv_op, feed_dict={
                        gen_labels.name: zerolabels,
                        features.name: train_images,
                        #we are not training the weights!
                        is_training.name: False})

            adv_images = session.run(gen_images)
            cont_img = np.concatenate((cont_img, adv_images), axis=0)
            #TODO redo this, we can't switch augmentation labels right now
            cont_labels = np.zeros(cont_img.shape[0])

            t = session.run([train_op], feed_dict={
                    features.name: train_images,
                    labels.name: train_labels,
                    cont_batch.name: cont_img,
                    cont_batch_la.name: cont_labels,
                    cont_beta.name: 1,
                    is_training.name: True})  
            return (adv_images, zerolabels)
        return {
            'acc_op': model['acc_op'], 
            'pred_op': model['pred_op'], 
            'train_fn': trainingContStep, 
            'summ_op': model['summ_op']
        }

    def pairwiseL2Norm(x, y, num_input):
        x_reshaped = tf.reshape(x, shape=[-1, 1, num_input])
        y_reshaped = tf.reshape(y, shape=[1, -1, num_input])
        x_squared = x_reshaped*x_reshaped
        y_squared = y_reshaped*y_reshaped
        multiplied = x_reshaped * y_reshaped

        #x^2-2xy+y^2
        #((x_i)_xy)^2-2*((x_i)_xy)((y_j)_xy)+((y_j)_xy)^2
        combined = x_squared - 2*multiplied + y_squared

        summed = tf.reduce_sum(combined, axis=2)
        return tf.sqrt(summed)

    def calcMaxDist(dataset, num_input):
        length = dataset.shape[0]
        subLen= 50
        subsets = [tf.convert_to_tensor(dataset[n*subLen:(n+1)*subLen], dtype=tf.float32) for n in range(0, math.ceil(length/subLen))]
        products = list(itertools.product(subsets, repeat=2))
        print("numer of products", len(list(products)))
        def max_l2(pairing):
            (x, y) = pairing
            max_val = tf.reduce_max(tf.reshape(Augmentation.pairwiseL2Norm(x, y, num_input), shape=[-1, 1]))
            return max_val
        startTime= datetime.now() 
        akk = []
        step = 200
        for i in range(0, len(products), step):
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                session.run(tf.tables_initializer())
                session.run(tf.local_variables_initializer())
                results = tf.stack(list(map(max_l2, products[i:i+step])))
                akk.append(session.run(tf.reduce_max(results)))
                timeElapsed=datetime.now()-startTime 
                print("calculating max dist, i:", i, " time elapsed since start: ", timeElapsed)
        max_dist = math.max(akk)
        print("max dist", max_dist)
        return max_dist
    