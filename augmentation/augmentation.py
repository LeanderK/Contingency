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
                num_input, model_fn, aug_data, max_dist, gen_aug_labels):
        """
        Parameters
        ----------
        learning_rate_adv : int
            learning rate for the adversarial (if used)
        num_adversarial : int
            the number of adversarial examples generated per iteration
        num_adversarial_train : int
            adversarial training iterations (if used)
        aug_data : AugmentationData
            the data to train on
        gen_aug_labels : int -> np.array
            generates n labels for the augmentation data
        """
        # Training Parameters
        self.learning_rate_adv = learning_rate_adv
        self.num_adversarial = num_adversarial
        self.num_adversarial_train = num_adversarial_train

        # Network Parameters
        self.num_input = num_input

        self.model_fn = model_fn

        self.augmentation_data = aug_data
        self.num_classes = aug_data.get_num_classes()
        (train_data, train_labels) = aug_data.get_valid_training_data()
        self.max_dist = max_dist
        self.gen_aug_labels = gen_aug_labels

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

            def trainWithout(iteration, session, training, aug_training):
                augmentation_data.next_batch()
                (train_images, train_labels) = training
                t = session.run([train_op], feed_dict={
                        features.name: train_images,
                        labels.name: train_labels,
                        is_training.name: True})  
                empty_imges = np.empty(shape=(0, self.num_input))
                empty_labels = np.empty(shape=(0, self.num_classes))
                empty = (empty_imges, empty_labels)
                return (empty)
            return {
                'acc_op':model['acc_op'], 
                'pred_op':model['pred_op'], 
                'train_fn':trainWithout, 
                'summ_op':model['summ_op']
            }

    def with_random_augmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('with_random_augmentation'):
            return self.internal_with_augmentation(learning_rate, features, labels, is_training, 0)

    def with_augmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('with_adversarial_augmentation'):
            return self.internal_with_augmentation(learning_rate, features, labels, is_training, self.num_adversarial_train)

    def internal_with_augmentation(self, learning_rate, features, labels, is_training, num_adversarial_train):
        model = self.model_fn(features, labels, self.num_classes, is_training, False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        aug_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.num_input], name="aug_batch")
        aug_batch_la = tf.placeholder(dtype=tf.float32, shape=[None], name="aug_batch_labels")

        #scalar that controls augmentation 
        aug_beta = tf.placeholder(dtype=tf.float32, shape=[], name="aug_beta")
        aug_model = self.model_fn(aug_batch, aug_batch_la, self.num_classes, is_training, True)

        loss_with_cont = model['loss_op'] + aug_beta * aug_model['loss_op']
        train_op = optimizer.minimize(loss_with_cont,
                                    global_step=tf.train.get_global_step())

        gen_aug = set_up_augmentation_generation(learning_rate, features, labels, is_training)

        def training_aug_step(iteration, session, batch_size_training, batch_size_augmentation):
            (training, aug_training) = self.augmentation_data.next_batch(batch_size_training, batch_size_augmentation)
            (train_images, train_labels) = training
            (aug_img, aug_labels) = aug_training
            
            (adv_images, adv_lables) = gen_aug(num_adversarial_train)

            adv_images = session.run(gen_images)
            aug_img = np.concatenate((aug_img, adv_images), axis=0)
            aug_labels = np.concatenate((aug_labels, adv_lables), axis=0)

            t = session.run([train_op], feed_dict={
                    features.name: train_images,
                    labels.name: train_labels,
                    aug_batch.name: aug_img,
                    aug_batch_la.name: aug_labels,
                    aug_beta.name: 1,
                    is_training.name: True})  
            self.augmentation_data.add_to_augmentation(adv_images, adv_lables)
            return (adv_images, adv_lables)
        return {
            'acc_op': model['acc_op'], 
            'pred_op': model['pred_op'], 
            'train_fn': training_aug_step, 
            'summ_op': model['summ_op']
        }
    
    def only_random_augmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('with_random_augmentation'):
            return self.internal_only_augmentation(learning_rate, features, labels, is_training, 0)

    def only_augmentation(self, learning_rate, features, labels, is_training):
        with tf.name_scope('with_adversarial_augmentation'):
            return self.internal_only_augmentation(learning_rate, features, labels, is_training, self.num_adversarial_train)

    def internal_only_augmentation(self, learning_rate, features, labels, is_training, num_adversarial_train):
        #scalar that controls augmentation 
        aug_beta = tf.placeholder(dtype=tf.float32, shape=[], name="aug_beta")
        aug_model = self.model_fn(aug_batch, aug_batch_la, self.num_classes, is_training, True)

        loss_with_cont = aug_beta * aug_model['loss_op']
        train_op = optimizer.minimize(loss_with_cont,
                                    global_step=tf.train.get_global_step())

        gen_aug = set_up_augmentation_generation(learning_rate, features, labels, is_training)

        def training_aug_step(iteration, session, batch_size_training, batch_size_augmentation):
            (training, aug_training) = self.augmentation_data.next_batch(batch_size_training, batch_size_augmentation)
            (train_images, train_labels) = training
            (aug_img, aug_labels) = aug_training

            (adv_images, adv_lables) = gen_aug(num_adversarial_train)

            aug_img = np.concatenate((aug_img, adv_images), axis=0)
            aug_labels = np.concatenate((aug_labels, adv_lables), axis=0)

            t = session.run([train_op], feed_dict={
                    features.name: train_images,
                    labels.name: train_labels,
                    aug_batch.name: aug_img,
                    aug_batch_la.name: aug_labels,
                    aug_beta.name: 1,
                    is_training.name: True})  
            self.augmentation_data.add_to_augmentation(adv_images, adv_lables)
            return (adv_images, adv_lables)
        return {
            'acc_op': model['acc_op'], 
            'pred_op': model['pred_op'], 
            'train_fn': training_aug_step, 
            'summ_op': model['summ_op']
        }
    
    def set_up_augmentation_generation(self, learning_rate, features, labels, is_training):
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
        
        def gen_augmentation(num_adversarial_train):
            # generates the congingency
            randomInput = nprandom.random((self.num_adversarial, self.num_input))
            gen_aug_labels = self.gen_aug_labels(self.num_adversarial)
            session.run(gen_images.assign(randomInput))

            for iteration in range(num_adversarial_train):
                #TODO does this work?
                a = session.run(adv_op, feed_dict={
                        gen_labels.name: gen_aug_labels,
                        features.name: train_images,
                        #we are not training the weights!
                        is_training.name: False})
            adv_images = session.run(gen_images)
            return (adv_images, gen_aug_labels)

        return gen_augmentation

    def eval(self, session, model):
        (eval_valid_im, eval_valid_la) = self.augmentation_data.get_valid_training_data()
        #a = session.run(model['acc_op'], feed_dict={images: eval_rel_im, labels: eval_rel_la, is_training.name: False})
        #print("Final Accuracy on all relabeled classes", iteration, ":", a)
        a = session.run(model['acc_op'], feed_dict={images: eval_valid_im, labels: eval_valid_la, is_training.name: False})
        print("Final Accuracy on only valid classes:", a)
        (unex_im, unex_la) = self.augmentation_data.unexpected_data()
        a = session.run(model['acc_op'], feed_dict={images: unex_im, labels: unex_la, is_training.name: False})
        print("Final Accuracy on unexpected data:", a)
        (rand_im, rand_la) = self.augmentation_data.generate_random(eval_valid_la.shape[0])
        a = session.run(model['acc_op'], feed_dict={images: rand_im, labels: rand_la, is_training.name: False})
        print("Final Accuracy on random data:", a)

        #TODO is this right?
        # from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        (data, labels) = self.augmentation_data.relabel_roc()
        pred = session.run(model['pred_op'], 
                feed_dict={images: data, is_training.name: False})
        pred_roc = 1 - pred[:,0]
        fpr[0], tpr[0], _ = roc_curve(labels, pred_roc)
        roc_auc[0] = auc(fpr[0], tpr[0])

        return (fpr, tpr, roc_auc, pred, model)

    @staticmethod
    def pairwise_l2_norm(x, y, num_input):
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

    @staticmethod
    def calc_max_dist(dataset, num_input):
        length = dataset.shape[0]
        subLen= 50
        subsets = [tf.convert_to_tensor(dataset[n*subLen:(n+1)*subLen], dtype=tf.float32) for n in range(0, math.ceil(length/subLen))]
        products = list(itertools.product(subsets, repeat=2))
        print("numer of products", len(list(products)))
        def max_l2(pairing):
            (x, y) = pairing
            max_val = tf.reduce_max(tf.reshape(Augmentation.pairwise_l2_norm(x, y, num_input), shape=[-1, 1]))
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

    @staticmethod
    def gen_labels_default_class(num_classes, default_class)
        """
        returns a function that generates n one-hot encoded labels with a value of 1 for 
        default class (element [0, num_classes))
        """
        def do_gen(length):
            data = np.zeros((length, num_classes))
            data[np.arange(length), default_class] = 1
            return data
        return do_gen

    @staticmethod
    def gen_labels_no_class(num_classes)
        """
        returns a function that generates n one-hot encoded labels with a value of 1 for 
        default class (element [0, num_classes))
        """
        def do_gen(length):
            data = np.full((length, num_classes), float(1)/num_classes, tf.float32)
            return data
        return do_gen
    