from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf
import numpy as np
from numpy import random as nprandom

from sklearn.metrics import roc_curve, auc

class Contingency:
    def __init__(self, learning_rate, num_steps, batch_size, num_adversarial, num_input, num_classes, model_fn, train, test):
        # Training Parameters
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_adversarial = num_adversarial

        # Network Parameters
        self.num_input = num_input
        self.num_classes = num_classes

        #TODO max size?
        self.contingency_imges = np.empty(shape=(0, num_input))
        self.contingency_labels = np.empty(shape=(0))

        self.model_fn = model_fn

        #train/test datasets
        self.train = train
        self.test = test

    def reset_contingency(self):
        self.contingency_imges = np.empty(shape=(0, self.num_input))


    def withoutContingency(self, features, labels, is_training):
        (loss_op, pred, acc) = self.model_fn(features, labels, self.num_classes, is_training, False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op,
                                    global_step=tf.train.get_global_step())

        def trainWithout(iteration, session, training, cont_training):
            (train_images, train_labels) = training
            a, t = session.run([acc, train_op], feed_dict={
                    features.name: train_images,
                    labels.name: train_labels,
                    is_training.name: True})  
            empty_imges = np.empty(shape=(0, self.num_input))
            empty_labels = np.empty(shape=(0))
            empty = (empty_imges, empty_labels)
            return (a, empty)
        
        return (acc, pred, trainWithout)

    def withRandomContingency(self, features, labels, is_training):
        return self.withContingency(features, labels, is_training, 0)

    def withContingency(self, features, labels, is_training, num_adversarial_train):
        (orig_loss_op, pred, acc) = self.model_fn(features, labels, is_training, False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        cont_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.num_input], name="cont_batch")
        cont_batch_la = tf.placeholder(dtype=tf.float32, shape=[None], name="cont_batch_labels")

        #scalar that controls contingency 
        cont_beta = tf.placeholder(dtype=tf.float32, shape=[], name="cont_beta")

        (cont_loss, cont_pred, cont_acc) = self.model_fn(cont_batch, cont_batch_la, is_training, True)
        loss_with_cont = orig_loss_op + cont_beta * cont_loss
        train_op = optimizer.minimize(loss_with_cont,
                                    global_step=tf.train.get_global_step())

        #generating the contingengy
        gen_images = tf.get_variable(dtype=tf.float32, shape=[self.num_adversarial, self.num_input], name="gen_images")
        gen_labels = tf.placeholder(dtype=tf.float32, shape=[self.num_adversarial], name="gen_labels")

        (gen_loss_op, gen_pred, gen_acc) = self.model_fn(gen_images, gen_labels, is_training, True)

        #TODO replace
        #there is a general mysterium surrounding this function. What does it do exactly? I have not get round 
        #testing/investigating it yet.
        diff = tf.contrib.gan.eval.frechet_classifier_distance(
            features, 
            gen_images, 
            #I don't really understand what this is used for...
            lambda imges : conv_net(imges, num_classes, False, is_training=is_training, should_reuse=True))

        adversial_fitness = gen_loss_op #- diff
        #TODO maybe switch to Adam and reset it for each (classifier-)training step? 
        optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
                        #we are not training the weights!
                        is_training.name: False})

            adv_images = session.run(gen_images)
            cont_img = np.concatenate((cont_img, adv_images), axis=0)
            #TODO redo this, we can't switch contingency labels right now
            cont_labels = np.zeros(cont_img.shape[0])

            a, t = session.run([acc, train_op], feed_dict={
                    features.name: train_images,
                    labels.name: train_labels,
                    cont_batch.name: cont_img,
                    cont_batch_la.name: cont_labels,
                    cont_beta.name: 1,
                    is_training.name: True})  

            return (a, (adv_images, zerolabels))
        return (acc, pred, trainingContStep)

    def run(self, run_fn): 
        images = tf.placeholder(tf.float32, shape=[None, self.num_input], name="images")
        labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        is_training = tf.placeholder(tf.bool, name="is_training")
        with tf.Session() as session:
            (acc_eval, pred_eval, train_fn) = run_fn(images, labels, is_training)
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())
            session.run(tf.local_variables_initializer())
            # Build the Estimator

            for iteration in range(self.num_steps):
                (training, cont_training) = self.next_batch(self.batch_size, (self.batch_size - self.num_adversarial), self.train)
                (a, cont) = train_fn(iteration, session, training, cont_training)
                (cont_imgs, cont_la) = cont
                self.contingency_imges = np.concatenate((self.contingency_imges, cont_imgs), axis=0)
                self.contingency_labels = np.concatenate((self.contingency_labels, cont_la), axis=0)
                # a = session.run(accEval, feed_dict={images.name: train_im, labels.name: train_la})
                if iteration % 50 == 0:
                    print("Generated contingency in iteration ", iteration, ":", self.contingency_imges.shape[0])
                    print("Training Accuracy in iteration ", iteration, ":", a)

            (eval_rel_im, eval_rel_la) = self.relabel(self.test)
            (eval_valid_im, eval_valid_la) = self.only_valid(self.test.images, self.test.labels)
            #a = session.run(acc_eval, feed_dict={images: eval_rel_im, labels: eval_rel_la, is_training.name: False})
            #print("Final Accuracy on all relabeled classes", iteration, ":", a)
            a = session.run(acc_eval, feed_dict={images: eval_valid_im, labels: eval_valid_la, is_training.name: False})
            print("Final Accuracy on only valid classes", iteration, ":", a)
            (unex_im, unex_la) = self.unexpected_data(self.test)
            a = session.run(acc_eval, feed_dict={images: unex_im, labels: unex_la, is_training.name: False})
            print("Final Accuracy on unexpected data", iteration, ":", a)

            #TODO is this right?
            # from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
            # Compute ROC curve and ROC area for each class
            # BEWARE: the zero class here is unexpected input, not the zero class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            (data, labels) = self.relabel_roc(self.test)
            pred = session.run(pred_eval, 
                    feed_dict={images: data, is_training.name: False})
            pred_roc = 1 - pred[:,0]
            fpr[0], tpr[0], _ = roc_curve(labels, pred_roc)
            roc_auc[0] = auc(fpr[0], tpr[0])

            return (fpr, tpr, roc_auc, pred)

    def only_valid(self, images, labels):
        indices = np.where(labels < self.num_classes )
        return (images[indices], labels[indices])

    def relabel(self, dataset):
        indices = np.where(dataset.labels >= self.num_classes )
        relabel = np.copy(dataset.labels)
        relabel[indices] = 0
        return (dataset.images, relabel)

    def relabel_roc(self, dataset):
        #50% unexpected data
        unexp_indices = np.where(dataset.labels >= self.num_classes )
        #we only want unexpected data
        not_null_valid = np.where(np.logical_and(np.greater(dataset.labels,0), np.less(dataset.labels, self.num_classes)))
        length = min(unexp_indices[0].shape[0], not_null_valid[0].shape[0])

        unexp_indices = unexp_indices[:length]
        unexp_imges = dataset.images[unexp_indices]
        unexp_labels = np.zeros(unexp_indices[0].shape[0])

        not_null_valid = not_null_valid[:length]
        valid_imges = dataset.images[not_null_valid]
        valid_labels = np.ones(not_null_valid[0].shape[0])

        roc_imges = np.concatenate((unexp_imges,valid_imges), axis=0)
        roc_labels = np.concatenate((unexp_labels,valid_labels), axis=0)
        return (roc_imges, roc_labels)

    def relabel_pred_roc(self, pred):
        #work around 'builtin_function_or_method' object does not support item assignment
        indices = np.where(pred > 0)
        rel_pred = np.zeros(pred.shape,dtype=int)
        rel_pred[indices] = 1
        return rel_pred

    def next_batch(self, batch_size_training, batch_size_contingency, train):
        training = self.only_valid(*train.next_batch(self.batch_size))
        indices = np.arange(0 , self.contingency_imges.shape[0])
        np.random.shuffle(indices)
        indices = indices[:batch_size_contingency]
        contingency_trainig_imgs = self.contingency_imges[indices]
        contingency_trainig_labels = self.contingency_labels[indices]
        contingency = (contingency_trainig_imgs, contingency_trainig_labels)
        return (training, contingency)

    def test_data_for_label(self, label, test):
        indices = np.where(test.labels == label)
        return (test.images[indices], test.labels[indices])

    def unexpected_data(self, dataset):
        indices = np.where(dataset.labels >= self.num_classes)
        imges = dataset.images[indices]
        #TODO more dynamic defaults
        zeros = np.zeros(indices[0].shape[0])
        return (imges, zeros)