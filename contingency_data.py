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

class ContingencyData:

    def __init__(self, train_data, train_labels, test_data, test_labels, validation_data
                , validation_labels, num_classes, num_input):
        """
        Parameters
        ----------
        num_classes : int
            the the number of classes trained on
        """
        self.train_data = train_data
        self.train_labels = train_labels

        self.test_data = test_data
        self.test_labels = test_labels

        train_indices_valid = np.where(train_labels < num_classes)
        self.train_data_valid = train_data[train_indices_valid]
        self.train_labels_valid = train_labels[train_indices_valid]
        train_random_indexes = np.arange(0 , self.train_data_valid.shape[0])
        np.random.shuffle(train_random_indexes)
        self.train_data_valid = self.train_data_valid[train_random_indexes]
        self.train_labels_valid = self.train_labels_valid[train_random_indexes]
        self.current_index_train = 0
        self.epoch_count = 0

        test_indices_valid = np.where(test_labels < num_classes)
        self.test_data_valid = test_data[test_indices_valid]
        self.test_labels_valid = test_labels[test_indices_valid]
        self.current_index_test = 0
        self.epoch_count_test = 0

        test_indices_invalid = np.where(test_labels >= num_classes)
        self.test_data_invalid = test_data[test_indices_invalid]
        self.test_labels_invalid = np.zeros(test_indices_invalid[0].shape[0])

        validation_indices_valid = np.where(validation_labels < num_classes)
        self.validation_data_valid = validation_data[validation_indices_valid]
        self.validation_labels_valid = validation_labels[validation_indices_valid]

        validation_indices_invalid = np.where(validation_labels >= num_classes)
        self.validation_data_invalid = validation_data[validation_indices_invalid]
        self.validation_labels_invalid = np.zeros(validation_indices_invalid[0].shape[0])

        self.num_classes = num_classes
        self.num_input = num_input

        #TODO max size?
        self.contingency_data = np.empty(shape=(0, num_input))
        self.contingency_labels = np.empty(shape=(0))
        self.current_index_contingency = 0

    def next_batch(self, batch_size_training, batch_size_contingency):
        training_step = self.internal_next_batch(self.train_data_valid, self.train_labels_valid
                                        , self.current_index_train, batch_size_training)
        (d, l, new_train_index, data_train, did_reshuffle) = training_step
        self.current_index_train = new_train_index
        if did_reshuffle:
            self.epoch_count += 1
            self.train_data_valid = d
            self.train_labels_valid = l
        contingency_step = self.internal_next_batch(self.contingency_data, self.contingency_labels
                                        , self.current_index_contingency, batch_size_contingency)
        (c_d, c_l, new_cont_index, data_contingency, c_did_reshuffle) = contingency_step
        self.current_index_contingency = new_cont_index
        if c_did_reshuffle:
            self.contingency_data = c_d
            self.contingency_labels = c_l

        return (data_train, data_contingency)

    def next_test_batch(self, batch_size_test):
        test_step = self.internal_next_batch(self.test_data_valid, self.test_labels_valid
                                        , self.current_index_test, batch_size_test)
        (d, l, new_test_index, data_test, did_reshuffle) = test_step
        self.current_index_test = new_test_index
        if did_reshuffle:
            self.epoch_count += 1
            self.test_data_valid = d
            self.test_labels_valid = l
        return data_test


    def internal_next_batch(self, data, labels, current_index, batch_size):
        end = current_index + batch_size
        if end > data.shape[0]:
            (d, l, i, data_batch) = self.reshuffle(data, labels, current_index, batch_size)
            return (d, l, i, data_batch, True)
        else:
            res_data = data[current_index: end]
            res_lbls = labels[current_index: end]
            return (data, labels, end, (res_data, res_lbls), False)

    def reshuffle(self, data, labels, current_index, batch_size):
        data_length = data.shape[0]
        remaining_data = data[current_index:data_length]
        remaining_labels = labels[current_index:data_length]

        train_random_indexes = np.arange(0 , data_length)
        np.random.shuffle(train_random_indexes)
        reshu_data = data[train_random_indexes]
        reshu_labels = labels[train_random_indexes]

        updated_index = (batch_size) - remaining_data.shape[0]
        additional_data = reshu_data[:updated_index]
        additional_labels = reshu_labels[:updated_index]

        train_data = np.concatenate((remaining_data,additional_data), axis=0)
        train_labels = np.concatenate((remaining_labels,additional_labels), axis=0)
        return (reshu_data, reshu_labels, updated_index, (train_data, train_labels))

    def full_test_data(self):
        """
        returns the normal, valid test data and relabeled data outside of the normal trained data-manifold
        """
        indices = np.where(self.test_labels >= self.num_classes )
        relabel = np.copy(self.test_labels)
        relabel[indices] = 0
        return (self.test_data, relabel)

    def relabel_roc(self):
        """
        returns 50% relabeled data outside of the normal trained data-manifold and 50% normal data
        """
        #50% unexpected data
        unexp_indices = np.where(self.test_labels >= self.num_classes )
        #we only want unexpected data
        not_null_valid = np.where(
                            np.logical_and(
                                np.greater_equal(self.test_labels,0), 
                                np.less(self.test_labels, self.num_classes)
                            ))
        length = min(unexp_indices[0].shape[0], not_null_valid[0].shape[0])

        unexp_indices = unexp_indices[0][:length]
        unexp_imges = self.test_data[unexp_indices]
        unexp_labels = np.zeros(unexp_indices.shape[0])

        not_null_valid = not_null_valid[0][:length]
        valid_imges = self.test_data[not_null_valid]
        valid_labels = self.test_labels[not_null_valid]
        valid_labels[valid_labels > 0] = 1

        roc_imges = np.concatenate((unexp_imges,valid_imges), axis=0)
        roc_labels = np.concatenate((unexp_labels,valid_labels), axis=0)
        return (roc_imges, roc_labels)

    def relabel_pred_roc(self, pred):
        #work around 'builtin_function_or_method' object does not support item assignment
        indices = np.where(pred > 0)
        rel_pred = np.zeros(pred.shape,dtype=int)
        rel_pred[indices] = 1
        return rel_pred

    def test_data_for_label(self, label):
        """
        returns all the test-data for a single label
        """
        indices = np.where(self.test_labels == label)
        return (self.test_data[indices], self.test_labels[indices])

    def unexpected_data(self):
        """
        returns all the relabeled data outside of the normal trained data-manifold
        """
        return (self.test_data_invalid, self.test_labels_invalid)

    def get_original_training_data(self):
        return (self.train_data, self.train_labels)

    def get_valid_training_data(self):
        return (self.train_data_valid, self.train_labels_valid)

    def get_num_classes(self):
        return self.num_classes

    def add_to_contingency(self, cont_data, cont_lables):
        self.contingency_data = np.concatenate((self.contingency_data, cont_data), axis=0)
        self.contingency_labels = np.concatenate((self.contingency_labels, cont_lables), axis=0)

    def reset_contingency(self):
        self.contingency_data = np.empty(shape=(0, self.num_input))
        self.contingency_labels = np.empty(shape=(0))

    def generate_random(self, num_random):
        randomImages = nprandom.random((num_random, self.num_input))
        zerolabels = np.zeros(num_random)
        return (randomImages, zerolabels)