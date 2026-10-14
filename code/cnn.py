from __future__ import absolute_import
from multiprocessing import pool
from matplotlib import pyplot as plt
from torch import manual_seed
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)

        self.hidden_layer_size = 320

        self.epsilon = 1e-3  # this is used for batch normalization only!

        # Fill the rest of this out!
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        #fully connected layers
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer_size, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.num_classes)

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, num_classes)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        if is_testing:
            manual_conv = ManualConv2d(filter_shape=list(self.conv1.kernal.shape), strides=[1,1,1,1], padding="SAME", use_bias=self.conv1.use_bias,trainable=False)
            manual_conv.set_weights(self.conv1.kernal, self.conv1.bias if self.conv1.use_bias else None)

            x = manual_conv(inputs)

            x = tf.nn.relu(x)
        else:
            x = self.conv1(inputs)

        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)

        return logits

