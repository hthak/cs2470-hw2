from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int]=[1,1,1,1], padding = "VALID", use_bias = True, trainable=True, *args, **kwargs):
        """
        :param filter_shape: list of [filter_height, filter_width, in_channels, out_channels]
        :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
        :param padding: either "SAME" or "VALID", capitalization matters
        """
        super().__init__()

        self.strides = strides
        self.padding = padding

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable = trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable)
        self.use_bias = use_bias
        if use_bias: self.bias = get_var("conv_bias", [filter_shape[-1]], trainable)
        else: self.bias = None

    def get_weights(self):
        if self.bias is not None: return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None):
        self.filters = filters
        if bias is not None: self.bias = bias

    def call(self, inputs):
        """
        :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
        """

        #define some useful variables
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape

        # fill out the rest!
        assert input_in_channels == filter_in_channels
        assert self.strides == [1, 1, 1, 1]

        #padding stuff
        if self.padding == "SAME":
            pad_y = filter_height - 1
            pad_x = filter_width - 1
        else:
            pad_y = 0
            pad_x = 0

        #subtracting for odd padding
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left

        #batch, height, width, channels
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]])

        #according to slides
        out_height = in_height + pad_y - filter_height + 1
        out_width = in_width + pad_x - filter_width + 1

        #filter multiplication
        output_rows = []
        for y in range(out_height):
            output_cols = []
            for x in range(out_width):
                #get the valid filter region to multiply against
                region = padded_inputs[
                    :,
                    y:y+filter_height,
                    x:x+filter_width,
                    :
                ]
                #broadcasting
                products = (tf.expand_dims(region, axis=-1)*(tf.expand_dims(tf.cast(self.filters, tf.float32), axis=0)))
                values = tf.reduce_sum(products, axis=[1, 2, 3])

                if self.bias is not None:
                    values = values + tf.cast(self.bias, tf.float32)

                output_cols.append(values)

            output_rows.append(tf.stack(output_cols, axis=1))

        output = tf.stack(output_rows, axis=1)
        return tf.convert_to_tensor(output, dtype=tf.float32)