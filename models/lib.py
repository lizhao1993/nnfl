#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-16
Brief:  The library module of neural network
"""

import numpy as np
import logging
import sys
import operator
sys.path.append("../utils/")
import metrics

logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)"\
        "s[function:%(funcName)s] %(message)s"
)

np.random.seed(1)

class Layer(object):
    """
    Layer class 
    """
    def __init__(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of softmax layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        use_bias: boolean
            Whether to use bias vector on this layer 
        tfloat: str
            Type of float used on the weights
        """

        self.n_i = n_i
        self.n_o = n_o
        self.use_bias = use_bias
        self.norm_func = norm_func
        self.tfloat = tfloat

        # Init weights
        self.w = np.random.uniform(
            low=-np.sqrt(1. / self.n_i), 
            high=np.sqrt(1. / self.n_i), 
            size=(self.n_o, self.n_i)
        ).astype(dtype=self.tfloat, copy=False)

        # Init bias on output layer
        if self.use_bias:
            self.b = np.zeros(shape=self.n_o, dtype=self.tfloat)


class SoftmaxLayer(Layer):
    """
    Softmax layer class(Numerically-stable)
    """
    def __init__(self, n_i, n_o, use_bias=True, tfloat='float64'):
        """
        Initialize parameters of softmax layer
        n_i: int.
            The number of units of the previous layer
        n_o: int.
            The number of units of current layer
        use_bias: boolean
            Whether to use bias vector on this layer 
        tfloat: str
            Type of float used on the weights
        """
        Layer.__init__(n_i, n_o, use_bias, tfloat)


    def forward(self, x):
        """
        Compute forward pass, numerically-stable softmax function    
        x: numpy.ndarray
            x is the input data with the shape [num_instances, num_inputs]
        """

        if x.shape[1] != self.n_i:
            logging.error("input data shape:%s, not match input unit:%s" \
                            % (x.shape, self.n_i))
            raise Exception

        if self.use_bias:
            net_input = x.dot(self.w.T) + self.b
        else:
            net_input = x.dot(self.w.T)
        stable_input = (net_input 
                    - np.max(net_input, axis=1).reshape(net_input.shape[0], 1))
        stable_input_exp = np.exp(stable_input)
        res = (stable_input_exp 
                / np.sum(stable_input_exp, axis=1
                    ).reshape(stable_input_exp.shape[0], 1))
        return res

    def backprop(self):
        """
        Back propagation
        """

            



