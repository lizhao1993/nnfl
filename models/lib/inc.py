#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-19
Brief:  Common headers
"""

import numpy as np
import logging

np.random.seed(1)

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s]%(filename)s:%(lineno)"
    "s[function:%(funcName)s] %(message)s"
)

# Common function

def sigmoid(x):
    """
    The sigmoid function. This is numerically-stable version
    x: float
    """
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (exp_x + 1)


def sigmoid_array(x):
    """
    Numerically-stable sigmoid function
    x: ndarray (float)
    """
    vfunc = np.vectorize(sigmoid)
    return vfunc(x)
