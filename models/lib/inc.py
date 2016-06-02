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


def get_col_from_jagged_array(pos, jagged_array):
    """
    Get column from jagged array 
    pos: int
        Column position at array. If pos == -1, the last column will return.
        Units will be not added if pos is invalid at some rows
    jagged_array: 3d array-like
        eg., [[1dndarray, 1dndarray], [...], [...]]
    Return
    -----
    column: numpy.ndarray
    """

    # Count the shape at first
    axis0 = 0
    axis1 = 0
    for row in jagged_array:
        if pos in range(0, len(row)) or pos == -1:
            axis0 += 1
    if axis0 != 0 and len(jagged_array[0]) != 0:
        axis1 = len(jagged_array[0][0])

    column = np.zeros((axis0, axis1))
    col_i = 0
    for i in range(0, len(jagged_array)):
        if pos in range(0, len(jagged_array[i])) or pos == -1:
            column[col_i] = jagged_array[i][pos]
            col_i += 1

    return column


def print_jagged_array(jagged_array):
    """
    Print jagged_array
    jagged_array: 3d array-like
        eg., [[1dndarray, 1dndarray], [...], [...]]
    """

    for row in jagged_array:
        str_row = ""
        for col in row:
            str_row += str(col) + " "
        print(str_row)


def jagged_array_test():
    x_num = 4
    x = []
    n_i = 3
    for i in range(0, x_num):
        # Random column
        col = np.random.randint(low=1, high=5)
        x_row = []
        for j in range(0, col):
            x_row.append(np.random.randint(low=0, high=5, size=(n_i, )))
        x.append(np.asarray(x_row))
    print("x:")
    print_jagged_array(x)
    print(get_col_from_jagged_array(-1, x))


if __name__ == "__main__":
    jagged_array_test()
