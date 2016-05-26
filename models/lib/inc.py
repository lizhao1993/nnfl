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
