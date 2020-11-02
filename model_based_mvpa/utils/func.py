#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.02
"""

import numpy as np
from scipy import stats
from functools import reduce


def array2pindex(array, p_value=0.05, flatten=False):
    confidence = 1 - p_value
    flattened_array = array.flatten()
    
    n = len(flattened_array)
    m = np.mean(flattened_array)
    std_err = stats.sem(flattened_array)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    end = m + h
    
    ret = (flattened_array >= end) if flatten is True else (array >= end)
    return ret


def get_flatten_shape(array):
    return reduce(lambda x, y: x * y, array)


def prepare_data(X, y):
    if type(X) == str:
        X = np.load(X)
    if type(y) == str:
        y = np.load(y)
    
    X_reshaped = X.reshape(get_flatten_shape(X), -1)
    y_reshaped = y.reshape(get_flatten_shape(y), -1)

    return X_reshaped, y_reshaped