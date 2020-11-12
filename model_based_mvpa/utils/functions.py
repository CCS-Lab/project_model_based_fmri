#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.02
"""

import numpy as np
from pathlib import Path
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


def prepare_data(root, n=None):
    if type(root) is str:
        X_path = root
    if type(y) is str:
        y = np.load(y)

    if n is None or n == 0:
        X_reshaped = X.reshape(-1, X.shape[-1])
        y_reshaped = y.reshape(-1, y.shape[-1])
    else:
        X_reshaped = X.reshape(n, -1)
        y_reshaped = y.reshape(n, -1)

    assert (X_reshaped.shape != y_reshaped.shape)

    return X_reshaped, y_reshaped