#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.16
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


def prepare_data(root=None, X=None, y=None):
    # input is root
    if root is not None:
        _root = Path(root) / 'derivatives/fmriprep/data/'
        X_list = sorted(list(_root.glob('X*.pkl')))
        y_list = sorted(list(_root.glob('y*.pkl')))

        X = []
        for partfile in X_list:
            X.append(np.load(partfile))
        X = np.concatenate(X)

        y = []
        for partfile in y_list:
            y.append(np.load(partfile))
        y = np.concatenate(y)

    else:
        # input is X, y
        if type(y) is list:
            y = np.array(y)

    X_reshaped = X.reshape(-1, X.shape[-1])
    y_reshaped = y.reshape(-1, y.shape[-1])

    assert (X_reshaped.shape != y_reshaped.shape, "X, y data shapes are difficult..")

    return X_reshaped, y_reshaped