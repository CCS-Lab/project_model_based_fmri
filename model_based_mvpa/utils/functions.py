#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong, Cheol Jun Cho
@contact: mybirth0407@gmail.com
          cjfwndnsl@gmail.com
@last modification: 2020.11.16
"""

import numpy as np
from pathlib import Path
from scipy import stats
import nibabel as nib
from functools import reduce

"""
This file is utility functions sets, which not included in data, models, preprocessing.
The functions that are not related to each module but are frequently used are implemented here.
"""


def array2pindex(array, p_value=0.05, flatten=False):
    """
    Alive only index above a given p value
    """

    """
    Arguments:
        array:
        p_value:
        flatten:

    Return:
        ret: binary array preprocessed by p-value.
    """

    confidence = 1 - p_value
    flattened_array = array.flatten()
    
    # Calculate confidence intervals using p-value.
    # end is upper parts of the confidence interval.
    n = len(flattened_array)
    m = np.mean(flattened_array)
    std_err = stats.sem(flattened_array)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    end = m + h
    
    ret = (flattened_array >= end) if flatten is True else (array >= end)
    return ret


def prepare_dataset(root=None, X_path=None, y_path=None, time_mask_path=None):
    """
    Get dataset for fitting model and time-masked brain map.
    """

    """
    Arguments:
        root: data path, if None, must be specified X, y, time_mask_path.
              default path is imported from layout.
        X_path: optional, X data path, if None, default is bids/derivates/data.
        y_path: optional, y data path, if None, default is bids/derivates/data.
        time_mask_path: optional, time mask data path, if None, default is bids/derivates/data.

    Return:
        TODO: explain to time points
        X: X, which is adjusted dimension and masked time points for training
        y: y, which is adjusted dimension and masked time points for training
    """

    # if root is given and path for any of X, y is not given, then use default path.
    if root is not None:
        root = Path(root)
        X_path = root / 'mvpa'
        y_path = root / 'mvpa'
        time_mask_path = root / 'mvpa'
    else:
        assert X_path is not None, "If root is None, you must be indicate data path (X, Y, time mask)"
        assert y_path is not None, "If root is None, you must be indicate data path (X, Y, time mask)"

        X_path = Path(X_path)
        y_path = Path(y_path)

    # aggregate X fragmented by subject to one matrix
    X_list = list(X_path.glob('X_*.npy'))
    X_list.sort(key=lambda v: int(str(v).split('_')[-1].split('.')[0]))
    X = np.concatenate([np.load(data_path) for data_path in X_list], 0)
    X = X.reshape(-1, X.shape[-1])

    y = np.load(y_path / 'y.npy', allow_pickle=True)
    y = np.concatenate(y, 0)
    X = X.reshape(-1, X.shape[-1])
    y = y.flatten()

    if time_mask_path is not None:
        # use data only at timepoints indicated in time_mask file.
        time_mask_path = Path(time_mask_path)
        time_mask = np.load(time_mask_path / 'time_mask.npy', allow_pickle=True)
        time_mask = np.concatenate(time_mask, 0)
        time_mask = time_mask.flatten()

    X = X[time_mask > 0]
    y = y[time_mask > 0]
    
    return X, y
