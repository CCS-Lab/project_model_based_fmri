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


def prepare_dataset(root=None, X_path=None, y_path=None, time_masks_path=None):
    """
    Get dataset for fitting model and time-masked brain map.
    """

    """
    Arguments:
        root: 
        X_path: 
        y_path: 

    Return:
        X: 
        y: 
    """

    # if root is given and path for any of X, y is not given, then use default path.
    if root is not None:
        root = Path(root)
        X_path = root / 'mvpa'
    else:
        assert X_path is None, "If root is None, you must be indicate data path (X, Y, time mask)"
        assert y_path is None, "If root is None, you must be indicate data path (X, Y, time mask)"
        assert time_masks_path is None, "If root is None, you must be indicate data path (X, Y, time mask)"
        
    # aggregate X fragmented by subject to one matrix
    X_list = list(X_path.glob('X_*.npy'))
    X_list.sort(key=lambda v: int(str(v).split('_')[-1].split('.')[0]))
    X = np.concatenate([np.load(data_path) for data_path in X_list],0)
    X = X.reshape(-1, X.shape[-1])

    y = np.load(y_path / 'y.npy' , allow_pickle=True)
    y = np.concatenate(y, 0)
    X = X.reshape(-1, X.shape[-1])
    y = y.flatten()

    # use data only at timepoints indicated in time_mask file.
    time_mask = np.load(y_path / 'time_masks.npy', allow_pickle=True)
    time_mask = np.concatenate(time_mask,0)
    time_mask = time_mask.flatten()

    X = X[time_mask>0]
    y = y[time_mask>0]
    
    return X, y


def get_mask_image(root=None, masked_data_path=None):
    if masked_data_path is None:
        masked_data_path = root / DEFAULT_SAVE_PATH_MASKED_DATA

    return nib.load(masked_data_path / 'masked_data.nii.gz')