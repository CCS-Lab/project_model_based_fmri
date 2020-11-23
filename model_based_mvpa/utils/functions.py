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
from model_based_mvpa.preprocessing.bids import DEFAULT_SAVE_PATH_X, DEFAULT_SAVE_PATH_MASKED_DATA
from model_based_mvpa.preprocessing.events import DEFAULT_SAVE_PATH_y

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


def prepare_data(root=None,
                 prep_path_X=None, 
                 prep_path_y=None,
                 ):
    
    # get X, y for fitting model and making map
    
    
    # if root is given and path for any of X, y is not given, then use default path.
    if root is None:
        assert(prep_path_X is not None)
        assert(prep_path_y is not None)
    else:
        root = Path(root)
        
    if prep_path_X is None:
        prep_path_X = root / DEFAULT_SAVE_PATH_X
        
    if prep_path_y is None:
        prep_path_y = root / DEFAULT_SAVE_PATH_y
    
    # aggregate X fragmented by subject to one matrix
    data_path_list = list(prep_path_X.glob('X_*.npy'))
    data_path_list.sort(key=lambda v: int(str(v).split('_')[-1].split('.')[0]))
    X = np.concatenate([np.load(data_path) for data_path in data_path_list],0)
    X = X.reshape(-1,X.shape[-1])

    y = np.load(prep_path_y / 'y.npy' , allow_pickle=True)
    y = np.concatenate(y,0)
    X = X.reshape(-1,X.shape[-1])
    y = y.flatten()

    # use data only at timepoints indicated in time_mask file.
    time_mask = np.load(prep_path_y / 'time_mask.npy', allow_pickle=True)
    time_mask = np.concatenate(time_mask,0)
    time_mask = time_mask.flatten()

    X = X[time_mask>0]
    y = y[time_mask>0]
    
    return X, y


def get_mask_image(root=None,
                 masked_data_path=None, 
                 ):
    
    if masked_data_path is None:
        masked_data_path = root / DEFAULT_SAVE_PATH_MASKED_DATA

    return nib.load(masked_data_path / 'masked_data.nii.gz')