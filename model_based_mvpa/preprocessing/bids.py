#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.02
"""

import os
import numpy as np
from pathlib import Path

import bids
from bids import BIDSLayout, BIDSValidator

from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm
from .fMRI import *

import logging


bids.config.set_option('extension_initial_dot', True)
logging.basicConfig(level=logging.INFO)


def bids_preprocess(root,
                    save_path=None,
                    save=True,
                    single_file=False,
                    zoom=(1, 1, 1),
                    smoothing_fwhm=6,
                    interpolation_func=np.mean,
                    motion_confounds=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
                    p_value=0.05,
                    task_name='task-zero',
                    standardize=True,
                    ncore=os.cpu_count(),
                    time_check=True):

    pbar = tqdm(total=4)
    s = time.time()

    pbar.set_description('loading bids dataset..'.center(40))
    layout = BIDSLayout(root, derivatives=True)
    nii_layout = layout.derivatives['fMRIPrep'].get(return_type='file', suffix='bold', extension='nii.gz')
    reg_layout = layout.derivatives['fMRIPrep'].get(return_type='file', suffix='regressors', extension='tsv')
    
    n_subject = len(layout.get_subjects())
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())
    pbar.update(1)
    
    pbar.set_description('make custom masking..'.center(40))
    root = Path(root)
    mask_path = Path(layout.derivatives['fMRIPrep'].root) / 'mask'

    masked_data, masker, m_true = custom_masking(
        mask_path, p_value, zoom,
        smoothing_fwhm, interpolation_func, standardize
    )
    pbar.update(1)

    pbar.set_description('image preprocessing using %d cores..'.center(40) % ncore)
    params = [[z[0], z[1], motion_confounds, masker, masked_data, i]
            for i, z in enumerate(zip(nii_layout, reg_layout))]

    with ProcessPoolExecutor() as executor:
        X = np.array(list(executor.map(image_preprocess, params)))
        
        if n_session != 0:
            X = X.reshape(n_subject, n_session, n_run, -1, m_true.shape[0])
        else:
            X = X.reshape(n_subject, n_run, -1, m_true.shape[0])
    pbar.update(1)

    pbar.set_description('file saving..'.center(40))
    if save:
        if save_path is None:
            sp = Path(layout.derivatives['fMRIPrep'].root) / 'data'
        else:
            sp = Path(save_path)
            
        if not sp.exists():
            sp.mkdir()
        
        if single_file:
            np.save(sp / 'X.npy', X)
        else:
            for i in range(X.shape[0]):
                np.save(sp / f'X_{i+1}.npy', X[i])
        nib.save(masked_data, sp / 'masked_data.nii.gz')
    pbar.update(1)
    pbar.set_description('bids preprocessing done!'.center(40))

    logging.info(f'result\nmasking data shape: {maksed_data.shape}\nnumber of voxels: {m_true.shape}')

    if time_check:
        e = time.time()
        logging.info(f'time elapsed: {e-s} seconds')
        
    return X, masked_data