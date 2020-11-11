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
import json

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
                    mask_path=None,
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
    
    n_subject = len(layout.get_subjects())
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())
    pbar.update(1)
    
    pbar.set_description('make custom masking..'.center(40))
    root = Path(root)
    
    if mask_path is None:
        mask_path = Path(layout.derivatives['fMRIPrep'].root) / 'mask'
    else:
        mask_path = Path(mask_path)
        
    masked_data, masker, m_true = custom_masking(
        mask_path, p_value, zoom,
        smoothing_fwhm, interpolation_func, standardize
    )
    pbar.update(1)

    pbar.set_description('image preprocessing using %d cores..'.center(40) % ncore)
    
    
    nii_layout = {subject_id:layout.derivatives['fMRIPrep'].get(subject=subject_id, 
                                                                 return_type='file', 
                                                                 suffix='bold', 
                                                                 extension='nii.gz') for subject_id in layout.get_subjects()}
    
    reg_layout = {subject_id:layout.derivatives['fMRIPrep'].get(subject=subject_id, 
                                                                return_type='file', 
                                                                suffix='regressors', 
                                                                extension='tsv') for subject_id in layout.get_subjects()}
    
    params = {subject_id:[[z[0], z[1], motion_confounds, masker, masked_data, i] 
                          for i, z in enumerate(zip(nii_layout[subject_id], reg_layout[subject_id]))] 
                          for subject_id in layout.get_subjects()}
    
    if save_path is None:
        sp = root / 'derivatives'/ 'data'
    else:
        sp = Path(save_path)
            
    if not sp.exists():
        sp.mkdir()
            
            
    nib.save(masked_data, sp / 'masked_data.nii.gz')        
    meta_info ={}
    
    for subject_id in layout.get_subjects():
        with ProcessPoolExecutor() as executor:
            X = np.array(list(executor.map(image_preprocess, params[subject_id])))
            
            # run number can be different among subjects. 
            
            '''
            if n_session != 0:
                X = X.reshape(n_session, n_run, -1, m_true.shape[0])
            else:
                X = X.reshape(n_run, -1, m_true.shape[0])
            '''
        meta_info[f'sub_{subject_id}'] = 
        pbar.set_description('file saving..'.center(40))
        np.save(sp / f'X_{subject_id}.npy', X)
    
    
    
    with open(sp /'meta_info.json','w') as f:
        json.dump(meta_info,f)
            
    pbar.update(1)
    pbar.set_description('bids preprocessing done!'.center(40))

    time.sleep(1)
    logging.info(f'result\nmasking data shape: {masked_data.shape}\nnumber of voxels: {m_true.shape}')

    if time_check:
        e = time.time()
        logging.info(f'time elapsed: {e-s} seconds')
        
    return X, masked_data, layout
