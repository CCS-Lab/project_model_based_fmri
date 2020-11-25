    #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong, Cheoljun cho
@contact: mybirth0407@gmail.com
          cjfwndnsl@gmail.com
@last modification: 2020.11.13
"""

import os
import numpy as np
from pathlib import Path

import bids
from bids import BIDSLayout, BIDSValidator

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
from .fMRI import *

import logging


DEFAULT_SAVE_DIR = "mvpa"
DEFAULT_MASK_DIR = "masks"

bids.config.set_option("extension_initial_dot", True)
logging.basicConfig(level=logging.INFO)


def bids_preprocess(root,
                    mask_path=None,
                    save_path=None,
                    save=True,
                    zoom=(2, 2, 2),
                    smoothing_fwhm=6,
                    interpolation_func=np.mean,
                    motion_confounds=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
                    p_value=0.05,
                    task_name="task-zero",
                    ncore=0,
                    nthread=0,
                    standardize=True):

    pbar = tqdm(total=6)
    s = time.time()
################################################################################
# load bids layout

    pbar.set_description("loading bids dataset..".ljust(50))
    layout = BIDSLayout(root, derivatives=True)
    
    subjects = layout.get_subjects()
    n_subject = len(subjects)
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())
    pbar.update(1)
################################################################################
# make masked data

    pbar.set_description("making custom masked data..".ljust(50))
    root = Path(root)
    
    if mask_path is None:
        mask_path = Path(layout.derivatives["fMRIPrep"].root) / DEFAULT_MASK_DIR
    
    masked_data, masker, m_true = custom_masking(
        mask_path, p_value, zoom,
        smoothing_fwhm, interpolation_func, standardize
    )
        
    pbar.update(1)
################################################################################
# setting parameter

    pbar.set_description("image preprocessing - parameter setting..".ljust(50))

    params = []
    for subject in subjects:
        nii_layout = layout.derivatives["fMRIPrep"].get(
            subject=subject, return_type="file", suffix="bold",
            extension="nii.gz"
        )
        reg_layout = layout.derivatives["fMRIPrep"].get(
            subject=subject, return_type="file", suffix="regressors",
            extension="tsv"
        )

        param = [nii_layout, reg_layout, motion_confounds,
                 masker, masked_data, subject]
        params.append(param)

    assert len(params) == n_subject, "length of params list and the number of subjects are not validate"
    pbar.update(1)
################################################################################
# create path for data

    pbar.set_description("image preprocessing - making path..".ljust(50))
    if save_path is None:
        sp = Path(layout.derivatives["fMRIPrep"].root) / DEFAULT_SAVE_DIR
    else:
        sp = Path(save_path)
        
    nib.save(masked_data, sp / "masked_data.nii.gz")
    pbar.update(1)
################################################################################
# image preprocessing using mutli-processing and threading

    pbar.set_description("image preprocessing - fMRI data..".ljust(50))
    X = []

    ## Todo ##
    chunk_size = int(np.log10(n_subject * n_run)) if n_session == 0 \
                    else int(np.log10(n_subject * n_session * n_run))
    chunk_size += 1

    params_chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]

    for i, params_chunk in enumerate(params_chunks):
        with ProcessPoolExecutor(max_workers=chunk_size) as executor:
            future_result = {
                executor.submit(
                    image_preprocess_mt, param, n_run): param for param in params_chunk
            }

            for future in as_completed(future_result):
                data, subject = future.result()
                np.save(sp / f"X_{subject}.npy", data)
                X.append(data)
            pbar.set_description(
                f"image preprocessing - fMRI data {i+1} / {len(params_chunks)}..".ljust(50))

    X = np.array(X)
    pbar.update(1)
################################################################################
# elapsed time check

    pbar.set_description("bids preprocessing done!".ljust(50))
    pbar.update(1)

    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")
    logging.info(f"result\nmasking data shape: {masked_data.shape}\n"
               + f"number of voxels: {m_true.shape}")

    return X, masked_data, layout
