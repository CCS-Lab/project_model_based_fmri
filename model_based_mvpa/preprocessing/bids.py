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
VOXEL_MASK_FILENAME = "voxel_mask.nii.gz"
PREP_IMG_FILEPREFIX = 'X'

bids.config.set_option("extension_initial_dot", True)
logging.basicConfig(level=logging.INFO)


def bids_preprocess(# path info
                    root,
                    layout=None,
                    save_path=None,
                    # ROI masking specification
                    mask_path=None,
                    threshold=2.58,
                    # preprocessing specification
                    zoom=(2, 2, 2),
                    smoothing_fwhm=6,
                    interpolation_func=np.mean,
                    standardize=True,
                    motion_confounds=["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"],
                    # multithreading option
                    ncore=0, 
                    nthread=0,
                    # other specification
                    save=True):
    
    """
    Make custom ROI mask file to reduce the number of features.
    """

    """
    Arguments:
        root (str or Path) : root directory of BIDS layout
        layout (BIDSLayout): BIDSLayout by bids package. if not provided, it will be obtained using root info.
        save_path (str or Path): path for saving output. if not provided, BIDS root/derivatives/data will be set as default path      
        mask_path (str or Path): path for mask file with nii.gz format. encourage get files from Neurosynth
        threshold (float): threshold for binarize masks
        zoom ((float,float,float)): zoom window to reduce the spatial dimension. the dimension will be reduced by the factor of corresponding axis.
                                    e.g. (2,2,2) will make the dimension half in all directions.
        smoothing_fwhm (int): the amount of gaussian smoothing  
        interpolation_func (numpy.func): to calculate representative value in the zooming window. e.g. numpy.mean, numpy.max
                                         e.g. zoom=(2,2,2) and interpolation_func=np.mean will convert 2x2x2 cube to a single value of its mean.
        standardize (boolean): if true, conduct gaussian normalization 
        motion_confounds (list[str]): list of name indicating motion confound names in confounds tsv file
        ncore (int): number of core 
        nthread (int): number of thread
    Return:
        voxel_mask (Nifti1Image): nifti image for voxel-wise binary mask
        masker (NiftiMasker): masker object. fitted and used for correcting motion confounds, and masking.
        layout (BIDSLayout): loaded layout. 
    """

    pbar = tqdm(total=6)
    s = time.time()
################################################################################
# load bids layout

    if layout is None:
        pbar.set_description("loading bids dataset..".ljust(50))
        layout = BIDSLayout(root, derivatives=True)
    else:
        pbar.set_description("loading layout..".ljust(50))
    
    subjects = layout.get_subjects()
    n_subject = len(subjects)
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())
    pbar.update(1)
################################################################################
# make voxel mask

    pbar.set_description("making custom voxel mask..".ljust(50))
    root = Path(root)
    
    if mask_path is None:
        mask_path = Path(layout.derivatives["fMRIPrep"].root) / DEFAULT_MASK_DIR
    
    voxel_mask, masker = custom_masking(
        mask_path, threshold, zoom,
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
                 masker, voxel_mask, subject]
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
        
    nib.save(voxel_mask, sp / VOXEL_MASK_FILENAME)
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
                np.save(sp / f"{PREP_IMG_FILEPREFIX}_{subject}.npy", data)
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
    logging.info(f"result\nmasking data shape: {voxel_mask.shape}\n"
               + f"number of voxels: {m_true.shape}")

    return X, voxel_mask, layout