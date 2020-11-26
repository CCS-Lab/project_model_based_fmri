#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong, Cheoljun cho
@contact: mybirth0407@gmail.com
          cjfwndnsl@gmail.com
@last modification: 2020.11.13
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import bids
from bids import BIDSLayout
from tqdm import tqdm

from .fMRI import custom_masking, image_preprocess, image_preprocess_mt
from ..utils import config

import logging


bids.config.set_option("extension_initial_dot", True)
logging.basicConfig(level=logging.INFO)


def bids_preprocess(root,  # path info
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
                    motion_confounds=["trans_x", "trans_y",
                                      "trans_z", "rot_x", "rot_y", "rot_z"],
                    # multithreading option
                    ncore=2,
                    nthread=2,
                    # other specification
                    save=True):
    """
    This function is implemented to preprocess fMRI image data in BIDS layout. 
    The major goals of this function are 1) remove motion artifacts and drift,2) reduce dimensionality by masking and pooling, and 3) re-organize data for fitting MVPA model.
    1) is done by wrapping functions in nilearn package.
    2) is important as the high dimensionality is an obstacle for fitting regression model, in terms of both computing time and optimization.
    To do so, here, mask images (downloaded from Neurosynth by user) are integrated to make a binary voxel-wise mask. So only voxel with mask value of 1 will survive, reducing the total number of voxels. 
    For 3), the preprocessed image will be saved subject-wisely.
    Also, to reduce total computing time, parallel computing is utilized in this function.

    Arguments:
        root (str or Path) : root directory of BIDS layout
        layout (BIDSLayout): BIDSLayout by bids package. if not provided, it will be obtained using root info.
        save_path (str or Path): path for saving output. if not provided, BIDS root/derivatives/data will be set as default path      
        mask_path (str or Path): path for mask file with nii.gz format. encourage get files from Neurosynth
        threshold (float): threshold for binarize masks
        zoom ((float,float,float)): zoom window to reduce the spatial dimension. the dimension will be reduced by the factor of corresponding axis.
                                    e.g. (2,2,2) will make the dimension half in all directions.
        smoothing_fwhm (int): the amount of gaussian smoothing. if None, image will not be smoothed.
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

    ###########################################################################
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

    ###########################################################################
    # make voxel mask

    pbar.set_description("making custom voxel mask..".ljust(50))
    root = Path(root)

    if mask_path is None:
    # TODO: where does `custom_masking` come from?
        mask_path = Path(layout.derivatives["fMRIPrep"].root) / config.DEFAULT_MASK_DIR
      
    voxel_mask, masker = custom_masking(
        mask_path, threshold, zoom,
        smoothing_fwhm, interpolation_func, standardize
    )

    pbar.update(1)

    ###########################################################################
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

    assert len(params) == n_subject, (
        "The length of params list and number of subjects are not validated."
    )
    pbar.update(1)

    ###########################################################################
    # create path for data

    pbar.set_description("image preprocessing - making path..".ljust(50))
    if save_path is None:
        sp = Path(layout.derivatives["fMRIPrep"].root) / config.DEFAULT_SAVE_DIR
    else:
        sp = Path(save_path)
        
    # TODO: where does `nib` come from?
    nib.save(voxel_mask, sp / config.DEFAULT_VOXEL_MASK_FILENAME)
    pbar.update(1)

    ###########################################################################
    # image preprocessing using mutli-processing and threading

    pbar.set_description("image preprocessing - fMRI data..".ljust(50))
    X = []

    ## Todo ##
    chunk_size = 4 if nthread > 4 else ncore

    params_chunks = [params[i:i + chunk_size]
                     for i in range(0, len(params), chunk_size)]

    for i, params_chunk in enumerate(params_chunks):
        with ProcessPoolExecutor(max_workers=chunk_size) as executor:
            # TODO: where does `image_preprocess_mt` come from?
            future_result = {
                executor.submit(
                    image_preprocess_mt, param, n_run): param for param in params_chunk
            }

            for future in as_completed(future_result):
                data, subject = future.result()
                np.save(sp / f"{config.DEFAULT_FEATURE_PREFIX}_{subject}.npy", data)
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

    # TODO: where does `m_true` come from?
    logging.info(f"result\nmasking data shape: {voxel_mask.shape}\n"
                 + f"number of voxels: {m_true.shape}")

    return X, voxel_mask, layout
