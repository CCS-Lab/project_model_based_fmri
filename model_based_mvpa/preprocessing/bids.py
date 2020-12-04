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
from ..utils import config # configuration for default names used in the package
import nibabel as nib

import logging


bids.config.set_option("extension_initial_dot", True)
logging.basicConfig(level=logging.INFO)



def bids_preprocess(root=None,  # path info
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
    This function is for preprocessing fMRI image data organized in BIDS layout.

    What you need as input:
    1) fMRI image data organized in BIDS layout 
        ** you need to provide fMRI data which has gone through the conventional primary preprocessing pipeline (recommended link: https://fmriprep.org/en/stable/), and should be in "BIDSroot/derivatives/fmriprep" 
    2) mask images (nii or nii.gz format) either downloaded from Neurosynth or created by the user

    Goals of this function:
    1) Remove motion artifacts and drift - Done by wrapping functions in nilearn package
    2) Reduce dimensionality by masking and pooling 
        - an important process as the high dimensionality is an obstacle for fitting regression model, in terms of both computing time and convergence.

        2-1) Masking - Related argument (mask_path),(threshold)
            Probabilistic maps are integrated to make the maps as ROIs (a binary voxel-wise mask). 
            It is needed to threshold the map so that you create a mask that only includes voxels with a z-score of a specific value greater than the threshold.
            After thresholding, the surviving voxels are binarized - in other words, set to 1.
            The mask extracts the data from voxels within that region (We can extract the voxels from the mask / Only voxel whose mask value is 1 will be  extracted)
            -> This process reduces the total number of voxels that will be included in the analysis.
            If the masking information is not provided, all the voxels in MNI 152 space will be included in the data.

        2-2) Pooling - Related argument (zoom), (interpolation_func)
            The number of voxels is further diminished by zooming (or resacling) fMRI images to a coarse-grained resolution. 
            You can give a tuple indicating a zooming window size in x,y,z directions. e.g. (2,2,2)
            Voxels in a cube with the zooming window size will be converted to one representative value reducing resolution and the total number of voxels.
            You can also indicate the method to extract representative value with numpy function. e.g. np.mean means using the average value.

    3) re-organize data for fitting MVPA model - the preprocessed image will be saved subject-wise.
    """
    
    """
    Arguments:
        root (str or Path) : the root directory of BIDS layout
        layout (nibabel.BIDSLayout): BIDSLayout by bids package. if not provided, it will be obtained from root path.
        save_path (str or Path): a path for the directory to save output (X, voxel_mask). if not provided, "BIDS root/derivatives/data" will be set as default path      
        mask_path (str or Path): a path for the directory containing mask files (nii or nii.gz). encourage get files from Neurosynth
        threshold (float): threshold for binarizing mask images
        zoom ((float,float,float)): zoom window, indicating a scaling factor for each dimension in x,y,z. the dimension will be reduced by the factor of corresponding axis.
                                    e.g. (2,2,2) will make the dimension half in all directions, so 2x2x2=8 voxels will be 1 voxel.
        smoothing_fwhm (int): the amount of spatial smoothing. if None, image will not be smoothed.
        interpolation_func (numpy.func): a method to calculate a representative value in the zooming window. e.g. numpy.mean, numpy.max
                                         e.g. zoom=(2,2,2) and interpolation_func=np.mean will convert 2x2x2 cube into a single value of its mean.
        standardize (boolean): if true, conduct standard normalization within each image of a single run. 
        motion_confounds (list[str]): list of motion confound names in confounds tsv file. 
        ncore (int): the number of core for the tparallel computing 
        nthread (int): the number of thread for the parallel computing
    Return:
        X (numpy.array): subject-wise & run-wise BOLD time series data. shape : subject # x run # x timepoint # x voxel #
        voxel_mask (nibabel.Nifti1Image): a nifti image for voxel-wise binary mask (ROI mask)
        masker (nilearn.NiftiMasker): the masker object. fitted and used for correcting motion confounds, and masking.
        layout (nibabel.BIDSLayout): the loaded layout. 
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

    if mask_path is None:
        mask_path = Path(
            layout.derivatives["fMRIPrep"].root) / config.DEFAULT_MASK_DIR
      
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
        # get a list of nii file paths of fMRI images spread in BIDS layout
        nii_layout = layout.derivatives["fMRIPrep"].get(
            subject=subject, return_type="file", suffix="bold",
            extension="nii.gz") 
        # get a list of tsv file paths of regressors spread in BIDS layout
        # e.g. tsv file with motion confound parameters. 
        reg_layout = layout.derivatives["fMRIPrep"].get(
            subject=subject, return_type="file", suffix="regressors",
            extension="tsv")

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
        if not sp.exists():
            sp.mkdir()
    else:
        sp = Path(save_path)
    
    if save:
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
    task_size = len(params_chunks)

    for i, params_chunk in enumerate(params_chunks):
        with ProcessPoolExecutor(max_workers=chunk_size) as executor:
            future_result = {
                executor.submit(
                    image_preprocess_mt, param, n_run): param for param in params_chunk
            }

            for future in as_completed(future_result):
                data, subject = future.result()
                np.save(
                    sp / f"{config.DEFAULT_FEATURE_PREFIX}_{subject}.npy", data)
                X.append(data)

            pbar.set_description(
                f"image preprocessing - fMRI data.. {i+1} / {task_size} done..".ljust(50))

    X = np.array(X)
    pbar.update(1)

    ################################################################################
    # elapsed time check

    pbar.set_description("bids preprocessing done!".ljust(50))
    pbar.update(1)

    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")
    logging.info(f"result\nmasking data shape: {voxel_mask.shape}\n")

    return X, voxel_mask, layout
