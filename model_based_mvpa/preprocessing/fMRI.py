#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.13
"""

import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import block_reduce

from concurrent.futures import ThreadPoolExecutor, as_completed

from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_brain_mask
from scipy import stats
from scipy.ndimage import gaussian_filter

import nibabel as nib
from ..utils import functions as F

import logging

logging.basicConfig(level=logging.INFO)


def custom_masking(mask_path, threshold, zoom,
                   smoothing_fwhm, interpolation_func, standardize):
    """
    Make custom ROI mask to reduce the number of features.
    """

    """
    Arguments:
        mask_path (str or Path): a path for the directory containing mask files (nii or nii.gz). encourage get files from Neurosynth
        threshold (float): threshold for binarizing mask images
        zoom ((float,float,float)): zoom window, indicating a scaling factor for each dimension in x,y,z. the dimension will be reduced by the factor of corresponding axis.
                                    e.g. (2,2,2) will make the dimension half in all directions, so 2x2x2=8 voxels will be 1 voxel.
        smoothing_fwhm (int):the amount of spatial smoothing. if None, image will not be smoothed.
        interpolation_func (numpy.func): a method to calculate a representative value in the zooming window. e.g. numpy.mean, numpy.max
                                         e.g. zoom=(2,2,2) and interpolation_func=np.mean will convert 2x2x2 cube into a single value of its mean.
        standardize (boolean): if true, conduct standard normalization within each image of a single run. 
    Return:
        voxel_mask (nibabel.Nifti1Image): a nifti image for voxel-wise binary mask (ROI mask)
        masker (nilearn.NiftiMasker): a masker object. will be used for correcting motion confounds, and masking.
    """
    
    # list up mask image file
    if mask_path is None:
        mask_files = []
    else:
        if type(mask_path) is not type(Path()):
            mask_path = Path(mask_path)
        mask_files = [file for file in mask_path.glob("*.nii.gz")]

    mni_mask = load_mni152_brain_mask()
    
    # integrate binary mask data
    if len(mask_files) > 0 :
        # binarize
        m = abs(nib.load(mask_files[0]).get_fdata(), 1) >= threshold
        for i in range(len(mask_files)-1):
            # binarize and stack
            m |= abs(nib.load(mask_files[0]).get_fdata(), 1) >= threshold
    else:
        # if not provided, use min_152 mask instead.
        m = mni_mask.get_fdata()
    
    # reduce dimension by averaging zoom window
    if zoom != (1, 1, 1):
        m = block_reduce(m, zoom, interpolation_func)
    m = 1 * (m > 0)

    voxel_mask = nib.Nifti1Image(m, affine=mni_mask.affine)
    
    # masking is done by NiftiMasker provided by nilearn package
    masker = NiftiMasker(mask_img=voxel_mask,
                         standardize=standardize,
                         smoothing_fwhm=smoothing_fwhm)

    return voxel_mask, masker


def image_preprocess(params):
    """
    Make image that is motion corrected and ROI masked.
    This function wrapped up functions from nilearn package.
    """

    """
    Arguments:
        params: params must have below contents
            image_path (str or Path): path of fMRI nii file 
            confounds_path (str or Path): path of corresponding motion confounds tsv file
            motion_confounds (list[str]): list of name indicating motion confound names in confounds tsv file
            masker (nilearn.NiftiMasker): masker object. will be used for correcting motion confounds, and masking.
            voxel_mask (nibabel.Nifti1Image): nifti image for voxel-wise binary mask
            subject_id (str): subject ID. used to track the owner of the file in multiprocessing

    Return:
        fmri_masked (numpy.array): preprocessed image with shape run # x time point # x voxel #
        subject_id (str): subject ID. used to track the owner of the file in multiprocessing
    """

    image_path, confounds_path,\
    motion_confounds, masker,\
    voxel_mask, subject_id = params

    preprocessed_images = []
    if confounds_path is not None:
        confounds = pd.read_table(confounds_path, sep="\t")
        confounds = confounds[motion_confounds]
        confounds = confounds.to_numpy()
    else:
        confounds = None

    # different from resample_img.
    # resample_img: need to target affine and shape.
    # resample_to_img: need to target image including affine.
    # ref.: https://nilearn.github.io/modules/generated/nilearn.image.resample_img.html
    #       https://nilearn.github.io/modules/generated/nilearn.image.resample_to_img.html
    fmri_masked = resample_to_img(image_path, voxel_mask)
    fmri_masked = masker.fit_transform(fmri_masked, confounds=confounds)

    return fmri_masked, subject_id


def image_preprocess_mt(params, nthread):
    """
    Call image_preprocess function using multithreading.
    """

    """
    Arguments:
        params: params must have below contents
            image_path (str or Path): path of fMRI nii file 
            confounds_path (str or Path): path of corresponding motion confounds tsv file
            motion_confounds (list[str]): list of name indicating motion confound names in confounds tsv file
            masker (nilearn.NiftiMasker): masker object. will be used for correcting motion confounds, and masking.
            voxel_mask (nibabel.Nifti1Image): nifti image for voxel-wise binary mask
            subject_id (str): subject ID. used to track the owner of the file in multiprocessing

    Return:
        fmri_masked (numpy.array): preprocessed image with shape run # x time point # x voxel #
        subject_id (str): subject ID. used to track the owner of the file in multiprocessing
    """
    image_paths, confounds_paths,\
    motion_confounds, masker,\
    voxel_mask, subject_id = params

    image_params = []
    for i in range(len(params[0])):
        image_params.append(
            [image_paths[i], confounds_paths[i], motion_confounds,
             masker, voxel_mask, subject_id])

    preprocessed_images = []
    n_worker = 4 if nthread > 4 else nthread
    
    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        future_result = {
            executor.submit(
                image_preprocess, image_param): image_param \
                    for image_param in image_params
        }

        for future in as_completed(future_result):
            data, subject_id = future.result()
            preprocessed_images.append(data)

    preprocessed_images = np.array(preprocessed_images)

    return preprocessed_images, subject_id
