#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.02
"""

import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import block_reduce

from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_brain_mask

import nibabel as nib
from ..utils import func

import logging


logging.basicConfig(level=logging.INFO)


def custom_masking(mask_path, p_value, zoom,
                   smoothing_fwhm, interpolation_func, standardize,
                   flatten=False):

    if mask_path is None:
        #assert (mask_path is None)
        mask_files = []
    elif type(mask_path) is str:
        mask_files = [mask_path]
    else:
        if type(mask_path) is not type(Path()):
            mask_path = Path(mask_path)
        mask_files = [file for file in mask_path.glob('*.nii.gz')]

    #image_sample = nib.load(mask_files[0])
    image_sample = load_mni152_brain_mask()
    m = func.array2pindex(image_sample.get_fdata(), p_value, flatten)

    for i in range(len(mask_files)):
        m |= func.array2pindex(nib.load(mask_files[i]).get_fdata(), p_value, flatten)
    
    if zoom != (1, 1, 1):
        m = block_reduce(m, zoom, interpolation_func)
    m = 1 * (m > 0)

    m_true = np.array([i for i, v in enumerate(m.flatten()) if v != 0])
    masked_data = nib.Nifti1Image(m, affine=image_sample.affine)
    masker = NiftiMasker(mask_img=masked_data,
                         standardize=standardize,
                         smoothing_fwhm=smoothing_fwhm)

    return masked_data, masker, m_true


def image_preprocess(params):
    image_filepath, confounds, motion_confounds, masker, masked_data, num = params
    
    if confounds is not None:
        confounds = pd.read_table(confounds, sep='\t')
        confounds = confounds[motion_confounds]
        confounds = confounds.to_numpy()

    fmri_masked = resample_to_img(image_filepath, masked_data)
    fmri_masked = masker.fit_transform(fmri_masked, confounds=confounds)
    
    return fmri_masked
