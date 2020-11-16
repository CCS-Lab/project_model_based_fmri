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

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    
    if len(mask_files) > 0 :
        m = func.array2pindex(nib.load(mask_files[0]).get_fdata(), p_value, flatten)
        for i in range(len(mask_files)-1):
            m |= func.array2pindex(nib.load(mask_files[i]).get_fdata(), p_value, flatten)
    else:
        m = func.array2pindex(image_sample.get_fdata(), p_value, flatten)
    
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
    image_path, confounds_path,\
    motion_confounds, masker,\
    masked_data, subject = params

    preprocessed_images = []
    if confounds_path is not None:
        confounds = pd.read_table(confounds_path, sep='\t')
        confounds = confounds[motion_confounds]
        confounds = confounds.to_numpy()
    else:
        confounds = None

    fmri_masked = resample_to_img(image_path, masked_data)
    fmri_masked = masker.fit_transform(fmri_masked, confounds=confounds)

    return fmri_masked, subject
    

def image_preprocess_mt(params, n_thread):
    image_paths, confounds_paths,\
    motion_confounds, masker,\
    masked_data, subject = params
    # print(image_paths, confounds_paths, motion_confounds,
    #       masker, masked_data, subject)

    image_params = []
    for i in range(len(params[0])):
        image_params.append(
            [image_paths[i], confounds_paths[i], motion_confounds,
             masker, masked_data, subject])

    preprocessed_images = []
    n_worker = n_thread if n_thread < 5 else n_thread // 2
    n_worker += 1
    
    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        future_result = {
            executor.submit(image_preprocess, image_param): image_param for image_param in image_params
        }

        for future in as_completed(future_result):
            data, subject = future.result()
            preprocessed_images.append(data)

    preprocessed_images = np.array(preprocessed_images)

    return preprocessed_images, subject
