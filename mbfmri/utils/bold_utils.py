#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

## https://github.com/nilearn/nilearn

import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import block_reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_brain_mask, fetch_icbm152_brain_gm_mask
import importlib
import nibabel as nib
from mbfmri.utils import config
from mbfmri.utils.atlas import get_roi_mask
from scipy import ndimage

def _zoom_affine(affine, zoom):
    affine = affine.copy()
    affine[0,:3] *= zoom[0]
    affine[1,:3] *= zoom[1]
    affine[2,:3] *= zoom[2]
    return affine

def _zoom_img(img_array, original_affine, zoom, binarize=False, threshold=.5):
    
    new_img_array = block_reduce(img_array, zoom, np.mean)
    if binarize:
        new_img_array = (new_img_array>threshold) * 1.0
    precise_zoom = np.array(img_array.shape[:3])/np.array(new_img_array.shape[:3])
    
    return nib.Nifti1Image(new_img_array, 
                           affine=_zoom_affine(original_affine, precise_zoom))
    

def _integrate_mask(mask_files,template,threshold,smoothing_fwhm,verbose=0):
    m = np.zeros(template.shape).astype(bool)
    for i in range(len(mask_files)):
        # binarize and stack
        mask_loaded = resample_to_img(str(mask_files[i]), template)
        assert (mask_loaded.affine==template.affine).all()
        mask = mask_loaded.get_fdata()
        affine = mask_loaded.affine
        
        # retrieved from nilearn.image._smooth_array
        if smoothing_fwhm is not None  and smoothing_fwhm > 0:
            smoothing_fwhm = np.asarray([smoothing_fwhm]).ravel()
            smoothing_fwhm = np.asarray([0. if elem is None else elem for elem in smoothing_fwhm])
            affine = affine[:3, :3]  # Keep only the scale part.
            fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
            vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
            sigma = smoothing_fwhm / (fwhm_over_sigma_ratio * vox_size)
            for n, s in enumerate(sigma):
                if s > 0.0:
                    ndimage.gaussian_filter1d(mask, s, output=mask, axis=n)
                
        binarized_mask = mask >= threshold
        m |= binarized_mask
        if verbose > 0:
            survived = int(binarized_mask.sum())
            total = np.prod(binarized_mask.shape)
            print('      '+str(mask_files[i].stem)+f': {survived}/{total}')
    return m

def _build_mask(mask_path, threshold, zoom,smoothing_fwhm,include_default_mask=True,
                atlas=None,rois=[],gm_only=False,verbose=1):
    
    
    
    # list up mask image file
    
    default_mask = ()
    if not include_default_mask:
        default_mask = nib.Nifti1Image(np.ones(default_mask.shape),
                                   affine=default_mask.affine)
        
    if gm_only:
        gm = fetch_icbm152_brain_gm_mask()
        gm = resample_to_img(gm, default_mask)
        default_mask = gm
        
    if atlas is not None:
        roi_mask = get_roi_mask(atlas, rois)
        roi_mask = resample_to_img(roi_mask, default_mask )
        default_mask = roi_mask
        
    if mask_path is None:
        include_mask_files = []
        exclude_mask_files = []
        m = (default_mask.get_fdata().round() >0)
    else:
        if type(mask_path) is not type(Path()):
            mask_path = Path(mask_path)
        include_mask_path = mask_path/config.DEFAULT_MASK_INCLUDE_DIR
        exclude_mask_path = mask_path/config.DEFAULT_MASK_EXCLUDE_DIR
        if include_mask_path.exists():
            include_mask_files = [file for file in include_mask_path.glob("*.nii*")]
        else:
            include_mask_files = []
        if exclude_mask_path.exists():
            exclude_mask_files = [file for file in exclude_mask_path.glob("*.nii*")]
        else:
            exclude_mask_files = []
            
        if not exclude_mask_path.exists() and \
            not include_mask_path.exists():
            include_mask_files = [file for file in mask_path.glob("*.nii*")]
            exclude_mask_files = []
    
        # integrate binary mask data
        report_dict = {}
        print('INFO: start loading & intergrating masks to include')
        include_mask = _integrate_mask(include_mask_files,default_mask,threshold,smoothing_fwhm, verbose)
        if verbose > 0:
            survived = int(include_mask.sum())
            total = np.prod(include_mask.shape)
            print('INFO: integrated mask to include'+f': {survived}/{total}')
        print('INFO: start loading & intergrating masks to exclude')
        exclude_mask = _integrate_mask(exclude_mask_files,default_mask,threshold,smoothing_fwhm, verbose)
        if verbose > 0:
            survived = int(exclude_mask.sum())
            total = np.prod(exclude_mask.shape)
            print('INFO: integrated mask to exclude'+f': {survived}/{total}')

        m = (default_mask.get_fdata().round() > 0) & include_mask  & (~exclude_mask)

    if verbose > 0:
        survived = int(m.sum())
        total = np.prod(m.shape)
        print('      final mask'+f': {survived}/{total}')
    # reduce dimension by averaging zoom window
    affine = default_mask.affine.copy()
    
    if zoom != (1, 1, 1):
        voxel_mask = _zoom_img(m, affine, zoom, binarize=True)
        if verbose > 0:
            m = voxel_mask.get_fdata()
            survived = int(m.sum())
            total = np.prod(m.shape)
            print(f'      zoomed {str(zoom)}'+f': {survived}/{total}')
    else:
        voxel_mask = nib.Nifti1Image(m.astype(float), affine=affine)
    
    
    return voxel_mask

def _custom_masking(voxel_mask, t_r,
                   smoothing_fwhm, standardize,
                   high_pass, detrend):
        
    # masking is done by NiftiMasker provided by nilearn package
    masker = NiftiMasker(mask_img=voxel_mask,
                         t_r=t_r,
                         standardize=standardize,
                         smoothing_fwhm=smoothing_fwhm,
                         high_pass=high_pass,
                         detrend=detrend)
    masker = masker.fit()
    return masker


def _image_preprocess(params):

    image_path, confounds_path, save_path,\
    confound_names, masker = params

    preprocessed_images = []
    if confounds_path is not None:
        confounds = pd.read_table(confounds_path, sep="\t")
        if isinstance(confound_names,list) and len(confound_names) > 0:
            confounds = confounds[confound_names]
            confounds = confounds.to_numpy()
            confounds[np.isnan(confounds)] = 0
            std = confounds.std(0)
            mean = confounds.mean(0)
            confounds = (confounds-mean)/std
            confounds[np.isnan(confounds)] = 0
        elif confound_names == 'all':
            confounds = confounds
            confounds = confounds.to_numpy()
            confounds[np.isnan(confounds)] = 0
            std = confounds.std(0)
            mean = confounds.mean(0)
            confounds = (confounds-mean)/std
            confounds[np.isnan(confounds)] = 0
        else:
            confounds = None
    else:
        confounds = None

    fmri_masked = masker.transform_single_imgs(nib.load(str(image_path)), confounds=confounds)
    np.save(save_path, fmri_masked)
    
    return 1