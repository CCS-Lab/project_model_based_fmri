#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_1samp, zscore


def reconstruct(array, mask):
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    blackboard = np.zeros(mask.shape)
    blackboard[mask.nonzero()] = array
    
    return blackboard
    
    
def get_map(coefs, voxel_mask, experiment_name, standardize=False, save_path=".", smoothing_fwhm=6):
    """
    make nii image file from coefficients of model using masking info.
    """
    """
    Arguments:
        coefs (list or numpy.array): extracted coefs by fitting model.
            shape: N x voxel #
        voxel_mask (nibabel.nifti1.Nifti1Image):
            a binary nii image of masking info.
            should be same shape with the original fMRI image data
        task_name (str): if provided, the saved nii file will be named after this
        map_type (str): the type of making map. ("z" or "t")
            "z": default type. z_map by mean z_score among N mapped images.
            "t": t_map by voxel-wise one sample t-test among N mapped images.
        save_path (str or pathlib.Path): path to save file. if None, then will be saved in "results" directory fo current working directory
            the result_map nii image file will be saved in "save_path/{task_name}_{map_type}_map.nii"
        sigma (int): the sigma value for spatial gaussian smoothing for each converted map.  
               if 0, there will be no spatial smoothing of extracted coefficients.
               higher the value, more smoother the final map will be 

    Return:
        result_map (nibabel.nifti1.Nifti1Image): a nii image of brain activation map.
    """

    ###########################################################################
    # parameter check

    assert (isinstance(coefs, list)
        or isinstance(coefs, np.ndarray))
    assert isinstance(voxel_mask, nib.nifti1.Nifti1Image)
    assert isinstance(experiment_name, str)
    assert (isinstance(save_path, str)
        or isinstance(save_path, Path))
    ###########################################################################
    # make result map

    activation_maps = []
    mask = voxel_mask.get_fdata()
    
    for coef in coefs:
        if standardize:
            coef = zscore(coef,axis=None)
        # converting flattened coefs to brain image.
        if len(coef.shape) != 3:
            activation_map = reconstruct(coef.ravel(), mask)
        else:
            activation_map = coef
            
        # retrieved from nilearn.image._smooth_array
        if smoothing_fwhm is not None  and smoothing_fwhm > 0:
            smoothing_fwhm = np.asarray([smoothing_fwhm]).ravel()
            smoothing_fwhm = np.asarray([0. if elem is None else elem for elem in smoothing_fwhm])
            affine = voxel_mask.affine[:3, :3]  # Keep only the scale part.
            fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))  # FWHM to sigma.
            vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
            sigma = smoothing_fwhm / (fwhm_over_sigma_ratio * vox_size)
            for n, s in enumerate(sigma):
                if s > 0.0:
                    gaussian_filter1d(activation_map, s, output=activation_map, axis=n)
                
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    # voxel-wise one sample ttest
    m = ttest_1samp(activation_maps, 0).statistic

    m[np.isnan(m)] = 0
    m *= mask
    result_map = nib.Nifti1Image(m, affine=voxel_mask.affine)
    ###########################################################################
    # saving
    
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / f"{experiment_name}_attribution_map.nii"
    result_map.to_filename(file_path)

    return result_map, file_path
