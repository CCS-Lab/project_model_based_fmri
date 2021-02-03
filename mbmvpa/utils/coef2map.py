#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp, zscore


def reconstruct(array, mask):
    assert (mask==1).sum() == len(array)
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    blackboard = np.zeros(list(mask.shape))
    blackboard[mask.nonzero()] = array.squeeze()
    
    return blackboard
    
    
def get_map(coefs, voxel_mask, task_name,
            map_type="z", save_path=None, sigma=1):
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
    assert isinstance(task_name, str)
    assert isinstance(map_type, str)
    assert (isinstance(save_path, str)
        or isinstance(save_path, Path))
    assert isinstance(sigma, int)
    ###########################################################################
    # make result map

    activation_maps = []
    
    if not isinstance(voxel_mask,np.ndarray):
        mask = voxel_mask.get_fdata()
    else:
        mask = voxel_mask
        
    for coef in coefs:
        # converting flattened coefs to brain image.
        activation_map = reconstruct(coef, mask)
        
        if sigma > 0:
            activation_map = gaussian_filter(activation_map, sigma)
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    if map_type == "t":
        # voxel-wise one sample ttest
        m = ttest_1samp(activation_maps, 0).statistic
    else:
        # averaging z_score of each voxel
        m = zscore(activation_maps, axis=None).mean(0)

    m[np.isnan(m)] = 0
    m *= mask
    result_map = nib.Nifti1Image(m, affine=voxel_mask.affine)
    ###########################################################################
    # saving
    if save_path is None:
        sp = Path(".")
    else:
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()

    result_map.to_filename(sp / f"{task_name}_{map_type}_map.nii")

    return result_map
