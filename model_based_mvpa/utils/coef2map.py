#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho
@contact: cjfwndnsl@gmail.com
@last modification: 2020.11.02
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp, zscore


def get_map(coefs, masked_data, layout=None, task_name=None,
            map_type='z', save=False, save_path=None, sigma=1):
    """
    make nii image file from coefficients of model using masking info.
    """

    """
    Arguments:
        coefs: extracted coefs by fitting model.
               shape: N x voxel #
        masked_data: a binary nii image of masking info. type : nibabel.nifti1.Nifti1Image, 
                     should be same shape with the original fMRI image data
        layout: layout info to get task_name if None, task_name should not be None instead
        task_name: if provided, the saved nii file will be named after this
        map_type: the type of making map. ('z' or 't')
                  'z': default type. z_map by mean z_score among N mapped images.
                  't': t_map by voxel-wise one sample t-test among N mapped images.
        save_path: path to save file. if None, then will be saved in 'results' directory fo current working directory
                   the result_map nii image file will be saved in 'save_path/{task_name}_{map_type}_map.nii'
        sigma: the sigma value for spatial gaussian smoothing for each converted map.  
               if 0, there will be no spatial smoothing of extracted coefficients.
               higher the value, more smoother the final map will be 

    return:
        result_map: a nii image of brain activation map.
    """

    assert (not(layout is None and task_name is None))

    activation_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0]

    for coef in coefs:
        # converting flattened coefs to brain image.
        activation_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id, coef):
            activation_map[i] = v

        activation_map = activation_map.reshape(masked_data.shape)
        if sigma > 0:
            activation_map = gaussian_filter(activation_map, sigma)
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    if map_type == 't':
        # voxel-wise one sample ttest
        m = ttest_1samp(activation_maps, 0).statistic
    else:
        # averaging z_score of each voxel
        m = zscore(activation_maps, axis=None).mean(0)

    m[np.isnan(m)] = 0
    m *= masked_data.get_fdata()
    result_map = nib.Nifti1Image(m, affine=masked_data.affine)

    # saving
    if save_path is None:
        sp = Path('./results')
    else:
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()

    if task_name is None:
        task_name = layout.get_task()[0]

    if save:
        result_map.to_filename(sp / f'{task_name}_{map_type}_map.nii')

    return result_map
