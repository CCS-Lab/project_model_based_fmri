#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho
@contact: cjfwndnsl@gmail.com
@last modification: 2020.11.02
"""

import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp, zscore
from scipy.ndimage import gaussian_filter

from pathlib import Path


def get_map(coefs, masked_data, task_name,
            map_type='t', save_path=None, smoothing_sigma=1):

    activation_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0]
    
    for coef in coefs:
        activation_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id, coef):
            activation_map[i] = v

        activation_map = activation_map.reshape(masked_data.shape)
        if smoothing_sigma > 0:
            activation_map = gaussian_filter(activation_map,smoothing_sigma)
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    if map_type == 't':
        m = ttest_1samp(activation_maps, 0).statistic
    else:
        mean = activation_maps.mean()
        std = activation_maps.std()
        m = ((activation_maps-mean)/std).mean(0)
        
    m[np.isnan(m)] = 0
    m *= masked_data.get_fdata()
    result = nib.Nifti1Image(m, affine=masked_data.affine)
    
    if save_path is None:
        sp = Path('./results')
    else: 
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()
    result.to_filename(sp / f'{task_name}_{map_type}_map.nii')

    return result