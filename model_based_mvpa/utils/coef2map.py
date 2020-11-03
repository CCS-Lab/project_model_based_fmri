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


def get_map(coefs, masked_data, layout,
            map_type='t', save_path=None, smoothing_sigma=1):

    activation_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0] 

    for coef in coefs:
        activation_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id, coef):
            activation_map[i] = v

        activation_map = activation_map.reshape(masked_data.shape)
        activation_map = gaussian_filter(activation_map,smoothing_sigma)
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    if map_type == 't':
        m = ttest_1samp(activation_maps, 0).statistic
    else:
        m = zscore(activation_maps)

    m *= masked_data.get_fdata()
    result = nib.Nifti1Image(m, affine=masked_data.affine)
    
    if save_path is None:
        sp = Path('./results')
    else: 
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()
        
    result.to_filename(sp / f'{layout.get_task()[0]}_{map_type}_map.nii')

    return result


def get_zmap(coefs, masked_data, layout, save_path=None, smoothing_sigma=1):
    activation_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0] 

    for coef in coefs:
        activation_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id, coef):
            activation_map[i] = v
            
        activation_map = activation_map.reshape(masked_data.shape)
        activation_map = gaussian_filter(activation_map,smoothing_sigma)
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    z_map = ((activation_maps-  activation_maps.mean()) / activation_maps.std()).mean(0)
    z_map *= masked_data.get_fdata()
    result = nib.Nifti1Image(z_map, affine=masked_data.affine)

    if save_path is None:
        sp = Path('./results')
    else: 
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()
        
    result.to_filename(sp / f'{layout.get_task()[0]}_z_map.nii')
        
    return result


def get_tmap(coefs, masked_data, layout, save_path=None, smoothing_sigma=1):
    activation_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0] 

    for coef in coefs:
        activation_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id, coef):
            activation_map[i] = v

        activation_map = activation_map.reshape(masked_data.shape)
        activation_map = gaussian_filter(activation_map,smoothing_sigma)
        activation_maps.append(activation_map)

    activation_maps = np.array(activation_maps)
    activation_maps[np.isnan(activation_maps)] = 0

    t_map = ttest_1samp(activation_maps,0).statistic
    t_map *= masked_data.get_fdata()
    result = nib.Nifti1Image(t_map, affine=masked_data.affine)
    
    if save_path is None:
        sp = Path('./results')
    else: 
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()
        
    result.to_filename(sp / f'{layout.get_task()[0]}_t_map.nii')

        
    return result