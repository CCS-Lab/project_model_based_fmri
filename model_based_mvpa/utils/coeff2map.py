#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho
@contact: cjfwndnsl@gmail.com
@last modification: 2020.11.02
"""

import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp
from scipy.ndimage import gaussian_filter


def coeff_to_zmap(coeffs, masked_data, smoothing_sigma=1, save_file_name=None):
    
    act_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0] 

    for coeff in coeffs:
        act_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id,coeff):
            act_map[i] = v
        act_map = act_map.reshape(masked_data.shape)
        act_map = gaussian_filter(act_map,smoothing_sigma)
        act_maps.append(act_map)

    act_maps = np.array(act_maps)
    act_maps[np.isnan(act_maps)] = 0

    z_map = ((act_maps-act_maps.mean())/act_maps.std()).mean(0)
    z_map *= masked_data.get_fdata()
    nii_i = nib.Nifti1Image(z_map, affine=masked_data.affine)
    if save_file_name:
        nii_i.to_filename(save_file_name+'_zmap.nii')
        
    return nii_i
    

def coeff_to_tmap(coeffs, masked_data, smoothing_sigma=1):
    
    act_maps = []
    mapping_id = np.nonzero(masked_data.get_fdata().flatten())[0] 

    for coeff in coeffs:
        act_map = np.zeros(masked_data.get_fdata().flatten().shape[0])
        for i, v in zip(mapping_id,coeff):
            act_map[i] = v
        act_map = act_map.reshape(masked_data.shape)
        act_map = gaussian_filter(act_map,smoothing_sigma)
        act_maps.append(act_map)

    act_maps = np.array(act_maps)
    act_maps[np.isnan(act_maps)] = 0

    t_map = ttest_1samp(act_maps,0).statistic
    t_map *= masked_data.get_fdata()
    nii_i = nib.Nifti1Image(t_map, affine=masked_data.affine)
    
    if save_file_name:
        nii_i.to_filename(save_file_name+'_tmap.nii')
        
    return nii_i