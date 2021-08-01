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
    '''reconstruct flattened array to 3D with the given mask.
    
    Parameters
    ----------
    
    array : numpy.ndarray
        array with shape of (N,) or (N,1)
    mask : numpy.ndarray
        3D binary mask where sum(mask)==N
        
    Returns
    -------
    
    numpy.ndarray
        Reconstructed 3D array
    
    '''
    
    if array.shape[-1] == 1:
        array = array.squeeze(-1)
    
    blackboard = np.zeros(mask.shape)
    blackboard[mask.nonzero()] = array
    
    return blackboard
    
def cluster_level_correction(brainmap, threshold, cluster_threshold):
    '''thresholding and cluster-level correction
    
    Parameters
    ----------
    
    brainmap : nibabel.nifti1.Nifti1Image
        Nii image to be thresholded and cluster-level corrected.
        
    threshold : float
        Threshold value to cutoff the image
        Both negative and positie values will be zero if abs(v) <= threshold
        
    cluster_threshold : int
        Threshold for the number of points in a cluster to be cutoff
        
    Returns
    -------
    
    nibabel.nifti1.Nifti1Image
        Thresholded and cluster-level corrected nii image.
    
    '''
    stat_map = brainmap
    affine = stat_map.affine
    stat_map = stat_map.get_fdata()
    if cluster_threshold > 0:
        label_map, n_labels = label(stat_map > threshold)

        for label_ in range(1, n_labels + 1):
            if np.sum(label_map == label_) < cluster_threshold:
                stat_map[label_map == label_] = 0
    
    if cluster_threshold > 0:
        label_map, n_labels = label(stat_map < -threshold)

        for label_ in range(1, n_labels + 1):
            if np.sum(label_map == label_) < cluster_threshold:
                stat_map[label_map == label_] = 0
    
    
    stat_map[np.abs(stat_map) <= threshold] = 0
    stat_map = nib.Nifti1Image(stat_map,affine)
    
    return stat_map

def get_map(coefs, voxel_mask, experiment_name, standardize=False, save_path=".", smoothing_fwhm=0,
            threshold=0, cluster_threshold=0):
    
    """make nii image file from coefficients of model.
    
    Parameters
    ----------
    
        coefs : list of numpy.array or numpy.array
            List of coefficients extracted from MVPA models
            
        voxel_mask : nibabel.nifti1.Nifti1Image
            Nii image of mask
            
        experiment_name : str
            Name of experiment. It will be used to name the resulting image.
            
        standardize : bool, default=False
            Indicate if resulting brain map is required to be standardized.
        
        save_path : str or pathlib.PosixPath
            Path to save created map.
        
        smoothing_fwhm : float, default=0
            Size in millimeters of the spatial smoothing of each reconstructed map.
        
        threshold : float, default=0
            Threshold value for thresholding resulting image.
        
        cluster_threshold : int, default=0
            Threshold for the number of points in a cluster to be cutoff resulting image.
    
    Returns
    -------
        nibabel.nifti1.Nifti1Image
            Nii file for resulting brain map.
            
        pathlib.PosixPath
            Path where the resulting image is saved.
            
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
        #if standardize:
            #coef = zscore(coef,axis=None)
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
    #m = ttest_1samp(activation_maps, 0).statistic
    m = activation_maps.mean(0)
        
    m[np.isnan(m)] = 0
    
    if standardize:
        m_m = m[mask.nonzero()]
        m_mean = m_m.mean()
        m_std = m_m.std()
        m = (m-m_mean)/m_std
        
    m *= mask
    
    result_map = nib.Nifti1Image(m, affine=voxel_mask.affine)
    result_map = cluster_level_correction(result_map,threshold, cluster_threshold)
    ###########################################################################
    # saving
    
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / f"{experiment_name}_map.nii"
    result_map.to_filename(file_path)

    return result_map, file_path
