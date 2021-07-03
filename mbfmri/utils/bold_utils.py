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
from nilearn.datasets import load_mni152_brain_mask
import nibabel as nib
from mbfmri.utils import config
from scipy import ndimage


PATH_ROOT = Path(__file__).absolute().parent
PATH_EXTDATA = (PATH_ROOT/'extdata').resolve()
GM_PATH = PATH_EXTDATA/ 'gray_matter.nii.gz'
# downloaded from https://github.com/Jfortin1/MNITemplate

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
    


from scipy import ndimage

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

def _build_mask(mask_path, threshold, zoom,smoothing_fwhm, verbose=0, gm_only=False):
    
    # list up mask image file
    if gm_only:
        mni_mask = nib.load(str(GM_PATH))
    else:
        mni_mask = load_mni152_brain_mask()
    if mask_path is None:
        include_mask_files = []
        exclude_mask_files = []
        m = (mni_mask.get_fdata()>0)
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
        include_mask = _integrate_mask(include_mask_files,mni_mask,threshold,smoothing_fwhm, verbose)
        if verbose > 0:
            survived = int(include_mask.sum())
            total = np.prod(include_mask.shape)
            print('INFO: integrated mask to include'+f': {survived}/{total}')
        print('INFO: start loading & intergrating masks to exclude')
        exclude_mask = _integrate_mask(exclude_mask_files,mni_mask,threshold,smoothing_fwhm, verbose)
        if verbose > 0:
            survived = int(exclude_mask.sum())
            total = np.prod(exclude_mask.shape)
            print('INFO: integrated mask to exclude'+f': {survived}/{total}')

        m = (mni_mask.get_fdata()>0) & include_mask  & (~exclude_mask)

    if verbose > 0:
        survived = int(m.sum())
        total = np.prod(m.shape)
        print('      final mask'+f': {survived}/{total}')
    # reduce dimension by averaging zoom window
    affine = mni_mask.affine.copy()
    
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
    """
    Make custom ROI mask to reduce the number of features.
    """

    """
    Arguments:
        mask_path (str or Path): a path for the directory containing mask files (nii or nii.gz). encourage get files from Neurosynth
        threshold (float): threshold for binarizing mask images
        zoom ((float,float,float)): zoom window, indicating a scaling factor for each dimension in x,y,z. the dimension will be reduced by the factor of corresponding axis.
                                    e.g. (2,2,2) will make the dimension half in all directions, so 2x2x2=8 voxels will be 1 voxel.
        smoothing_fwhm (int):the amount of spatial smoothing. if None, image will not be smoothed.
        interpolation_func (numpy.func): a method to calculate a representative value in the zooming window. e.g. numpy.mean, numpy.max
                                         e.g. zoom=(2,2,2) and interpolation_func=np.mean will convert 2x2x2 cube into a single value of its mean.
        standardize (boolean): if true, conduct standard normalization within each image of a single run. 
    Return:
        voxel_mask (nibabel.nifti1.Nifti1Image): a nifti image for voxel-wise binary mask (ROI mask)
        masker (nilearn.input_data.NiftiMasker): a masker object. will be used for correcting motion confounds, and masking.
    """
    
        
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
    """
    Make image that is motion corrected and ROI masked.
    This function wrapped up functions from nilearn package.
    """

    """
    Arguments:
        params: params must have below contents
            image_path (str or Path): path of fMRI nii file 
            confounds_path (str or Path): path of corresponding motion confounds tsv file
            motion_confounds (list[str]): list of name indicating motion confound names in confounds tsv file
            masker (nilearn.input_data.NiftiMasker): masker object. will be used for correcting motion confounds, and masking.
            voxel_mask (nibabel.nifti1.Nifti1Image): nifti image for voxel-wise binary mask
            subject_id (str): subject ID. used to track the owner of the file in multiprocessing

    Return:
        fmri_masked (numpy.ndarray): preprocessed image with shape run # x time point # x voxel #
        subject_id (str): subject ID. used to track the owner of the file in multiprocessing
    """

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

    # different from resample_img.
    # resample_img: need to target affine and shape.
    # resample_to_img: need to target image including affine.
    # ref.: https://nilearn.github.io/modules/generated/nilearn.image.resample_img.html
    #       https://nilearn.github.io/modules/generated/nilearn.image.resample_to_img.html
    #fmri_masked = resample_to_img(str(image_path), voxel_mask)
    #fmri_masked = masker.fit_transform(fmri_masked, confounds=confounds)
    fmri_masked = masker.transform_single_imgs(nib.load(str(image_path)), confounds=confounds)
    np.save(save_path, fmri_masked)
    
    return 1