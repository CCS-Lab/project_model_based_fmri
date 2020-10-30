import numpy as np
import pandas as pd
from pathlib import Path

from skimage.measure import block_reduce
from scipy import stats
from nilearn.image import resample_to_img
import logging


logging.basicConfig(encoding='utf-8', level=logging.INFO)


def array2pindex(array, p_value=0.05, flatten=False):
    confidence = 1 - p_value
    flattened_array = array.flatten()
    
    n = len(flattened_array)
    m = np.mean(flattened_array)
    std_err = stats.sem(flattened_array)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    end = m + h
    
    ret = (flattened_array >= end) if flatten is True else (array >= end)
    return ret


def custom_masking(mask_path, p_value, zoom,
                   smoothing_fwhm, interpolation_func, standardize,
                   flatten=False):

    if mask_path is None:
        assert (mask_path is None)
    elif type(mask_path) is str:
        mask_files = [mask_path]
    else:
        if type(mask_path) is not type(Path()):
            mask_path = Path(mask_path)
        mask_files = [file for file in mask_path.glob('*.nii.gz')]

    image_sample = nib.load(mask_files[0])
    m = array2pindex(image_sample.get_fdata(), p_value, flatten)

    for i in range(1, len(mask_files)):
        m |= array2pindex(nib.load(mask_files[i]).get_fdata(), p_value, flatten)
    
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
    image_filepath, confounds, motion_confounds, masker, masked_data, num = params
    
    if confounds is not None:
        confounds = pd.read_table(confounds, sep='\t')
        confounds = confounds[motion_confounds]
        confounds = confounds.to_numpy()

    fmri_masked = resample_to_img(image_filepath, masked_data)
    fmri_masked = masker.fit_transform(fmri_masked, confounds=confounds)
    
    return fmri_masked
