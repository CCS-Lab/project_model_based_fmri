import numpy as np
import pandas as pd

from pathlib import Path

from hbayesdm.models import ra_prospect

import nibabel as nib

from bids import BIDSLayout
from tqdm import tqdm

from sklearn.linear_model import ElasticNet, LinearRegression

from scipy.stats import zscore

import matplotlib.pyplot as plt

import nilearn as nil
from nilearn.image import resample_to_img, load_img, smooth_img
from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map, show

from nipy.modalities.fmri import hemodynamic_models
from nipy.modalities.fmri.hemodynamic_models import compute_regressor

import pickle

RAW_DATA_DIR = Path('data/tom_2007/ds000005')
PREP_DIR = Path('output/fmriprep')

layout = BIDSLayout(RAW_DATA_DIR, derivatives=True)
layout.add_derivatives(PREP_DIR)

template = load_mni152_template()
mask = load_mni152_brain_mask()

def get_masked_fmri(layout,subj_i,run_j):
    subj_i = 1
    run_j = 1
    image_sample = layout.derivatives['fMRIPrep'].get(
            subject=f'{subj_i:02d}',
            run= f'{run_j}',
            return_type='file',
            suffix='bold',
            extension='nii.gz')[0]

    masker = NiftiMasker(mask_img=mask, standardize=True)
    fmri_masked = masker.fit_transform(image_sample)
    return fmri_masked 

subj_num = 16
run_num = 3
Y = np.array([[get_masked_fmri(layout,subj_i,run_j) for run_j in tqdm(range(1,run_num+1))] for subj_i in range(1,subj_num+1)])
#print(Y.shape)

Y.dump("temp_Y.dat")