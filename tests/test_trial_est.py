import nibabel as nib
from bids import BIDSLayout
from pathlib import Path
import pandas as pd
from skimage.measure import block_reduce
from nilearn.datasets import load_mni152_brain_mask
from nilearn.image import resample_to_img
import numpy as np
from scipy import linalg
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from nilearn.glm.first_level.first_level import run_glm
from nilearn.image.resampling import resample_img
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from tqdm import tqdm
import pdb

root = "/data2/project_modelbasedMVPA/ds000005"
layout = BIDSLayout(root=root, derivatives=True)
mask_path = "/data2/project_modelbasedMVPA/ds000005/derivatives/fmriprep/masks"

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


def get_roi_mask(mask_path, zoom, threshold):    
    if mask_path is None:
        mask_files = []
    else:
        if type(mask_path) is not type(Path()):
            mask_path = Path(mask_path)
        mask_files = [file for file in mask_path.glob("*.nii.gz")]

    mni_mask = load_mni152_brain_mask()
    # integrate binary mask data
    if len(mask_files) > 0 :
        # binarize
        m = abs(nib.load(mask_files[0]).get_fdata()) >= threshold
        for i in range(len(mask_files)-1):
            # binarize and stack
            m |= abs(nib.load(mask_files[0]).get_fdata()) >= threshold
    else:
        # if not provided, use min_152 mask instead.
        m = mni_mask.get_fdata()
    
    affine = mni_mask.affine.copy()
    
    if zoom != (1, 1, 1):
        voxel_mask = _zoom_img(m, affine, zoom, binarize=True)
    else:
        voxel_mask = nib.Nifti1Image(m.astype(float), affine=affine)
        
    return voxel_mask
        

voxel_mask = get_roi_mask(mask_path, zoom=(2,2,2), threshold = 2.58)


def trial_estimation(prepimg_path, voxel_mask, confounds_path, confound_names, events_path):
    fmri_img = resample_to_img(str(prepimg_path ), voxel_mask)
    masker = NiftiMasker(mask_img=voxel_mask,standardize=True,t_r=2.0,high_pass=1/128)
    y_data = masker.fit_transform(fmri_img )
    confounds = pd.read_table(confounds_path,sep="\t")
    confounds = confounds[confound_names]
    confounds = confounds.to_numpy()
    confounds[np.isnan(confounds)] = 0
    std = confounds.std(0)
    mean = confounds.mean(0)
    confounds = (confounds-mean)/std
    
    events = pd.read_table(events_path, sep="\t")

    onsets = events['onset'].to_numpy()
    durations = events['duration'].to_numpy()

    signals = []
    frame_times = 2.0 * \
                np.arange(y_data.shape[0]) + \
                 2.0 / 2.
    for onset, duration in zip(onsets,durations):
        boldified_signals, name = compute_regressor(
            exp_condition=np.array([[onset, duration, 1]]).T,
            hrf_model='spm', #'spm + derivative + dispersion',
            frame_times=frame_times)
        signals.append(boldified_signals)
    
    confounds_length = confounds.shape[-1]
    matrix = np.concatenate(signals+[confounds]+[np.ones([y_data.shape[0],1])],-1)
    labels, results = run_glm(y_data,matrix)
    
    trial_t_maps = []
    for i in range(len(onsets)):
        contrast = np.zeros(matrix.shape[-1])
        contrast[i] = 1
        t_value = {}
        for label,data in results.items():
            t_value[label] = list(data.Tcontrast(contrast).t.reshape(-1))

        trial_t_map = np.array([t_value[label].pop(0) for label in labels])
        trial_t_maps.append(trial_t_map)

    trial_t_maps = np.array(trial_t_maps)

    return trial_t_maps


total_prepimg_infos = layout.derivatives['fMRIPrep'].get(suffix='bold',extension='nii.gz')


confound_names = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

data_dict = {}
for prepimg_info in tqdm(total_prepimg_infos):
    run_id = prepimg_info.entities['run']
    subject_id = prepimg_info.entities['subject']
    confounds_path = layout.derivatives['fMRIPrep'].get(subject=subject_id,run=run_id,suffix='regressors',extension='tsv')[0].path
    events_path = layout.get(subject=subject_id,run=run_id,suffix='events',extension='tsv')[0].path
    prepimg_path = prepimg_info.path
    data_dict[f'{subject_id}/{run_id}'] = trial_estimation(prepimg_path, voxel_mask, confounds_path, confound_names, events_path)

from scipy.io import savemat, loadmat

savemat("trial_estimation.mat", data_dict)

target_dict = {}
for key in data_dict.keys():
    subject_id, run_id = key.split('/')
    modulation_path = layout.derivatives['MB-MVPA'].get(subject=subject_id,run=run_id,suffix='modulation')[0].path
    modulation = pd.read_table(modulation_path, sep='\t')
    modulation = modulation['modulation'].to_numpy()
    target_dict[key] = modulation

subjectwise_dict = {}
for key,data in target_dict.items():
    subject_id, run_id = key.split('/')
    if subject_id not in subjectwise_dict.keys():
        subjectwise_dict[subject_id]  = []
    subjectwise_dict[subject_id].append(data)

subject_stat ={}
for key,data in subjectwise_dict.items():
    data = np.concatenate(data,-1)
    subject_stat[key] = {'mean':data.mean(), 'std':data.std()}
    

normalized_target_dict = {}
for key,data in target_dict.items():
    subject_id, run_id = key.split('/')
    mean = subject_stat[subject_id]['mean']
    std = subject_stat[subject_id]['std']
    normalized_target_dict[key] = (data-mean)/mean

savemat("trial_modulation.mat", normalized_target_dict)