#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com

r'''
referenece - https://github.com/nilearn/nilearn/blob/297b1509/nilearn/glm/first_level/first_level.py#L803

'''

import glob, json, os, sys, time
from warnings import warn
import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nibabel import Nifti1Image
from nilearn.glm.first_level import *
import nibabel as nib
from bids import BIDSLayout
from mbfmri.utils import config

def _fit_firstlevel_model(params):
            
    models, models_run_imgs, models_events, \
        models_confounds, process_name, save_path_first = params
    
    models.fit([nib.load(run_img) for run_img in models_run_imgs],
              events=models_events,
              confounds=models_confounds)
    contrast_def = [np.zeros( len(dm.columns)) for dm in models.design_matrices_]
    for i, dm in enumerate(models.design_matrices_):
        contrast_def[i][dm.columns.get_loc(process_name)] = 1

    z_map = models.compute_contrast(contrast_def=contrast_def,
                                               output_type='z_score')
    subject_id = models.subject_label
    nib.save(z_map, save_path_first / f'sub-{subject_id}_map.nii')

            
            
def first_level_from_bids(bids_layout, task_name, process_name, 
                          space_name="MNI152NLin2009cAsym",
                          subjects='all',sessions='all',bold_suffix="bold",
                          modulation_suffix="modulation",
                          confound_suffix="regressors",
                          img_filters=None, t_r=None, slice_time_ref=0.,
                          hrf_model='glover', drift_model='cosine',
                          high_pass=1/128, drift_order=1, fir_delays=[0],
                          min_onset=-24, mask_img=None,
                          target_affine=None, target_shape=None,
                          smoothing_fwhm=None, memory=Memory(None),
                          memory_level=1, standardize=True,
                          signal_scaling=0, noise_model='ar1',
                          verbose=0, n_jobs=1,
                          minimize_memory=True,
                          confound_names=['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']):
    
    """Create FirstLevelModel objects and fit arguments from a BIDS dataset.
    """
    if high_pass is None :
        high_pass = .01
    if not isinstance(bids_layout,BIDSLayout):
        bids_layout =  BIDSLayout(root=bids_layout,derivatives=True)
        
    bold_kwargs = {'task':task_name,
                   'suffix':bold_suffix,
                     'extension':config.NIIEXT,
                  'space':space_name}
    
    confound_kwargs = {'task':task_name,
                       'suffix':confound_suffix,
                      'extension':config.CONFOUNDEXT}
    
    modulation_kwargs = {'task':task_name,
                         'suffix':modulation_suffix,
                        'extension':config.MODULATIONEXT,
                        'desc':process_name}
    
    spec_kwargs = {'task':task_name,
                   'suffix':bold_suffix,
                     'extension':config.SPECEXT,
                  'space':space_name}
    
    if subjects == 'all':
        subjects = bids_layout.get_subjects()
    else:
        subjects = subjects
    
    if sessions =='all':
        sessions = bids_layout.get_sessions()
        if len(sessions) ==0:
            has_session = False
        else:
            has_session = True
    else:
        sessions = sessions
        assert len(sessions) > 0
        has_session = True
    
    if has_session:
        bold_kwargs['session'] = sessions
        
    models = []
    models_bold_imgs = []
    models_modulations = []
    models_confounds = []
    
    for subject in subjects:
        candidate_bold_imgs = bids_layout.get(subject=subject,**bold_kwargs)
        bold_imgs = []
        modulations = []
        confounds = []
        specs = []
        for bold_img in candidate_bold_imgs:
            entities = bold_img.get_entities()
            if 'session' in entities.keys(): 
                # if session is included in BIDS
                ses_id = entities['session']
            else:
                ses_id = None
            if 'run' in entities.keys(): 
                # if run is included in BIDS
                run_id = entities['run']
            else:
                run_id = None
                
            modulation = bids_layout.get(subject=subject,
                                         run=run_id,
                                         session=ses_id,
                                         **modulation_kwargs)
            
            confound = bids_layout.get(subject=subject,
                                         run=run_id,
                                         session=ses_id,
                                         **confound_kwargs)
            
            spec = bids_layout.get(subject=subject,
                                         run=run_id,
                                         session=ses_id,
                                         **spec_kwargs)
            
            if len(modulation) ==0 or len(confound)==0 or len(spec)==0:
                continue
                
            modulation = pd.read_table(modulation[0])
            modulation['trial_type'] = [process_name]*len(modulation)
            confound = pd.read_csv(confound[0], sep='\t', index_col=None)
            confound = confound[confound_names]
            
            bold_imgs.append(bold_img)
            modulations.append(modulation)
            confounds.append(confound)
            specs.append(spec)
        
        
        if len(specs) != 0:
            specs = json.load(open(spec[0], 'r'))
        else:
            specs = {}
            
        if 'RepetitionTime' in specs:
            _t_r = float(specs['RepetitionTime'])
        else:
            _t_r = t_r
            
        if 'SliceTimingRef' in specs:
            _slice_time_ref = float(specs['SliceTimingRef'])
        else:
            _slice_time_ref = slice_time_ref
            
        model = FirstLevelModel(
            t_r=_t_r, slice_time_ref=_slice_time_ref, hrf_model=hrf_model,
            drift_model=drift_model, high_pass=high_pass,
            drift_order=drift_order, fir_delays=fir_delays,
            min_onset=min_onset, mask_img=mask_img,
            target_affine=target_affine, target_shape=target_shape,
            smoothing_fwhm=smoothing_fwhm, memory=memory,
            memory_level=memory_level, standardize=standardize,
            signal_scaling=signal_scaling, noise_model=noise_model,
            verbose=verbose, n_jobs=n_jobs,
            minimize_memory=minimize_memory, subject_label=subject)
        
        assert len(bold_imgs)==len(modulations)
        assert len(bold_imgs)==len(confounds)
        
        if len(bold_imgs) == 0:
            continue
            
        models.append(model)
        models_bold_imgs.append(bold_imgs)
        models_modulations.append(modulations)
        models_confounds.append(confounds)
            
    return models, models_bold_imgs, models_modulations, models_confounds