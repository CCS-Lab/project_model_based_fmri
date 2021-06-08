import glob
import json
import os
import sys
import time
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Memory, Parallel, delayed
from nibabel import Nifti1Image
from nibabel.onetime import auto_attr
from sklearn.base import clone

from nilearn._utils.glm import (_check_events_file_uses_tab_separators,
                                _check_run_tables, get_bids_files,
                                parse_bids_filename)
from nilearn._utils.niimg_conversions import check_niimg
from nilearn.glm.contrasts import (_compute_fixed_effect_contrast,
                                   expression_to_contrast_vector)
from nilearn.glm.first_level.design_matrix import \
    make_first_level_design_matrix
from nilearn.image import get_data
from nilearn.glm.regression import (ARModel, OLSModel, RegressionResults,
                                    SimpleRegressionResults)
from nilearn.glm._base import BaseGLM

from nilearn.glm.first_level import *

import nibabel as nib

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
            nib.save(z_map, save_path_first / f'sub-{subject_id}.nii')
            
def first_level_from_bids(dataset_path, task_label, space_label=None,
                          img_filters=None, t_r=None, slice_time_ref=0.,
                          hrf_model='glover', drift_model='cosine',
                          high_pass=.01, drift_order=1, fir_delays=[0],
                          min_onset=-24, mask_img=None,
                          target_affine=None, target_shape=None,
                          smoothing_fwhm=None, memory=Memory(None),
                          memory_level=1, standardize=False,
                          signal_scaling=0, noise_model='ar1',
                          verbose=0, n_jobs=1,
                          minimize_memory=True,
                          derivatives_folder='derivatives',
                          subjects='all'):
    """Create FirstLevelModel objects and fit arguments from a BIDS dataset.
    It t_r is not specified this function will attempt to load it from a
    bold.json file alongside slice_time_ref. Otherwise t_r and slice_time_ref
    are taken as given.
    Parameters
    ----------
    dataset_path : str
        Directory of the highest level folder of the BIDS dataset. Should
        contain subject folders and a derivatives folder.
    task_label : str
        Task_label as specified in the file names like _task-<task_label>_.
    space_label : str, optional
        Specifies the space label of the preprocessed bold.nii images.
        As they are specified in the file names like _space-<space_label>_.
    img_filters : list of tuples (str, str), optional
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Possible filters are 'acq', 'ce', 'dir', 'rec', 'run', 'echo', 'res',
        'den', and 'desc'. Filter examples would be ('desc', 'preproc'),
        ('dir', 'pa') and ('run', '10').
    derivatives_folder : str, optional
        derivatives and app folder path containing preprocessed files.
        Like "derivatives/FMRIPREP". Default="derivatives".
    All other parameters correspond to a `FirstLevelModel` object, which
    contains their documentation. The subject label of the model will be
    determined directly from the BIDS dataset.
    Returns
    -------
    models : list of `FirstLevelModel` objects
        Each FirstLevelModel object corresponds to a subject. All runs from
        different sessions are considered together for the same subject to run
        a fixed effects analysis on them.
    models_run_imgs : list of list of Niimg-like objects,
        Items for the FirstLevelModel fit function of their respective model.
    models_events : list of list of pandas DataFrames,
        Items for the FirstLevelModel fit function of their respective model.
    models_confounds : list of list of pandas DataFrames or None,
        Items for the FirstLevelModel fit function of their respective model.
    """
    # check arguments
    img_filters = img_filters if img_filters else []
    if not isinstance(dataset_path, str):
        raise TypeError(
            'dataset_path must be a string, instead %s was given' %
            type(task_label))
    if not os.path.exists(dataset_path):
        raise ValueError('given path do not exist: %s' % dataset_path)
    if not isinstance(task_label, str):
        raise TypeError('task_label must be a string, instead %s was given' %
                        type(task_label))
    if space_label is not None and not isinstance(space_label, str):
        raise TypeError('space_label must be a string, instead %s was given' %
                        type(space_label))
    if not isinstance(img_filters, list):
        raise TypeError('img_filters must be a list, instead %s was given' %
                        type(img_filters))
    for img_filter in img_filters:
        if (not isinstance(img_filter[0], str)
                or not isinstance(img_filter[1], str)):
            raise TypeError('filters in img filters must be (str, str), '
                            'instead %s was given' % type(img_filter))
        if img_filter[0] not in ['acq', 'ce', 'dir', 'rec', 'run',
                                 'echo', 'desc', 'res', 'den',
                                 ]:
            raise ValueError(
                "field %s is not a possible filter. Only "
                "'acq', 'ce', 'dir', 'rec', 'run', 'echo', "
                "'desc', 'res', 'den' are allowed." % img_filter[0])

    # check derivatives folder is present
    derivatives_path = os.path.join(dataset_path, derivatives_folder)
    if not os.path.exists(derivatives_path):
        raise ValueError('derivatives folder does not exist in given dataset')

    # Get acq specs for models. RepetitionTime and SliceTimingReference.
    # Throw warning if no bold.json is found
    if t_r is not None:
        warn('RepetitionTime given in model_init as %d' % t_r)
        warn('slice_time_ref is %d percent of the repetition '
             'time' % slice_time_ref)
    else:
        filters = [('task', task_label)]
        for img_filter in img_filters:
            if img_filter[0] in ['acq', 'rec', 'run']:
                filters.append(img_filter)

        img_specs = get_bids_files(derivatives_path, modality_folder='func',
                                   file_tag='bold', file_type='json',
                                   filters=filters)
        # If we dont find the parameter information in the derivatives folder
        # we try to search in the raw data folder
        if not img_specs:
            img_specs = get_bids_files(dataset_path, modality_folder='func',
                                       file_tag='bold', file_type='json',
                                       filters=filters)
        if not img_specs:
            warn('No bold.json found in derivatives folder or '
                 'in dataset folder. t_r can not be inferred and will need to'
                 ' be set manually in the list of models, otherwise their fit'
                 ' will throw an exception')
        else:
            specs = json.load(open(img_specs[0], 'r'))
            if 'RepetitionTime' in specs:
                t_r = float(specs['RepetitionTime'])
            else:
                warn('RepetitionTime not found in file %s. t_r can not be '
                     'inferred and will need to be set manually in the '
                     'list of models. Otherwise their fit will throw an '
                     ' exception' % img_specs[0])
            if 'SliceTimingRef' in specs:
                slice_time_ref = float(specs['SliceTimingRef'])
            else:
                warn('SliceTimingRef not found in file %s. It will be assumed'
                     ' that the slice timing reference is 0.0 percent of the '
                     'repetition time. If it is not the case it will need to '
                     'be set manually in the generated list of models' %
                     img_specs[0])

    # Infer subjects in dataset
    sub_folders = glob.glob(os.path.join(derivatives_path, 'sub-*/'))
    sub_labels = [os.path.basename(s[:-1]).split('-')[1] for s in sub_folders]
    sub_labels = sorted(list(set(sub_labels)))
    
    if subjects != 'all':
        sub_labels = [l for l in sub_labels if l in subjects]

    # Build fit_kwargs dictionaries to pass to their respective models fit
    # Events and confounds files must match number of imgs (runs)
    models = []
    models_run_imgs = []
    models_events = []
    models_confounds = []
    for sub_label in sub_labels:
        # Create model
        model = FirstLevelModel(
            t_r=t_r, slice_time_ref=slice_time_ref, hrf_model=hrf_model,
            drift_model=drift_model, high_pass=high_pass,
            drift_order=drift_order, fir_delays=fir_delays,
            min_onset=min_onset, mask_img=mask_img,
            target_affine=target_affine, target_shape=target_shape,
            smoothing_fwhm=smoothing_fwhm, memory=memory,
            memory_level=memory_level, standardize=standardize,
            signal_scaling=signal_scaling, noise_model=noise_model,
            verbose=verbose, n_jobs=n_jobs,
            minimize_memory=minimize_memory, subject_label=sub_label)

        # Get preprocessed imgs
        if space_label is None:
            filters = [('task', task_label)] + img_filters
        else:
            filters = [('task', task_label),
                       ('space', space_label)] + img_filters
        imgs = get_bids_files(derivatives_path, modality_folder='func',file_tag='bold', file_type='nii*',sub_label=sub_label, filters=filters)
        # If there is more than one file for the same (ses, run), likely we
        # have an issue of underspecification of filters.
        run_check_list = []
        # If more than one run is present the run field is mandatory in BIDS
        # as well as the ses field if more than one session is present.
        if len(imgs) > 1:
            for img in imgs:
                img_dict = parse_bids_filename(img)
                if (
                    '_ses-' in img_dict['file_basename']
                    and '_run-' in img_dict['file_basename']
                ):
                    if (img_dict['ses'], img_dict['run']) in run_check_list:
                        raise ValueError(
                            'More than one nifti image found '
                            'for the same run %s and session %s. '
                            'Please verify that the '
                            'desc_label and space_label labels '
                            'corresponding to the BIDS spec '
                            'were correctly specified.' %
                            (img_dict['run'], img_dict['ses']))
                    else:
                        run_check_list.append((img_dict['ses'],
                                               img_dict['run']))

                elif '_ses-' in img_dict['file_basename']:
                    if img_dict['ses'] in run_check_list:
                        raise ValueError(
                            'More than one nifti image '
                            'found for the same ses %s, while '
                            'no additional run specification present'
                            '. Please verify that the desc_label and '
                            'space_label labels '
                            'corresponding to the BIDS spec '
                            'were correctly specified.' %
                            img_dict['ses'])
                    else:
                        run_check_list.append(img_dict['ses'])

                elif '_run-' in img_dict['file_basename']:
                    if img_dict['run'] in run_check_list:
                        raise ValueError(
                            'More than one nifti image '
                            'found for the same run %s. '
                            'Please verify that the desc_label and '
                            'space_label labels '
                            'corresponding to the BIDS spec '
                            'were correctly specified.' %
                            img_dict['run'])
                    else:
                        run_check_list.append(img_dict['run'])
        

        # Get events and extra confounds
        filters = [('task', task_label)]
        for img_filter in img_filters:
            if img_filter[0] in ['acq', 'rec', 'run']:
                filters.append(img_filter)

        # Get events files
        events = get_bids_files(dataset_path, modality_folder='func',
                                file_tag='events', file_type='tsv',
                                sub_label=sub_label, filters=filters)
        if events:
            if len(events) != len(imgs):
                continue
            events = [pd.read_csv(event, sep='\t', index_col=None)
                      for event in events]
        else:
            continue

        # Get confounds. If not found it will be assumed there are none.
        # If there are confounds, they are assumed to be present for all runs.
        confounds = get_bids_files(derivatives_path, modality_folder='func',
                                   file_tag='desc-confounds*',
                                   file_type='tsv', sub_label=sub_label,
                                   filters=filters)

        if confounds:
            if len(confounds) != len(imgs):
                import pdb
                pdb.set_trace()
                continue
                continue
            confounds = [pd.read_csv(c, sep='\t', index_col=None)
                         for c in confounds]
        models.append(model)
        models_run_imgs.append(imgs)
        models_events.append(events)
        models_confounds.append(confounds)
        
    return models, models_run_imgs, models_events, models_confounds


