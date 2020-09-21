# -*- coding: utf-8 -*- 

import hbayesdm.models

import numpy as np
import pandas as pd

from pathlib import Path

import nibabel as nib

import bids
from bids import BIDSLayout
from tqdm import tqdm

import nilearn as nil
from nilearn.glm.first_level import first_level_from_bids


def prepare_events(params, funcs):
    import hbayesdm.models
    from nilearn.glm.first_level.hemodynamic_models import compute_regressor
    from sklearn.preprocessing import minmax_scale

    layout = BIDSLayout(params['data_dir'], derivatives=True)
    t_r = layout.get_tr()
    events = layout.get(suffix='events', extension='tsv')
    image_sample = nib.load(
        layout.derivatives['fMRIPrep'].get(
            return_type='file',
            suffix='bold',
            extension='nii.gz')[0]
    )
    frame_time = image_sample.shape[-1]
    
    df_events_list = [event.get_df() for event in events]
    df_events_info = [event.get_entities() for event in events]
    
    if len(df_events_list) != len(df_events_info):
        assert()
    
    funcs[0](params, df_events_list, df_events_info)
    
    dm_model = getattr(hbayesdm.models, params['dm_model'])(
        data=pd.concat(df_events_list), ncore=params['ncore'])
    
    funcs[1](params, df_events_list, dm_model.all_ind_pars)
    
    frame_times = np.linspace(0,
                              t_r * frame_time,
                              t_r * frame_time + 1)
    
    df_events = pd.concat(df_events_list)
    signals = []
    for name0, group0 in df_events.groupby(['subjID']):
        signal_subject = []
        for name1, group1 in df_events.groupby(['run']):
            exp_condition = np.array(group1[['onset', 'duration', 'modulation']]).T

            signal, name = compute_regressor(
                exp_condition=exp_condition,
                hrf_model=params['hrf_model'],
                frame_times=frame_times)
            signal_subject.append(signal)
        
        signal_subject = np.array(signal_subject)
        reshape_target = signal_subject.shape
        
        normalized_signal = minmax_scale(signal_subject.flatten(), axis=0, copy=True)
        normalized_signal = normalized_signal.reshape(reshape_target)
        signals.append(normalized_signal)

    return df_events, np.array(signals)
