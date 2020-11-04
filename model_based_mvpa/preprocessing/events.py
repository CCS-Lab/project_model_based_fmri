#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.02
"""

import os
from pathlib import Path
import bids
from bids import BIDSLayout, BIDSValidator

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

from nilearn.glm.first_level.hemodynamic_models import compute_regressor
import nibabel as nib
from tqdm import tqdm

import hbayesdm.models
import time

import logging


logging.basicConfig(level=logging.INFO)


def adjust_event_columns(df_events_list, df_events_info):
    new_df_list = df_events_list.copy()
    for i in range(len(new_df_list)):
        new_df_list[i]['run'] = df_events_info[i]['run']
        new_df_list[i]['subjID'] = df_events_info[i]['subject']
        new_df_list[i]['gamble'] = new_df_list[i]['respcat'].apply(lambda x: 1 if x == 1 else 0)
        new_df_list[i]['cert'] = 0 # certain..?
        
    return new_df_list


def calculate_modulation(df_events_list, latent_params):
    new_df_list = df_events_list.copy()
    for i in range(len(new_df_list)):
        idx = new_df_list[i].iloc[0]['subjID']
        new_df_list[i]['rho'] = latent_params.loc[idx]['rho']
        new_df_list[i]['lambda'] = latent_params.loc[idx]['lambda']
        new_df_list[i]['modulation'] = \
            (new_df_list[i]['gain'] ** new_df_list[i]['rho']) \
            - (new_df_list[i]['lambda'] * (new_df_list[i]['loss'] ** new_df_list[i]['rho']))
        
    return new_df_list


def preprocess_events(root, dm_model, funcs,
                      hrf_model='glover',
                      save_path=None,
                      save=True,
                      single_file=True,
                      ncore=os.cpu_count(),
                      time_check=True):
    
    pbar = tqdm(total=6)
    s = time.time()

    pbar.set_description('loading bids dataset..'.center(40))
    layout = BIDSLayout(root, derivatives=True)
    t_r = layout.get_tr()
    events = layout.get(suffix='events', extension='tsv')
    image_sample = nib.load(
        layout.derivatives['fMRIPrep'].get(
            return_type='file',
            suffix='bold',
            extension='nii.gz')[0]
    )
    n_scans = image_sample.shape[-1]
    
    df_events_list = [event.get_df() for event in events]
    df_events_info = [event.get_entities() for event in events]

    if len(df_events_list) != len(df_events_info):
        assert()
    pbar.update(1)

    pbar.set_description('adjusting event file columns..'.center(40))
    df_events_list = funcs[0](df_events_list, df_events_info)
    pbar.update(1)
    
    pbar.set_description('hbayesdm doing (model: %s)..'.center(40) % dm_model)
    dm_model = getattr(hbayesdm.models, dm_model)(
        data=pd.concat(df_events_list), ncore=ncore)
    pbar.update(1)

    pbar.set_description('calculating modulation..'.center(40))
    df_events_list = funcs[1](df_events_list, dm_model.all_ind_pars)
    pbar.update(1)
    
    pbar.set_description('modulation signal making..'.center(40))
    frame_times = t_r * (np.arange(n_scans) + t_r/2)
    
    df_events = pd.concat(df_events_list)
    signals = []
    for name0, group0 in df_events.groupby(['subjID']):
        signal_subject = []
        for name1, group1 in group0.groupby(['run']):
            exp_condition = group1[['onset', 'duration', 'modulation']].to_numpy().T

            signal, name = compute_regressor(
                exp_condition=exp_condition,
                hrf_model=hrf_model,
                frame_times=frame_times)
            signal_subject.append(signal)
        
        signal_subject = np.array(signal_subject)
        reshape_target = signal_subject.shape
        
        normalized_signal = minmax_scale(signal_subject.flatten(), axis=0, copy=True)
        normalized_signal = normalized_signal.reshape(-1, n_scans, 1)
        signals.append(normalized_signal)
    signals = np.array(signals)
    pbar.update(1)
    
    pbar.set_description('modulation signal making..'.center(40))
    if save:
        if save_path is None:
            sp = Path(layout.derivatives['fMRIPrep'].root) / 'data'
        else:
            sp = Path(save_path)
            
        if not sp.exists():
            sp.mkdir()
        
        if single_file:
            np.save(sp / 'y.npy', signals)
        else:
            for i in range(signals.shape[0]):
                np.save(sp / f'y_{i+1}.npy', signals[i])
    pbar.update(1)
    pbar.set_description('events preproecssing done!'.center(40))
    
    if time_check:
        e = time.time()
        logging.info(f'time elapsed: {e-s} seconds')
        
    return dm_model, df_events, signals
