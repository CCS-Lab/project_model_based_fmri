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


def example_adjust_event_columns(df_events_list, df_events_info):
    new_df_list = df_events_list.copy()
    for i in range(len(new_df_list)):
        new_df_list[i]['run'] = df_events_info[i]['run']
        new_df_list[i]['subjID'] = df_events_info[i]['subject']
        new_df_list[i]['gamble'] = new_df_list[i]['respcat'].apply(lambda x: 1 if x == 1 else 0)
        new_df_list[i]['cert'] = 0 # certain..?
    return new_df_list


def example_calculate_modulation(df_events_list, latent_params):
    new_df_list = df_events_list.copy()
    for i in range(len(new_df_list)):
        idx = new_df_list[i].iloc[0]['subjID']
        new_df_list[i]['rho'] = latent_params.loc[idx]['rho']
        new_df_list[i]['lambda'] = latent_params.loc[idx]['lambda']
        new_df_list[i]['modulation'] = \
            (new_df_list[i]['gain'] ** new_df_list[i]['rho']) \
            - (new_df_list[i]['lambda'] * (new_df_list[i]['loss'] ** new_df_list[i]['rho']))
        
    return new_df_list

def get_indiv_par(subjId, all_ind_pars, parameter_names):
    try:
        ind_pars = all_ind_pars.loc[subjId]
    except:
        ind_pars = all_ind_pars.loc[int(subjId)]
        
    return {name : ind_pars[name] for name in parameter_names}

def get_time_mask(cond_func, df_events, time_len, TR, use_duration=False):
    
    df_events = df_events.sort_values(by='onset')
    onsets = np.array(df_events['onset'])
    if use_duration:
        durations = np.array(df['duration'])
    else:
        durations = np.array(list(df_events['onset'][1:])+[time_len*TR]) - onsets
    
    mask = [cond_func(row) for _, row in df_events.iterrows()]
    time_mask = np.zeros(time_len)
    
    for flag,onset,duration in zip(mask,onsets,durations):
            if flag:
                time_mask[int(onset/TR):int((onset+duration)/TR)] = 1
        
    return time_mask

def preprocess_event(prep_func, cond_func, df_events, event_infos, **kwargs):
    
    new_datarows = []
    df_events = df_events.sort_values(by='onset')
    
    for _, row in df_events.iterrows():
        if cond_func is not None and cond_func(row):
            new_datarows.append(prep_func(row,event_infos,**kwargs))
    
    new_datarows = pd.concat(new_datarows, axis=1, keys=[s.name for s in new_datarows]).transpose()
    
    return new_datarows

def preprocess_events(root, dm_model,   
                      latent_func,par_names,
                      prep_func=lambda x : x,
                      cond_func=lambda _ : True,
                      df_events=None,
                      all_ind_pars=None,
                      use_duration=False,
                      layout=None,
                      hrf_model='glover',
                      save_path=None,
                      save=True,
                      ncore=4,
                      time_check=True,
                      **kwargs # hBayesDM fitting argument):
    
    pbar = tqdm(total=6)
    s = time.time()
################################################################################
# load bids layout
    if save_path is None:
        sp = Path(layout.derivatives['fMRIPrep'].root) / 'data'
    else:
        sp = Path(save_path)
    
    if not sp.exists():
            sp.mkdir()
            
    pbar.set_description('loading bids dataset..'.center(40))

    if layout is None:
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
    event_infos_list = [event.get_entities() for event in events]
    pbar.update(1)
    
    pbar.set_description('adjusting event file columns..'.center(40))
    
    df_events_list = [preprocess_event(prep_func, cond_func, df_events, event_infos) for df_events, event_infos in zip(df_events_list, event_infos_list)]
    pbar.update(1)

    pbar.set_description('calculating time mask..'.center(40))
    
    time_masks = []
    for name0, group0 in pd.concat(df_events_list).groupby(['subjID']):
        time_mask_subject = []
        for name1, group1 in group0.groupby(['run']):
            time_mask_subject.append(get_time_mask(cond_func, group1 , n_scans, t_r, use_duration))
            
        
        time_mask_subject = np.array(time_mask_subject)
        time_masks.append(time_mask_subject)
   
    time_masks = np.array(time_masks)

    if save:
        np.save(sp / 'y_mask.npy',time_masks)
        
    pbar.update(1)
    pbar.set_description('time mask preproecssing done!'.center(40))
    
    if df_events is None:
        
        if all_ind_pars is None:
            pbar.set_description('hbayesdm doing (model: %s)..'.center(40) % dm_model)
            dm_model = getattr(hbayesdm.models, dm_model)(
                data=pd.concat(df_events_list), ncore=ncore, **kwargs)
            pbar.update(1)
            all_ind_pars = dm_model.all_ind_pars
            if save:
                all_ind_pars.to_csv(sp / 'all_ind_pars.tsv', sep="\t")
                
            # all_ind_pars = pd.read_csv(sp / 'all_ind_pars.tsv',sep = '\t',index_col='Unnamed: 0')
        else:
            pbar.update(1)
            
        pbar.set_description('calculating modulation..'.center(40))


        df_events_list =[preprocess_event(latent_func, cond_func, df_events, event_infos,
                                          **get_indiv_par(event_infos['subject'], all_ind_pars,par_names))for df_events, event_infos in zip(df_events_list, event_infos_list)]
        df_events = pd.concat(df_events_list)
        pbar.update(1)
    else:
        pbar.update(2)
    
    pbar.set_description('modulation signal making..'.center(40))
    frame_times = t_r * (np.arange(n_scans) + t_r/2)
    
    
    
    signals = []
    for name0, group0 in df_events.groupby(['subjID']):
        signal_subject = []
        for name1, group1 in group0.groupby(['run']):
            exp_condition = group1[['onset', 'duration', 'modulation']].to_numpy().T
            exp_condition = exp_condition.astype(float)
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
    
    if save:
        np.save(sp / 'y.npy', signals)
        
    pbar.update(1)
    pbar.set_description('events preproecssing done!'.center(40))
    
    e = time.time()
    logging.info(f'time elapsed: {(e-s) / 60:.2f} minutes')
    
    
    return dm_model, df_events, signals, time_masks
